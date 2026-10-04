// Distilled from pinned LlamaModel, LlamaDecoderLayer, LlamaAttention,
// LlamaMLP, LlamaRMSNorm and default rotary arithmetic (sources.json).
// Unpadded prefill to logits. No decode cache, sampling, loss or kernel rounding.
import {
  add,
  affine,
  causalMask,
  Ctx,
  embedding,
  matmul,
  mul,
  permute,
  reshape,
  rmsNorm,
  scale,
  silu,
  softmax,
  tensor,
  type Tensor,
} from "../../../tnsr/tensor";
import * as f from "../facts";
export type LlamaConfig = {
  vocab: number;
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  inner: number;
  normEps: number;
  theta: number;
};
export const realConfig: LlamaConfig = {
  vocab: f.vocab.value,
  hidden: f.hidden.value,
  layers: f.layers.value,
  heads: f.heads.value,
  kvHeads: f.kvHeads.value,
  headDim: f.headDim.value,
  inner: f.mlpWidth.value,
  normEps: f.normEps.value,
  theta: f.theta.value,
};
function linear(c: Ctx, name: string, x: Tensor, out: number) {
  return affine(x, c.param(`${name}.weight`, [out, x.shape.at(-1)!]), null, "out_in");
}
export function LlamaRMSNorm(c: Ctx, name: string, x: Tensor, cfg: LlamaConfig) {
  return rmsNorm(x, c.param(`${name}.weight`, [x.shape.at(-1)!]), cfg.normEps);
}
/** Half-split pairs (i, i+D/2), with absolute prefill positions. */
export function LlamaRotaryEmbedding(x: Tensor, theta: number): Tensor {
  if (x.data === null) return { shape: x.shape, data: null };
  const [b, h, s, d] = x.shape;
  const out = new Float32Array(x.data.length);
  for (let batch = 0; batch < b; batch++)
    for (let head = 0; head < h; head++)
      for (let pos = 0; pos < s; pos++)
        for (let i = 0; i < d / 2; i++) {
          const at = ((batch * h + head) * s + pos) * d + i;
          const a = pos * theta ** ((-2 * i) / d);
          const co = Math.cos(a),
            si = Math.sin(a);
          const u = x.data[at],
            v = x.data[at + d / 2];
          out[at] = u * co - v * si;
          out[at + d / 2] = v * co + u * si;
        }
  return tensor(x.shape, out);
}
export function repeatKv(x: Tensor, groups: number): Tensor {
  const [b, h, s, d] = x.shape;
  const shape = [b, h * groups, s, d];
  if (x.data === null) return { shape, data: null };
  const out = new Float32Array(b * h * groups * s * d);
  for (let batch = 0; batch < b; batch++)
    for (let head = 0; head < h * groups; head++) {
      const src = (batch * h + Math.floor(head / groups)) * s * d;
      out.set(x.data.subarray(src, src + s * d), (batch * h * groups + head) * s * d);
    }
  return tensor(shape, out);
}
export function LlamaAttention(c: Ctx, name: string, id: string, x: Tensor, cfg: LlamaConfig): Tensor {
  const [b, s] = x.shape;
  const heads = (t: Tensor, h: number) => permute(reshape(t, [b, s, h, cfg.headDim]), [0, 2, 1, 3]);
  const q0 = reshape(linear(c, `${name}.q_proj`, x, cfg.heads * cfg.headDim), [b, s, cfg.heads, cfg.headDim]);
  const k0 = reshape(linear(c, `${name}.k_proj`, x, cfg.kvHeads * cfg.headDim), [b, s, cfg.kvHeads, cfg.headDim]);
  const qRaw = c.mark(`${id}.q_raw`, permute(q0, [0, 2, 1, 3]));
  const kRaw = c.mark(`${id}.k_raw`, permute(k0, [0, 2, 1, 3]));
  const q = c.mark(`${id}.q`, LlamaRotaryEmbedding(qRaw, cfg.theta));
  const k = c.mark(`${id}.k`, LlamaRotaryEmbedding(kRaw, cfg.theta));
  const v = c.mark(`${id}.v`, heads(linear(c, `${name}.v_proj`, x, cfg.kvHeads * cfg.headDim), cfg.kvHeads));
  const groups = cfg.heads / cfg.kvHeads;
  const score = (queries: Tensor, keys: Tensor) =>
    scale(matmul(queries, permute(repeatKv(keys, groups), [0, 1, 3, 2])), cfg.headDim ** -0.5);
  // Diagnostic comparisons only: the forward path uses rotated scores.
  c.mark(`${id}.scores_raw`, score(qRaw, kRaw));
  c.mark(`${id}.k_repeated`, repeatKv(k, groups));
  const vRepeated = c.mark(`${id}.v_repeated`, repeatKv(v, groups));
  const scores = c.mark(`${id}.scores`, score(q, k));
  const weights = c.mark(`${id}.weights`, softmax(causalMask(scores)));
  const mix = matmul(weights, vRepeated);
  const merged = reshape(permute(mix, [0, 2, 1, 3]), [b, s, cfg.heads * cfg.headDim]);
  return c.mark(`${id}.out`, linear(c, `${name}.o_proj`, merged, cfg.hidden));
}
export function LlamaMLP(c: Ctx, name: string, id: string, x: Tensor, cfg: LlamaConfig) {
  const gate = c.mark(`${id}.gate`, silu(linear(c, `${name}.gate_proj`, x, cfg.inner)));
  const up = c.mark(`${id}.up`, linear(c, `${name}.up_proj`, x, cfg.inner));
  return c.mark(`${id}.out`, linear(c, `${name}.down_proj`, mul(gate, up), cfg.hidden));
}
export function LlamaDecoderLayer(c: Ctx, index: number, x: Tensor, cfg: LlamaConfig) {
  const name = `model.layers.${index}`,
    id = `h${index}`;
  const y = add(
    x,
    LlamaAttention(c, `${name}.self_attn`, `${id}.attn`, LlamaRMSNorm(c, `${name}.input_layernorm`, x, cfg), cfg),
  );
  return c.mark(
    `${id}.out`,
    add(y, LlamaMLP(c, `${name}.mlp`, `${id}.mlp`, LlamaRMSNorm(c, `${name}.post_attention_layernorm`, y, cfg), cfg)),
  );
}
export function forward(c: Ctx, cfg: LlamaConfig, ids: readonly (readonly number[])[]) {
  let x = c.mark("embed", embedding(c.param("model.embed_tokens.weight", [cfg.vocab, cfg.hidden]), ids));
  for (let i = 0; i < cfg.layers; i++) x = LlamaDecoderLayer(c, i, x, cfg);
  x = c.mark("final_norm", LlamaRMSNorm(c, "model.norm", x, cfg));
  // tie_word_embeddings=true: reuse the exact embedding tensor.
  return c.mark("logits", affine(x, c.param("model.embed_tokens.weight", [cfg.vocab, cfg.hidden]), null, "out_in"));
}
