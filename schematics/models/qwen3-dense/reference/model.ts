// Distilled from pinned Qwen3Model, Qwen3DecoderLayer, Qwen3Attention,
// Qwen3MLP, Qwen3RMSNorm and default rotary arithmetic (sources.json).
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
} from "../../../kit/ref/tensor";
import * as f from "../facts";
export type QwenConfig = {
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
export const realConfig: QwenConfig = {
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
export function Qwen3RMSNorm(c: Ctx, name: string, x: Tensor, cfg: QwenConfig) {
  return rmsNorm(x, c.param(`${name}.weight`, [x.shape.at(-1)!]), cfg.normEps);
}
/** Half-split pairs (i, i+D/2), with absolute prefill positions. */
export function Qwen3RotaryEmbedding(x: Tensor, theta: number): Tensor {
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
export function Qwen3Attention(c: Ctx, name: string, id: string, x: Tensor, cfg: QwenConfig): Tensor {
  const [b, s] = x.shape;
  const heads = (t: Tensor, h: number) => permute(reshape(t, [b, s, h, cfg.headDim]), [0, 2, 1, 3]);
  const q0 = reshape(linear(c, `${name}.q_proj`, x, cfg.heads * cfg.headDim), [b, s, cfg.heads, cfg.headDim]);
  const k0 = reshape(linear(c, `${name}.k_proj`, x, cfg.kvHeads * cfg.headDim), [b, s, cfg.kvHeads, cfg.headDim]);
  const q = c.mark(
    `${id}.q`,
    Qwen3RotaryEmbedding(permute(Qwen3RMSNorm(c, `${name}.q_norm`, q0, cfg), [0, 2, 1, 3]), cfg.theta),
  );
  const k = c.mark(
    `${id}.k`,
    Qwen3RotaryEmbedding(permute(Qwen3RMSNorm(c, `${name}.k_norm`, k0, cfg), [0, 2, 1, 3]), cfg.theta),
  );
  const v = c.mark(`${id}.v`, heads(linear(c, `${name}.v_proj`, x, cfg.kvHeads * cfg.headDim), cfg.kvHeads));
  const groups = cfg.heads / cfg.kvHeads;
  const scores = c.mark(
    `${id}.scores`,
    scale(matmul(q, permute(repeatKv(k, groups), [0, 1, 3, 2])), cfg.headDim ** -0.5),
  );
  const weights = c.mark(`${id}.weights`, softmax(causalMask(scores)));
  const mix = matmul(weights, repeatKv(v, groups));
  const merged = reshape(permute(mix, [0, 2, 1, 3]), [b, s, cfg.heads * cfg.headDim]);
  return c.mark(`${id}.out`, linear(c, `${name}.o_proj`, merged, cfg.hidden));
}
export function Qwen3MLP(c: Ctx, name: string, id: string, x: Tensor, cfg: QwenConfig) {
  const gate = c.mark(`${id}.gate`, silu(linear(c, `${name}.gate_proj`, x, cfg.inner)));
  const up = c.mark(`${id}.up`, linear(c, `${name}.up_proj`, x, cfg.inner));
  return c.mark(`${id}.out`, linear(c, `${name}.down_proj`, mul(gate, up), cfg.hidden));
}
export function Qwen3DecoderLayer(c: Ctx, index: number, x: Tensor, cfg: QwenConfig) {
  const name = `model.layers.${index}`,
    id = `h${index}`;
  const y = add(
    x,
    Qwen3Attention(c, `${name}.self_attn`, `${id}.attn`, Qwen3RMSNorm(c, `${name}.input_layernorm`, x, cfg), cfg),
  );
  return c.mark(
    `${id}.out`,
    add(y, Qwen3MLP(c, `${name}.mlp`, `${id}.mlp`, Qwen3RMSNorm(c, `${name}.post_attention_layernorm`, y, cfg), cfg)),
  );
}
export function forward(c: Ctx, cfg: QwenConfig, ids: readonly (readonly number[])[]) {
  let x = c.mark("embed", embedding(c.param("model.embed_tokens.weight", [cfg.vocab, cfg.hidden]), ids));
  for (let i = 0; i < cfg.layers; i++) x = Qwen3DecoderLayer(c, i, x, cfg);
  x = c.mark("final_norm", Qwen3RMSNorm(c, "model.norm", x, cfg));
  return c.mark("logits", linear(c, "lm_head", x, cfg.vocab));
}
