// Distilled from pinned GlmMoeDsa{Model,Attention,Indexer,MoE,Experts,TopkRouter,MLP}.
// See sources.json for Transformers commit. Text decoder prefill to logits only.
// No training loss, MTP, quantized storage, incremental cache or GPU kernels.
import * as r from "../../../tnsr/tensor";
import * as f from "../facts";
export type Config = {
  hidden: number;
  vocab: number;
  heads: number;
  qRank: number;
  kvRank: number;
  nope: number;
  rope: number;
  v: number;
  indexHeads: number;
  indexDim: number;
  indexTopK: number;
  experts: number;
  topK: number;
  shared: number;
  denseWidth: number;
  expertWidth: number;
  eps: number;
  theta: number;
  scale: number;
  indexTypes: readonly string[];
  mlpTypes: readonly string[];
};
export const REAL: Config = {
  hidden: f.hidden.value,
  vocab: f.vocab.value,
  heads: f.heads.value,
  qRank: f.qRank.value,
  kvRank: f.kvRank.value,
  nope: f.nope.value,
  rope: f.rope.value,
  v: f.valueDim.value,
  indexHeads: f.indexHeads.value,
  indexDim: f.indexDim.value,
  indexTopK: f.indexTopK.value,
  experts: f.experts.value,
  topK: f.topK.value,
  shared: f.shared.value,
  denseWidth: f.denseWidth.value,
  expertWidth: f.expertWidth.value,
  eps: f.eps.value,
  theta: f.theta.value,
  scale: f.scale.value,
  indexTypes: f.indexTypes.value,
  mlpTypes: f.mlpTypes.value,
};
export const TOY: Config = f.toyConfig.value;
function linear(ctx: r.Ctx, x: r.Tensor, name: string, out: number) {
  return r.affine(x, ctx.param(name + ".weight", [out, x.shape.at(-1)!]), null, "out_in");
}
function norm(ctx: r.Ctx, x: r.Tensor, name: string, eps: number) {
  return r.rmsNorm(x, ctx.param(name + ".weight", [x.shape.at(-1)!]), eps);
}
// Upstream interleaved input pairs produce half-split output storage.
export function rotate(x: r.Tensor, theta: number): r.Tensor {
  const [b, s, h, d] = x.shape;
  if (!x.data) return r.tensor(x.shape, null);
  const out = new Float32Array(x.data.length);
  for (let bi = 0; bi < b; bi++)
    for (let pos = 0; pos < s; pos++)
      for (let head = 0; head < h; head++)
        for (let pair = 0; pair < d / 2; pair++) {
          const base = ((bi * s + pos) * h + head) * d,
            a = pos * theta ** ((-2 * pair) / d),
            u = x.data[base + 2 * pair],
            v = x.data[base + 2 * pair + 1];
          out[base + pair] = u * Math.cos(a) - v * Math.sin(a);
          out[base + d / 2 + pair] = v * Math.cos(a) + u * Math.sin(a);
        }
  return r.tensor(x.shape, out);
}
function repeatHeads(x: r.Tensor, heads: number): r.Tensor {
  const [b, s, , d] = x.shape;
  if (!x.data) return r.tensor([b, s, heads, d], null);
  const out = new Float32Array(b * s * heads * d);
  for (let i = 0; i < b * s; i++)
    for (let h = 0; h < heads; h++) out.set(x.data.subarray(i * d, (i + 1) * d), (i * heads + h) * d);
  return r.tensor([b, s, heads, d], out);
}
export function GlmMoeDsaIndexer(ctx: r.Ctx, c: Config, x: r.Tensor, qr: r.Tensor, p: string, id: string) {
  const [b, s] = x.shape;
  let q = r.reshape(linear(ctx, qr, p + ".wq_b", c.indexHeads * c.indexDim), [b, s, c.indexHeads, c.indexDim]);
  const raw = linear(ctx, x, p + ".wk", c.indexDim);
  let k = r.layerNorm(
    raw,
    ctx.param(p + ".k_norm.weight", [c.indexDim]),
    ctx.param(p + ".k_norm.bias", [c.indexDim]),
    f.indexEps.value,
  );
  k = r.reshape(k, [b, s, 1, c.indexDim]);
  const [qrp, qp] = r.splitLast(q, [c.rope, c.indexDim - c.rope]),
    [krp, kp] = r.splitLast(k, [c.rope, c.indexDim - c.rope]);
  q = r.concatLast([rotate(qrp, c.theta), qp]);
  k = r.concatLast([rotate(krp, c.theta), kp]);
  const weights = linear(ctx, x, p + ".weights_proj", c.indexHeads);
  const scoreShape = [b, s, s];
  let scores: r.Tensor;
  if (!q.data || !k.data || !weights.data) scores = r.tensor(scoreShape, null);
  else {
    const out = new Float32Array(b * s * s);
    for (let bi = 0; bi < b; bi++)
      for (let qi = 0; qi < s; qi++)
        for (let ki = 0; ki < s; ki++) {
          let total = 0;
          for (let h = 0; h < c.indexHeads; h++) {
            let dot = 0;
            for (let d = 0; d < c.indexDim; d++)
              dot +=
                q.data[((bi * s + qi) * c.indexHeads + h) * c.indexDim + d] * k.data[(bi * s + ki) * c.indexDim + d];
            total +=
              (Math.max(0, dot / Math.sqrt(c.indexDim)) * weights.data[(bi * s + qi) * c.indexHeads + h]) /
              Math.sqrt(c.indexHeads);
          }
          out[(bi * s + qi) * s + ki] = ki > qi ? -Infinity : total;
        }
    scores = r.tensor(scoreShape, out);
  }
  return ctx.mark(id + ".indices", r.topk(scores, Math.min(c.indexTopK, s)).indices);
}
export function GlmMoeDsaAttention(
  ctx: r.Ctx,
  c: Config,
  x: r.Tensor,
  p: string,
  id: string,
  previous: r.Tensor | null,
) {
  const [b, s] = x.shape;
  const qr = ctx.mark(
    id + ".qnorm",
    norm(ctx, linear(ctx, x, p + ".q_a_proj", c.qRank), p + ".q_a_layernorm", f.latentEps.value),
  );
  const q = r.reshape(linear(ctx, qr, p + ".q_b_proj", c.heads * (c.nope + c.rope)), [b, s, c.heads, c.nope + c.rope]);
  const [qp, qrot] = r.splitLast(q, [c.nope, c.rope]);
  const [latent, krot] = r.splitLast(linear(ctx, x, p + ".kv_a_proj_with_mqa", c.kvRank + c.rope), [c.kvRank, c.rope]);
  const kn = norm(ctx, latent, p + ".kv_a_layernorm", f.latentEps.value);
  const expanded = r.reshape(linear(ctx, kn, p + ".kv_b_proj", c.heads * (c.nope + c.v)), [
    b,
    s,
    c.heads,
    c.nope + c.v,
  ]);
  const [kp, v] = r.splitLast(expanded, [c.nope, c.v]);
  const query = r.permute(r.concatLast([qp, rotate(qrot, c.theta)]), [0, 2, 1, 3]);
  const key = r.permute(
    r.concatLast([kp, repeatHeads(rotate(r.reshape(krot, [b, s, 1, c.rope]), c.theta), c.heads)]),
    [0, 2, 1, 3],
  );
  let indices: r.Tensor;
  if (c.indexTypes[Number(id.slice(1))] === "full") indices = GlmMoeDsaIndexer(ctx, c, x, qr, p + ".indexer", id);
  else {
    if (!previous) throw new Error("Shared layer requires previous selection");
    indices = ctx.mark(id + ".indices", previous);
  }
  const scores = r.causalMask(r.scale(r.matmul(query, r.permute(key, [0, 1, 3, 2])), 1 / Math.sqrt(c.nope + c.rope)));
  let masked = scores;
  if (scores.data && indices.data) {
    const out = Float32Array.from(scores.data),
      count = indices.shape.at(-1)!;
    for (let bi = 0; bi < b; bi++)
      for (let h = 0; h < c.heads; h++)
        for (let qi = 0; qi < s; qi++)
          for (let ki = 0; ki < s; ki++) {
            const selected = indices.data.subarray((bi * s + qi) * count, (bi * s + qi + 1) * count);
            if (!selected.includes(ki)) out[((bi * c.heads + h) * s + qi) * s + ki] = -Infinity;
          }
    masked = r.tensor(scores.shape, out);
  }
  const att = ctx.mark(id + ".attention", r.softmax(masked));
  const output = r.reshape(r.permute(r.matmul(att, r.permute(v, [0, 2, 1, 3])), [0, 2, 1, 3]), [b, s, c.heads * c.v]);
  return { output: linear(ctx, output, p + ".o_proj", c.hidden), indices };
}
export function GlmMoeDsaMLP(ctx: r.Ctx, c: Config, x: r.Tensor, p: string, width: number) {
  return linear(
    ctx,
    r.mul(r.silu(linear(ctx, x, p + ".gate_proj", width)), linear(ctx, x, p + ".up_proj", width)),
    p + ".down_proj",
    c.hidden,
  );
}
export function GlmMoeDsaTopkRouter(
  ctx: r.Ctx,
  c: Config,
  x: r.Tensor,
  p: string,
  id: string,
  bias?: readonly number[],
) {
  const flat = r.reshape(x, [x.shape[0] * x.shape[1], c.hidden]);
  const logits = ctx.mark(id + ".router_logits", linear(ctx, flat, p, c.experts));
  const scores = r.sigmoid(logits),
    choice = r.add(scores, r.tensor([c.experts], bias ?? Array(c.experts).fill(0)));
  // Exact checkpoint has one group: group selection cannot exclude an expert.
  const indices = r.topk(choice, c.topK).indices;
  const selected = r.gatherLast(scores, indices);
  let weights = selected;
  if (selected.data) {
    const out = Float32Array.from(selected.data);
    for (let row = 0; row < out.length / c.topK; row++) {
      let total = 1e-20;
      for (let k = 0; k < c.topK; k++) total += out[row * c.topK + k];
      for (let k = 0; k < c.topK; k++) out[row * c.topK + k] = (out[row * c.topK + k] / total) * c.scale;
    }
    weights = r.tensor(selected.shape, out);
  }
  ctx.mark(id + ".route_weights", weights);
  ctx.mark(id + ".route_indices", indices);
  ctx.mark(id + ".route_scores", scores);
  return { indices, weights };
}
export function GlmMoeDsaExperts(ctx: r.Ctx, c: Config, x: r.Tensor, p: string, indices: r.Tensor, weights: r.Tensor) {
  const gu = ctx.param(p + ".gate_up_proj", [c.experts, 2 * c.expertWidth, c.hidden]);
  const down = ctx.param(p + ".down_proj", [c.experts, c.hidden, c.expertWidth]);
  if (!x.data || !gu.data || !down.data || !indices.data || !weights.data) return r.tensor(x.shape, null);
  const out = new Float32Array(x.data.length),
    rows = x.data.length / c.hidden;
  for (let row = 0; row < rows; row++)
    for (let k = 0; k < c.topK; k++) {
      const e = indices.data[row * c.topK + k],
        a = new Float32Array(c.expertWidth);
      for (let j = 0; j < c.expertWidth; j++) {
        let gate = 0,
          up = 0;
        for (let d = 0; d < c.hidden; d++) {
          gate += x.data[row * c.hidden + d] * gu.data[(e * 2 * c.expertWidth + j) * c.hidden + d];
          up += x.data[row * c.hidden + d] * gu.data[(e * 2 * c.expertWidth + c.expertWidth + j) * c.hidden + d];
        }
        a[j] = (gate / (1 + Math.exp(-gate))) * up;
      }
      for (let d = 0; d < c.hidden; d++) {
        let sum = 0;
        for (let j = 0; j < c.expertWidth; j++) sum += a[j] * down.data[(e * c.hidden + d) * c.expertWidth + j];
        out[row * c.hidden + d] += sum * weights.data[row * c.topK + k];
      }
    }
  return r.tensor(x.shape, out);
}
export function GlmMoeDsaMoE(ctx: r.Ctx, c: Config, x: r.Tensor, p: string, id: string, bias?: readonly number[]) {
  const { indices, weights } = GlmMoeDsaTopkRouter(ctx, c, x, p + ".gate", id, bias);
  return r.add(
    GlmMoeDsaExperts(ctx, c, x, p + ".experts", indices, weights),
    GlmMoeDsaMLP(ctx, c, x, p + ".shared_experts", c.expertWidth * c.shared),
  );
}
export function forward(
  ctx: r.Ctx,
  c: Config,
  ids: readonly (readonly number[])[],
  biases: Record<string, readonly number[]> = {},
) {
  let x = ctx.mark("embedding", r.embedding(ctx.param("model.embed_tokens.weight", [c.vocab, c.hidden]), ids));
  let previous: r.Tensor | null = null;
  for (let i = 0; i < c.mlpTypes.length; i++) {
    const p = `model.layers.${i}`,
      id = `l${i}`,
      a = GlmMoeDsaAttention(ctx, c, norm(ctx, x, p + ".input_layernorm", c.eps), p + ".self_attn", id, previous);
    previous = a.indices;
    x = r.add(x, a.output);
    const xn = norm(ctx, x, p + ".post_attention_layernorm", c.eps);
    x = ctx.mark(
      id + ".output",
      r.add(
        x,
        c.mlpTypes[i] === "dense"
          ? GlmMoeDsaMLP(ctx, c, xn, p + ".mlp", c.denseWidth)
          : GlmMoeDsaMoE(ctx, c, xn, p + ".mlp", id, biases[p + ".mlp.gate.e_score_correction_bias"]),
      ),
    );
  }
  return ctx.mark("logits", linear(ctx, norm(ctx, x, "model.norm", c.eps), "lm_head", c.vocab));
}
