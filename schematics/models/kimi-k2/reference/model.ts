// Distilled from the publisher DeepseekV3Attention, MoEGate, DeepseekV3MoE,
// DeepseekV3MLP, DecoderLayer, Model and CausalLM at ../sources.json's reference pin.
// Eval eager, unpadded full sequences to logits; no decode cache, sampling, losses
// or FP8 simulation. Adjacent-channel RoPE shuffle and YaRN scale are preserved.
import {
  add,
  affine,
  causalMask,
  concatLast,
  embedding,
  gatherLast,
  matmul,
  mul,
  permute,
  reshape,
  rmsNorm,
  scale,
  sigmoid,
  silu,
  softmax,
  splitLast,
  tensor,
  topk,
  type Ctx,
  type Tensor,
} from "../../../kit/ref/tensor";

export type KimiConfig = Readonly<{
  vocab: number;
  hidden: number;
  layers: number;
  heads: number;
  qRank: number;
  kvRank: number;
  nope: number;
  rope: number;
  v: number;
  dense: number;
  expert: number;
  experts: number;
  shared: number;
  topK: number;
  firstDense: number;
  eps: number;
  theta: number;
  factor: number;
  original: number;
  betaFast: number;
  betaSlow: number;
  mscale: number;
  allDim: number;
  scale: number;
}>;

export function yarnFrequency(pair: number, c: KimiConfig): number {
  const correction = (rotations: number) =>
    (c.rope * Math.log(c.original / (rotations * 2 * Math.PI))) / (2 * Math.log(c.theta));
  const low = Math.max(Math.floor(correction(c.betaFast)), 0);
  let high = Math.min(Math.ceil(correction(c.betaSlow)), c.rope - 1);
  if (low === high) high += 0.001;
  const ramp = Math.max(0, Math.min(1, (pair - low) / (high - low)));
  const extra = c.theta ** ((-2 * pair) / c.rope);
  return extra * (1 - ramp + ramp / c.factor);
}
export function yarnMscale(factor: number, coefficient: number): number {
  return factor <= 1 ? 1 : 0.1 * coefficient * Math.log(factor) + 1;
}
export function attentionScale(c: KimiConfig): number {
  return (c.nope + c.rope) ** -0.5 * yarnMscale(c.factor, c.allDim) ** 2;
}
export function rotary(x: Tensor, c: KimiConfig): Tensor {
  const [batch, heads, seq, width] = x.shape;
  if (width !== c.rope || width % 2) throw new RangeError("rotary width must be even and match config");
  if (x.data === null) return { shape: x.shape, data: null };
  const out = new Float32Array(x.data.length);
  const amplitude = yarnMscale(c.factor, c.mscale) / yarnMscale(c.factor, c.allDim);
  for (let b = 0; b < batch; b++)
    for (let h = 0; h < heads; h++)
      for (let s = 0; s < seq; s++) {
        const base = ((b * heads + h) * seq + s) * width;
        for (let p = 0; p < width / 2; p++) {
          const angle = s * yarnFrequency(p, c),
            co = Math.cos(angle) * amplitude,
            si = Math.sin(angle) * amplitude;
          const a = x.data[base + 2 * p],
            v = x.data[base + 2 * p + 1];
          out[base + p] = a * co - v * si;
          out[base + p + width / 2] = v * co + a * si;
        }
      }
  return tensor(x.shape, out);
}
function expandHeads(x: Tensor, heads: number): Tensor {
  const [batch, one, seq, width] = x.shape;
  if (one !== 1) throw new RangeError("only shared positional key can be expanded");
  if (x.data === null) return { shape: [batch, heads, seq, width], data: null };
  const out = new Float32Array(batch * heads * seq * width);
  for (let b = 0; b < batch; b++)
    for (let h = 0; h < heads; h++)
      out.set(x.data.subarray(b * seq * width, (b + 1) * seq * width), (b * heads + h) * seq * width);
  return tensor([batch, heads, seq, width], out);
}
function linear(ctx: Ctx, name: string, x: Tensor, out: number): Tensor {
  return affine(x, ctx.param(`${name}.weight`, [out, x.shape.at(-1)!]), null, "out_in");
}
function norm(ctx: Ctx, name: string, x: Tensor, eps: number): Tensor {
  return rmsNorm(x, ctx.param(`${name}.weight`, [x.shape.at(-1)!]), eps);
}
export function DeepseekV3Attention(ctx: Ctx, name: string, id: string, x: Tensor, c: KimiConfig): Tensor {
  const [batch, seq] = x.shape;
  const heads = (x: Tensor, width: number) => permute(reshape(x, [batch, seq, c.heads, width]), [0, 2, 1, 3]);
  const qa = ctx.mark(`${id}.qa`, linear(ctx, `${name}.q_a_proj`, x, c.qRank));
  const qn = ctx.mark(`${id}.qnorm`, norm(ctx, `${name}.q_a_layernorm`, qa, 1e-6));
  const qb = ctx.mark(`${id}.qb`, linear(ctx, `${name}.q_b_proj`, qn, c.heads * (c.nope + c.rope)));
  const [qNope, qPe] = splitLast(heads(qb, c.nope + c.rope), [c.nope, c.rope]);
  const ka = ctx.mark(`${id}.kva`, linear(ctx, `${name}.kv_a_proj_with_mqa`, x, c.kvRank + c.rope));
  const [latent, kPe] = splitLast(ka, [c.kvRank, c.rope]);
  const cn = ctx.mark(`${id}.latent`, norm(ctx, `${name}.kv_a_layernorm`, latent, 1e-6));
  const kv = ctx.mark(`${id}.kvb`, linear(ctx, `${name}.kv_b_proj`, cn, c.heads * (c.nope + c.v)));
  const [kNope, v] = splitLast(heads(kv, c.nope + c.v), [c.nope, c.v]);
  const qp = ctx.mark(`${id}.qrope`, rotary(qPe, c));
  const kp = ctx.mark(`${id}.krope`, rotary(permute(reshape(kPe, [batch, seq, 1, c.rope]), [0, 2, 1, 3]), c));
  const q = ctx.mark(`${id}.q`, concatLast([qNope, qp]));
  const k = ctx.mark(`${id}.k`, concatLast([kNope, expandHeads(kp, c.heads)]));
  ctx.mark(`${id}.v`, v);
  const scores = ctx.mark(`${id}.scores`, scale(matmul(q, permute(k, [0, 1, 3, 2])), attentionScale(c)));
  const weights = ctx.mark(`${id}.weights`, softmax(causalMask(scores)));
  const mixed = ctx.mark(`${id}.mixed`, matmul(weights, v));
  const merged = ctx.mark(`${id}.merged`, reshape(permute(mixed, [0, 2, 1, 3]), [batch, seq, c.heads * c.v]));
  return ctx.mark(`${id}.out`, linear(ctx, `${name}.o_proj`, merged, c.hidden));
}
export function DeepseekV3MLP(ctx: Ctx, name: string, id: string, x: Tensor, inner: number, c: KimiConfig): Tensor {
  const gate = ctx.mark(`${id}.gate`, linear(ctx, `${name}.gate_proj`, x, inner));
  const up = ctx.mark(`${id}.up`, linear(ctx, `${name}.up_proj`, x, inner));
  const act = ctx.mark(`${id}.silu`, silu(gate));
  const product = ctx.mark(`${id}.mul`, mul(act, up));
  return ctx.mark(`${id}.out`, linear(ctx, `${name}.down_proj`, product, c.hidden));
}
/** Single group (checkpoint n_group=topk_group=1), so group filtering is identity.
 * Correction bias changes ranking; output weights always use original sigmoid. */
export function routeScores(logits: readonly number[], bias: readonly number[], k: number, scaling: number) {
  if (logits.length !== bias.length || k < 1 || k > logits.length) throw new RangeError("invalid router shapes/top-k");
  const scores = logits.map((x) => 1 / (1 + Math.exp(-x)));
  const chosen = scores
    .map((_, i) => i)
    .sort((a, b) => scores[b] + bias[b] - (scores[a] + bias[a]) || a - b)
    .slice(0, k);
  const denominator = chosen.reduce((sum, i) => sum + scores[i], 0) + 1e-20;
  return { chosen, weights: chosen.map((i) => (scores[i] / denominator) * scaling), scores, chosenGroups: [0] };
}
export function MoEGate(ctx: Ctx, name: string, id: string, x: Tensor, c: KimiConfig) {
  const flat = reshape(x, [x.shape[0] * x.shape[1], c.hidden]);
  const logits = ctx.mark(`${id}.logits`, linear(ctx, name, flat, c.experts));
  const scores = ctx.mark(`${id}.scores`, sigmoid(logits));
  const bias = ctx.param(`${name}.e_score_correction_bias`, [c.experts]);
  const choice = ctx.mark(`${id}.choice`, add(scores, bias));
  const indices = ctx.mark(`${id}.indices`, topk(choice, c.topK).indices);
  const gathered = gatherLast(scores, indices);
  let weights: Tensor = { shape: gathered.shape, data: null };
  if (gathered.data !== null) {
    const out = new Float32Array(gathered.data.length);
    for (let row = 0; row < out.length / c.topK; row++) {
      let sum = 1e-20;
      for (let i = 0; i < c.topK; i++) sum += gathered.data[row * c.topK + i];
      for (let i = 0; i < c.topK; i++) out[row * c.topK + i] = (gathered.data[row * c.topK + i] / sum) * c.scale;
    }
    weights = tensor(gathered.shape, out);
  }
  ctx.mark(`${id}.weights`, weights);
  return { indices, weights };
}
export function DeepseekV3MoE(ctx: Ctx, name: string, id: string, x: Tensor, c: KimiConfig): Tensor {
  const { indices, weights } = MoEGate(ctx, `${name}.gate`, id, x, c);
  const flat = reshape(x, [x.shape[0] * x.shape[1], c.hidden]);
  const sum = ctx.mode === "shape" ? null : new Float32Array(flat.shape[0] * c.hidden);
  // Register every logical expert in both modes; only selected tokens execute it.
  // Parameters remain in upstream order so seeded weights reproduce across routes.
  for (let expert = 0; expert < c.experts; expert++) {
    const rows: number[] = [];
    if (indices.data !== null)
      for (let row = 0; row < flat.shape[0]; row++)
        if (Array.from(indices.data.subarray(row * c.topK, (row + 1) * c.topK)).includes(expert)) rows.push(row);
    const tokens: Tensor =
      ctx.mode === "shape"
        ? { shape: [0, c.hidden], data: null }
        : tensor(
            [rows.length, c.hidden],
            rows.flatMap((row) => Array.from(flat.data!.subarray(row * c.hidden, (row + 1) * c.hidden))),
          );
    const out = DeepseekV3MLP(ctx, `${name}.experts.${expert}`, `${id}.expert${expert}`, tokens, c.expert, c);
    if (sum !== null && out.data !== null)
      rows.forEach((row, j) => {
        const slot = Array.from(indices.data!.subarray(row * c.topK, (row + 1) * c.topK)).indexOf(expert);
        for (let h = 0; h < c.hidden; h++)
          sum[row * c.hidden + h] += out.data![j * c.hidden + h] * weights.data![row * c.topK + slot];
      });
  }
  const routed = ctx.mark(`${id}.routed`, tensor(x.shape, sum));
  const shared = DeepseekV3MLP(ctx, `${name}.shared_experts`, `${id}.shared`, x, c.expert * c.shared, c);
  return ctx.mark(`${id}.out`, add(routed, shared));
}
export function DeepseekV3DecoderLayer(ctx: Ctx, index: number, x: Tensor, c: KimiConfig): Tensor {
  const name = `model.layers.${index}`,
    id = `h${index}`;
  const nx = ctx.mark(`${id}.norm1`, norm(ctx, `${name}.input_layernorm`, x, c.eps));
  const after = ctx.mark(`${id}.add1`, add(x, DeepseekV3Attention(ctx, `${name}.self_attn`, `${id}.attn`, nx, c)));
  const nf = ctx.mark(`${id}.norm2`, norm(ctx, `${name}.post_attention_layernorm`, after, c.eps));
  const ff =
    index < c.firstDense
      ? DeepseekV3MLP(ctx, `${name}.mlp`, `${id}.mlp`, nf, c.dense, c)
      : DeepseekV3MoE(ctx, `${name}.mlp`, `${id}.mlp`, nf, c);
  return ctx.mark(`${id}.out`, add(after, ff));
}
export function forward(ctx: Ctx, c: KimiConfig, ids: readonly (readonly number[])[]): Tensor {
  let x = ctx.mark("embed", embedding(ctx.param("model.embed_tokens.weight", [c.vocab, c.hidden]), ids));
  for (let i = 0; i < c.layers; i++) x = DeepseekV3DecoderLayer(ctx, i, x, c);
  x = ctx.mark("final_norm", norm(ctx, "model.norm", x, c.eps));
  return ctx.mark("logits", linear(ctx, "lm_head", x, c.vocab));
}
