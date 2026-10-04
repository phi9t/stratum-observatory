// Distilled from publisher KimiLinear classes at f831ab66814297da540d832a5235f8e904f29d06.
// Text decoder prefill to logits. No padding, decode plumbing, vision, loss or packed-weight execution.
// FLA naive recurrence and lower-bound gate at pinned 3e52d5abb6ad underpin scalar KDA.
// Whole-model parity with publisher GPU kernels is UNVERIFIED; see REPORT.md.
import * as r from "../../../tnsr/tensor";
import { attnRes, route, mapped, convolution, l2norm, decayGate, kdaRecurrence, gatedNorm } from "./ops";
import * as f from "../facts";
export type Config = {
  hidden: number;
  layers: number;
  vocab: number;
  heads: number;
  dim: number;
  qRank: number;
  kvRank: number;
  nopeDim: number;
  sharedKeyDim: number;
  vDim: number;
  experts: number;
  topK: number;
  latent: number;
  intermediate: number;
  denseIntermediate: number;
  shared: number;
  blockSize: number;
  convKernel: number;
  lowerBound: number;
  eps: number;
  normLatentEps: number;
  beta: number;
  linearBeta: number;
  kda: readonly number[];
  denseLayers: number;
};
export const realConfig: Config = {
  hidden: f.hidden.value,
  layers: f.layers.value,
  vocab: f.vocab.value,
  heads: f.heads.value,
  dim: f.dim.value,
  qRank: f.qRank.value,
  kvRank: f.kvRank.value,
  nopeDim: f.nopeDim.value,
  sharedKeyDim: f.sharedKeyDim.value,
  vDim: f.vDim.value,
  experts: f.experts.value,
  topK: f.topK.value,
  latent: f.latent.value,
  intermediate: f.intermediate.value,
  denseIntermediate: f.denseIntermediate.value,
  shared: f.shared.value,
  blockSize: f.blockSize.value,
  convKernel: f.convKernel.value,
  lowerBound: f.lowerBound.value,
  eps: f.eps.value,
  normLatentEps: f.normLatentEps.value,
  beta: f.beta.value,
  linearBeta: f.linearBeta.value,
  kda: f.kdaLayers.value,
  denseLayers: f.denseLayers.value,
};
function linear(ctx: r.Ctx, x: r.Tensor, name: string, out: number) {
  return r.affine(x, ctx.param(`${name}.weight`, [out, x.shape.at(-1)!]), null, "out_in");
}
function norm(ctx: r.Ctx, x: r.Tensor, name: string, eps: number) {
  return r.rmsNorm(x, ctx.param(`${name}.weight`, [x.shape.at(-1)!]), eps);
}
export function SituAndMul(gate: r.Tensor, up: r.Tensor, beta: number, linearBeta: number) {
  return mapped(
    gate,
    (v, i) => ((beta * Math.tanh(v / beta)) / (1 + Math.exp(-v))) * linearBeta * Math.tanh(up.data![i] / linearBeta),
  );
}
export function KimiMLP(ctx: r.Ctx, c: Config, x: r.Tensor, p: string, I: number, sparse = false) {
  const gate = ctx.mark(`${p}.gate`, linear(ctx, x, `${p}.${sparse ? "w1" : "gate_proj"}`, I));
  const up = ctx.mark(`${p}.up`, linear(ctx, x, `${p}.${sparse ? "w3" : "up_proj"}`, I));
  const act = ctx.mark(`${p}.situ`, SituAndMul(gate, up, c.beta, c.linearBeta));
  return ctx.mark(`${p}.down`, linear(ctx, act, `${p}.${sparse ? "w2" : "down_proj"}`, x.shape.at(-1)!));
}
export function KimiMoEGate(ctx: r.Ctx, c: Config, x: r.Tensor, p: string) {
  const logits = r.affine(x, ctx.param(`${p}.weight`, [c.experts, c.hidden]), null, "out_in");
  const routing = route(logits, ctx.param(`${p}.e_score_correction_bias`, [c.experts]), c.topK);
  ctx.mark(`${p}.scores`, routing.scores);
  ctx.mark(`${p}.ids`, routing.ids);
  ctx.mark(`${p}.weights`, routing.weights);
  return routing;
}
export function KimiSparseMoeBlock(ctx: r.Ctx, c: Config, x: r.Tensor, p: string) {
  const routing = KimiMoEGate(ctx, c, x, `${p}.gate`);
  const z = ctx.mark(`${p}.latent`, linear(ctx, x, `${p}.routed_expert_down_proj`, c.latent));
  // Execute every expert in toy mode, but aggregate only selected outputs. Shape mode enumerates all weights.
  const ys = Array.from({ length: c.experts }, (_, e) => KimiMLP(ctx, c, z, `${p}.experts.${e}`, c.intermediate, true));
  let aggregate: r.Tensor = { shape: z.shape, data: null };
  if (z.data) {
    const out = new Float32Array(z.data.length);
    for (let token = 0; token < out.length / c.latent; token++)
      for (let j = 0; j < c.topK; j++) {
        const e = routing.ids.data![token * c.topK + j],
          w = routing.weights.data![token * c.topK + j];
        for (let d = 0; d < c.latent; d++) out[token * c.latent + d] += w * ys[e].data![token * c.latent + d];
      }
    aggregate = r.tensor(z.shape, out);
  }
  const y = ctx.mark(`${p}.sum`, aggregate);
  const normalized = ctx.mark(`${p}.norm`, norm(ctx, y, `${p}.routed_expert_norm`, c.eps));
  const expanded = linear(ctx, normalized, `${p}.routed_expert_up_proj`, c.hidden);
  const shared = KimiMLP(ctx, c, x, `${p}.shared_experts`, c.intermediate * c.shared);
  return ctx.mark(`${p}.out`, r.add(expanded, shared));
}
export function KimiMLAAttention(ctx: r.Ctx, c: Config, x: r.Tensor, p: string, mark = p) {
  const [B, S] = x.shape,
    H = c.heads,
    Q = c.nopeDim + c.sharedKeyDim;
  const qa = ctx.mark(`${mark}.qa`, linear(ctx, x, `${p}.q_a_proj`, c.qRank));
  const qn = norm(ctx, qa, `${p}.q_a_layernorm`, c.normLatentEps);
  const q = ctx.mark(
    `${mark}.q`,
    r.permute(r.reshape(linear(ctx, qn, `${p}.q_b_proj`, H * Q), [B, S, H, Q]), [0, 2, 1, 3]),
  );
  const compressed = linear(ctx, x, `${p}.kv_a_proj_with_mqa`, c.kvRank + c.sharedKeyDim);
  const [latent, sharedKey] = r.splitLast(compressed, [c.kvRank, c.sharedKeyDim]);
  ctx.mark(`${mark}.latent`, latent);
  const kv = r.reshape(
    linear(ctx, norm(ctx, latent, `${p}.kv_a_layernorm`, c.normLatentEps), `${p}.kv_b_proj`, H * (c.nopeDim + c.vDim)),
    [B, S, H, c.nopeDim + c.vDim],
  );
  const [keyPart, valuePart] = r.splitLast(kv, [c.nopeDim, c.vDim]);
  let broadcast: r.Tensor = { shape: [B, S, H, c.sharedKeyDim], data: null };
  if (sharedKey.data) {
    const a = new Float32Array(B * S * H * c.sharedKeyDim);
    for (let row = 0; row < B * S; row++)
      for (let h = 0; h < H; h++)
        a.set(
          sharedKey.data.subarray(row * c.sharedKeyDim, (row + 1) * c.sharedKeyDim),
          (row * H + h) * c.sharedKeyDim,
        );
    broadcast = r.tensor(broadcast.shape, a);
  }
  const key = r.permute(r.concatLast([keyPart, broadcast]), [0, 2, 1, 3]),
    value = r.permute(valuePart, [0, 2, 1, 3]);
  ctx.mark(`${mark}.key`, key);
  ctx.mark(`${mark}.value`, value);
  const scores = r.scale(r.matmul(q, r.permute(key, [0, 1, 3, 2])), Q ** -0.5);
  const weights = ctx.mark(`${mark}.weights`, r.softmax(r.causalMask(scores)));
  const attended = r.reshape(r.permute(r.matmul(weights, value), [0, 2, 1, 3]), [B, S, H * c.vDim]);
  const gated = r.mul(attended, r.sigmoid(linear(ctx, x, `${p}.g_proj`, H * c.vDim)));
  return ctx.mark(`${mark}.out`, linear(ctx, gated, `${p}.o_proj`, c.hidden));
}
export function KimiDeltaAttention(ctx: r.Ctx, c: Config, x: r.Tensor, p: string, mark = p) {
  const [B, S] = x.shape,
    H = c.heads,
    D = c.dim,
    W = H * D;
  const projected = (letter: string) => {
    const z = linear(ctx, x, `${p}.${letter}_proj`, W);
    const w = ctx.param(`${p}.${letter}_conv1d.weight`, [W, 1, c.convKernel]);
    return r.reshape(convolution(z, w), [B, S, H, D]);
  };
  const q = ctx.mark(`${mark}.q`, l2norm(projected("q"))),
    k = l2norm(projected("k")),
    v = projected("v");
  const g = r.reshape(linear(ctx, linear(ctx, x, `${p}.f_a_proj`, D), `${p}.f_b_proj`, W), [B, S, H, D]);
  const a = ctx.param(`${p}.A_log`, [H]),
    dt = ctx.param(`${p}.dt_bias`, [W]);
  const decay = ctx.mark(`${mark}.decay`, decayGate(g, a, dt, c.lowerBound));
  const beta = ctx.mark(`${mark}.beta`, r.sigmoid(linear(ctx, x, `${p}.b_proj`, H)));
  const recurrence = kdaRecurrence(q, k, v, decay, beta);
  ctx.mark(`${mark}.state`, recurrence.state);
  ctx.mark(`${mark}.output`, recurrence.output);
  const gate = r.reshape(linear(ctx, x, `${p}.g_proj`, W), [B, S, H, D]);
  const out = gatedNorm(recurrence.output, gate, ctx.param(`${p}.o_norm.weight`, [D]), c.eps);
  return linear(ctx, r.reshape(out, [B, S, W]), `${p}.o_proj`, c.hidden);
}
function mixture(ctx: r.Ctx, c: Config, prefix: r.Tensor, bank: readonly r.Tensor[], p: string, mark: string) {
  const gain = ctx.param(`${p}_norm.weight`, [c.hidden]),
    proj = ctx.param(`${p}_proj.weight`, [1, c.hidden]);
  const mix = attnRes(prefix, bank, gain, proj, c.eps);
  ctx.mark(`${mark}.depth-weights`, mix.weights);
  return ctx.mark(`${mark}.mix`, mix.output);
}
export function forward(ctx: r.Ctx, c: Config, ids: readonly (readonly number[])[]) {
  let x = r.embedding(ctx.param("language_model.model.embed_tokens.weight", [c.vocab, c.hidden]), ids);
  let bank: r.Tensor[] = [];
  for (let i = 0; i < c.layers; i++) {
    const p = `language_model.model.layers.${i}`,
      m = `h${i}`;
    // Parameters exist even when the empty initial bank bypasses the attention-side mixture.
    const mixed = mixture(ctx, c, x, bank, `${p}.self_attention_res`, `${m}.attn-res`);
    const attentionInput = bank.length ? mixed : x;
    let prefix: r.Tensor | null = x;
    if (i % c.blockSize === 0) {
      bank = [...bank, x];
      prefix = null;
    }
    const normalized = norm(ctx, attentionInput, `${p}.input_layernorm`, c.eps);
    const attention = c.kda.includes(i + 1)
      ? KimiDeltaAttention(ctx, c, normalized, `${p}.self_attn`, `${m}.kda`)
      : KimiMLAAttention(ctx, c, normalized, `${p}.self_attn`, `${m}.mla`);
    prefix = prefix ? r.add(prefix, attention) : attention;
    const ffnInput = norm(
      ctx,
      mixture(ctx, c, prefix, bank, `${p}.mlp_res`, `${m}.mlp-res`),
      `${p}.post_attention_layernorm`,
      c.eps,
    );
    const y =
      i >= c.denseLayers
        ? KimiSparseMoeBlock(ctx, c, ffnInput, `${p}.block_sparse_moe`)
        : KimiMLP(ctx, c, ffnInput, `${p}.mlp`, c.denseIntermediate);
    x = ctx.mark(`${m}.prefix`, r.add(prefix, y));
  }
  x = mixture(ctx, c, x, bank, "language_model.model.output_attn_res", "output-res");
  x = norm(ctx, x, "language_model.model.norm", c.eps);
  return ctx.mark("logits", linear(ctx, x, "language_model.lm_head", c.vocab));
}
