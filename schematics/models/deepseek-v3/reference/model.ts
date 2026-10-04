// Distilled from DeepseekV3ForCausalLM/Model/DecoderLayer/Attention/MLP/MoE,
// MoEGate and DeepseekV3YarnRotaryEmbedding in the publisher modeling_deepseek.py
// at e815299b0bcbac849fa540c768ef21845365c9eb (../sources.json reference).
// Main text decoder, eval, unpadded full sequence to logits. No sampling, loss,
// MTP, quantization, distributed execution or persistent decode cache. The marked
// normalized latent + rotated shared key show the compressed serving cache's
// contents, not an eager cache implementation. Eager keys/values are rebuilt here.
// Config selects low-rank Q, YaRN, SiLU, sigmoid/noaux_tc, renorm, untied head,
// ep_size=1, no biases in attention, zero dropout; only layer kind varies.
import * as r from "../../../tnsr/tensor";
import { repeatHeads, rotateYarn } from "./ops";
import * as f from "../facts";
import type { Routing } from "../../../kit";

export type Config = Readonly<{
  vocab: number;
  hidden: number;
  layers: number;
  heads: number;
  denseWidth: number;
  expertWidth: number;
  experts: number;
  shared: number;
  topK: number;
  firstDense: number;
  moeFrequency: number;
  groups: number;
  topGroups: number;
  qRank: number;
  kvRank: number;
  nope: number;
  rotary: number;
  valueDim: number;
  epsilon: number;
  theta: number;
  ropeFactor: number;
  original: number;
  fast: number;
  slow: number;
  mscale: number;
  mscaleAll: number;
  routeScale: number;
}>;
export const TOY: Config = f.TOY;
export const REAL: Config = Object.fromEntries(
  Object.keys(TOY)
    .filter((k) => !["tokens", "gain", "seed", "ids"].includes(k))
    .map((k) => [k, (f as unknown as Record<string, { value: number }>)[k].value]),
) as Config;
export function fromConfig(c: Record<string, unknown>): Config {
  const keys: Record<string, string> = {
    vocab: "vocab_size",
    hidden: "hidden_size",
    layers: "num_hidden_layers",
    heads: "num_attention_heads",
    denseWidth: "intermediate_size",
    expertWidth: "moe_intermediate_size",
    experts: "n_routed_experts",
    shared: "n_shared_experts",
    topK: "num_experts_per_tok",
    firstDense: "first_k_dense_replace",
    moeFrequency: "moe_layer_freq",
    groups: "n_group",
    topGroups: "topk_group",
    qRank: "q_lora_rank",
    kvRank: "kv_lora_rank",
    nope: "qk_nope_head_dim",
    rotary: "qk_rope_head_dim",
    valueDim: "v_head_dim",
    epsilon: "rms_norm_eps",
    theta: "rope_theta",
    routeScale: "routed_scaling_factor",
  };
  const rope = c.rope_scaling as Record<string, number>;
  return {
    ...Object.fromEntries(Object.entries(keys).map(([k, v]) => [k, c[v]])),
    ropeFactor: rope.factor,
    original: rope.original_max_position_embeddings,
    fast: rope.beta_fast,
    slow: rope.beta_slow,
    mscale: rope.mscale,
    mscaleAll: rope.mscale_all_dim,
  } as Config;
}
function linear(ctx: r.Ctx, name: string, x: r.Tensor, out: number): r.Tensor {
  return r.affine(x, ctx.param(`${name}.weight`, [out, x.shape.at(-1)!]), null, "out_in");
}
export function DeepseekV3RMSNorm(ctx: r.Ctx, name: string, x: r.Tensor, eps: number): r.Tensor {
  return r.rmsNorm(x, ctx.param(`${name}.weight`, [x.shape.at(-1)!]), eps);
}
export function DeepseekV3YarnRotaryEmbedding(c: Config) {
  const correction = (turns: number) =>
    (c.rotary * Math.log(c.original / (turns * 2 * Math.PI))) / (2 * Math.log(c.theta));
  const lo = Math.max(0, Math.floor(correction(c.fast))),
    hi = Math.min(c.rotary - 1, Math.ceil(correction(c.slow)));
  const mscale = (amount: number) => (c.ropeFactor <= 1 ? 1 : 1 + 0.1 * amount * Math.log(c.ropeFactor));
  const frequencies = Array.from({ length: c.rotary / 2 }, (_, j) => {
    const ramp = Math.max(0, Math.min(1, (j - lo) / (hi === lo ? 0.001 : hi - lo)));
    return c.theta ** ((-2 * j) / c.rotary) * (1 - ramp + ramp / c.ropeFactor);
  });
  return {
    frequencies,
    amplitude: mscale(c.mscale) / mscale(c.mscaleAll),
    scale: (c.nope + c.rotary) ** -0.5 * mscale(c.mscaleAll) ** 2,
  };
}
export function DeepseekV3Attention(ctx: r.Ctx, name: string, id: string, x: r.Tensor, c: Config): r.Tensor {
  const [b, s] = x.shape,
    qd = c.nope + c.rotary;
  const mark = (op: string, t: r.Tensor) => ctx.mark(`${id}.${op}`, t);
  const heads = (t: r.Tensor, d: number) => r.permute(r.reshape(t, [b, s, c.heads, d]), [0, 2, 1, 3]);
  const qa = mark("qa", linear(ctx, `${name}.q_a_proj`, x, c.qRank));
  // Upstream rank norms default to epsilon=1e-6, independently of block epsilon.
  const qn = mark("qn", DeepseekV3RMSNorm(ctx, `${name}.q_a_layernorm`, qa, 1e-6));
  const qb = mark("qb", linear(ctx, `${name}.q_b_proj`, qn, c.heads * qd));
  const qh = mark("qh", heads(qb, qd));
  const [qNope, qPe] = r.splitLast(qh, [c.nope, c.rotary]);
  mark("qs", qNope);
  const ka = mark("ka", linear(ctx, `${name}.kv_a_proj_with_mqa`, x, c.kvRank + c.rotary));
  const [latent, kPe] = r.splitLast(ka, [c.kvRank, c.rotary]);
  mark("ks", latent);
  const kn = mark("kn", DeepseekV3RMSNorm(ctx, `${name}.kv_a_layernorm`, latent, 1e-6));
  const kb = mark("kb", linear(ctx, `${name}.kv_b_proj`, kn, c.heads * (c.nope + c.valueDim)));
  const kh = mark("kh", heads(kb, c.nope + c.valueDim));
  const [kNope, v] = r.splitLast(kh, [c.nope, c.valueDim]);
  mark("knope", kNope);
  mark("kvsplit", v);
  const yarn = DeepseekV3YarnRotaryEmbedding(c);
  const rq = mark("rotate", rotateYarn(qPe, yarn.frequencies, yarn.amplitude));
  const rk = mark(
    "kpe",
    rotateYarn(r.permute(r.reshape(kPe, [b, s, 1, c.rotary]), [0, 2, 1, 3]), yarn.frequencies, yarn.amplitude),
  );
  const q = mark("joinq", r.concatLast([qNope, rq]));
  const k = mark("joink", r.concatLast([kNope, repeatHeads(rk, c.heads)]));
  const dot = mark("dot", r.matmul(q, r.permute(k, [0, 1, 3, 2])));
  const scaled = mark("scale", r.scale(dot, yarn.scale));
  const mask = mark("mask", r.causalMask(scaled));
  const p = mark("softmax", r.softmax(mask));
  const out = mark("weighted", r.matmul(p, v));
  const merge = mark("merge", r.reshape(r.permute(out, [0, 2, 1, 3]), [b, s, c.heads * c.valueDim]));
  return mark("o", linear(ctx, `${name}.o_proj`, merge, c.hidden));
}
export function DeepseekV3MLP(ctx: r.Ctx, name: string, id: string, x: r.Tensor, width: number, c: Config): r.Tensor {
  const gate = ctx.mark(`${id}.gate`, linear(ctx, `${name}.gate_proj`, x, width));
  const up = ctx.mark(`${id}.up`, linear(ctx, `${name}.up_proj`, x, width));
  const activated = ctx.mark(`${id}.silu`, r.silu(gate));
  const product = ctx.mark(`${id}.mul`, r.mul(activated, up));
  return ctx.mark(`${id}.down`, linear(ctx, `${name}.down_proj`, product, c.hidden));
}
/** Canonical order is increasing id; upstream sorted=False topk order is immaterial
 * to a mixture. Ties use lowest id here; PyTorch's tied ordering is unspecified. */
export function MoEGate(ctx: r.Ctx, name: string, id: string, x: r.Tensor, c: Config): Routing[] {
  const [b, s] = x.shape,
    n = b * s;
  const flat = ctx.mark(`${id}.flatten`, r.reshape(x, [n, c.hidden]));
  const logits = ctx.mark(`${id}.logits`, linear(ctx, name, flat, c.experts));
  const scores = ctx.mark(`${id}.sigmoid`, r.sigmoid(logits));
  const bias = ctx.param(`${name}.e_score_correction_bias`, [c.experts]);
  const choice = ctx.mark(`${id}.bias`, r.add(scores, bias));
  const ordered = (a: readonly number[]) => a.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v || a.i - b.i);
  const routes: Routing[] = [];
  if (scores.data !== null && choice.data !== null)
    for (let token = 0; token < n; token++) {
      const original = Array.from(scores.data.subarray(token * c.experts, (token + 1) * c.experts));
      const corrected = Array.from(choice.data.subarray(token * c.experts, (token + 1) * c.experts));
      const per = c.experts / c.groups;
      const gs = Array.from({ length: c.groups }, (_, g) =>
        ordered(corrected.slice(g * per, (g + 1) * per))
          .slice(0, 2)
          .reduce((a, v) => a + v.v, 0),
      );
      const chosenGroups = ordered(gs)
        .slice(0, c.topGroups)
        .map((v) => v.i)
        .sort((a, b) => a - b);
      const chosen = ordered(corrected.map((v, e) => (chosenGroups.includes(Math.floor(e / per)) ? v : -Infinity)))
        .slice(0, c.topK)
        .map((v) => v.i)
        .sort((a, b) => a - b);
      const total = chosen.reduce((a, e) => a + original[e], 0) + 1e-20;
      routes.push({
        scores: original,
        chosen,
        chosenGroups,
        weights: chosen.map((e) => (original[e] / total) * c.routeScale),
      });
    }
  ctx.mark(
    `${id}.groups`,
    r.tensor([n, c.topGroups], scores.data === null ? null : routes.flatMap((v) => v.chosenGroups)),
  );
  ctx.mark(`${id}.topk`, r.tensor([n, c.topK], scores.data === null ? null : routes.flatMap((v) => v.chosen)));
  ctx.mark(`${id}.routeweights`, r.tensor([n, c.topK], scores.data === null ? null : routes.flatMap((v) => v.weights)));
  return routes;
}
export function DeepseekV3MoE(ctx: r.Ctx, name: string, id: string, x: r.Tensor, c: Config): r.Tensor {
  const routes = MoEGate(ctx, `${name}.gate`, id, x, c);
  const n = x.shape[0] * x.shape[1];
  const flat = r.reshape(x, [n, c.hidden]);
  const sum = ctx.mode === "value" ? new Float32Array(n * c.hidden) : null;
  // Visit every expert even if its token batch is empty: parameters belong to the
  // model independent of routing. Shape mode never constructs a value-sized buffer.
  for (let e = 0; e < c.experts; e++) {
    const tokens =
      ctx.mode === "value" ? routes.map((route, t) => (route.chosen.includes(e) ? t : -1)).filter((t) => t >= 0) : [];
    const data =
      flat.data === null
        ? null
        : tokens.flatMap((t) => Array.from(flat.data!.subarray(t * c.hidden, (t + 1) * c.hidden)));
    const batch = r.tensor([tokens.length, c.hidden], data);
    const out = DeepseekV3MLP(ctx, `${name}.experts.${e}`, `${id}.expert${e}`, batch, c.expertWidth, c);
    if (sum !== null && out.data !== null)
      tokens.forEach((t, i) => {
        const weight = routes[t].weights[routes[t].chosen.indexOf(e)];
        for (let d = 0; d < c.hidden; d++) sum[t * c.hidden + d] += out.data![i * c.hidden + d] * weight;
      });
  }
  const routed = ctx.mark(`${id}.weighted`, r.tensor(x.shape, sum));
  const shared = DeepseekV3MLP(ctx, `${name}.shared_experts`, `${id}.shared`, x, c.shared * c.expertWidth, c);
  return ctx.mark(`${id}.sum`, r.add(routed, shared));
}
export function DeepseekV3DecoderLayer(ctx: r.Ctx, index: number, x: r.Tensor, c: Config): r.Tensor {
  const name = `model.layers.${index}`,
    id = `h${index}`;
  const normalized = DeepseekV3RMSNorm(ctx, `${name}.input_layernorm`, x, c.epsilon);
  const attn = DeepseekV3Attention(ctx, `${name}.self_attn`, id, normalized, c);
  const after = r.add(x, attn);
  const ffnInput = DeepseekV3RMSNorm(ctx, `${name}.post_attention_layernorm`, after, c.epsilon);
  const moe = index >= c.firstDense && index % c.moeFrequency === 0;
  const ffn = moe
    ? DeepseekV3MoE(ctx, `${name}.mlp`, id, ffnInput, c)
    : DeepseekV3MLP(ctx, `${name}.mlp`, id, ffnInput, c.denseWidth, c);
  return ctx.mark(`${id}.out`, r.add(after, ffn));
}
export function DeepseekV3Model(ctx: r.Ctx, c: Config, ids: readonly (readonly number[])[]): r.Tensor {
  let x = ctx.mark("embed", r.embedding(ctx.param("model.embed_tokens.weight", [c.vocab, c.hidden]), ids));
  for (let i = 0; i < c.layers; i++) x = DeepseekV3DecoderLayer(ctx, i, x, c);
  return ctx.mark("final_norm", DeepseekV3RMSNorm(ctx, "model.norm", x, c.epsilon));
}
export function DeepseekV3ForCausalLM(ctx: r.Ctx, c: Config, ids: readonly (readonly number[])[]): r.Tensor {
  return ctx.mark("logits", linear(ctx, "lm_head", DeepseekV3Model(ctx, c, ids), c.vocab));
}

export const forward = DeepseekV3ForCausalLM;
