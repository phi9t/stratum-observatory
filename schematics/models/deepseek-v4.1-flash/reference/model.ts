// Distilled from publisher inference/model.py, commit
// 2cba9e42aa026125f3ed06c6d98c1db82f7ca027: Transformer, Block,
// Attention, Compressor, Indexer, Gate, Expert, MoE and Engram.
// Scope: text-only prefill to full logits. Engram receives hash IDs at the
// tokenizer boundary. Value mode is floating arithmetic, NOT FP8/FP4 parity;
// caches, image encoder, draft heads, sampling and distributed collectives omitted.
import * as ref from "../../../tnsr/tensor";
import * as f from "../facts";
import { make, map, rope, route, sinkhorn, candidateMask } from "./ops";
export type Config = {
  d: number;
  layers: number;
  vocab: number;
  h: number;
  hd: number;
  rd: number;
  qr: number;
  groups: number;
  or: number;
  inter: number;
  experts: number;
  k: number;
  scale: number;
  window: number;
  ratios: readonly number[];
  kvSources: readonly number[];
  indexSources: readonly number[];
  ih: number;
  id: number;
  ik: number;
  candidateSource: number;
  candidateK: number;
  candidateBlock: number;
  hc: number;
  iterations: number;
  eps: number;
  hcEps: number;
  limit: number;
  engramLayers: readonly number[];
  engramRows: readonly number[];
  hashColumns: number;
  hashDim: number;
  base: number;
  compressedBase: number;
  original: number;
  factor: number;
  fast: number;
  slow: number;
};
export const REAL: Config = {
  d: f.hidden.value,
  layers: f.layers.value,
  vocab: f.vocab.value,
  h: f.heads.value,
  hd: f.headDim.value,
  rd: f.ropeDim.value,
  qr: f.qRank.value,
  groups: f.groups.value,
  or: f.oRank.value,
  inter: f.inter.value,
  experts: f.experts.value,
  k: f.topK.value,
  scale: f.routeScale.value,
  window: f.window.value,
  ratios: f.ratios.value,
  kvSources: f.kvSources.value,
  indexSources: f.indexSources.value,
  ih: f.indexHeads.value,
  id: f.indexDim.value,
  ik: f.indexTopK.value,
  candidateSource: f.candidateSource.value,
  candidateK: f.candidateTopK.value,
  candidateBlock: f.candidateBlock.value,
  hc: f.copies.value,
  iterations: f.iterations.value,
  eps: f.eps.value,
  hcEps: f.hcEps.value,
  limit: f.limit.value,
  engramLayers: f.engramLayers.value,
  engramRows: f.engramRows.value,
  hashColumns: f.hashColumns.value,
  hashDim: f.hashDim.value,
  base: f.ropeBase.value,
  compressedBase: f.compressedBase.value,
  original: f.original.value,
  factor: f.factor.value,
  fast: f.fast.value,
  slow: f.slow.value,
};
// Keep all layer ownership boundaries, residual copies and both compression ratios.
export const TOY: Config = {
  ...REAL,
  ...f.toySpec.value,
};
export const TOY_IDS = [[0, 1, 2, 3, 1, 2, 4, 5]];
type State = {
  kv: ref.Tensor | null;
  key: ref.Tensor | null;
  selected: number[][][] | null;
  candidates: boolean[][][] | null;
};
const linear = (ctx: ref.Ctx, name: string, x: ref.Tensor, output: number) =>
  ref.affine(x, ctx.param(name + ".weight", [output, x.shape.at(-1)!]), null, "out_in");
const norm = (ctx: ref.Ctx, name: string, x: ref.Tensor, eps: number) =>
  ref.rmsNorm(x, ctx.param(name + ".weight", [x.shape.at(-1)!]), eps);
const data = (x: ref.Tensor) => x.data!;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function Gate(ctx: ref.Ctx, c: Config, name: string, x: ref.Tensor) {
  const logits = linear(ctx, name, x, c.experts);
  const bias = ctx.param(name + ".bias", [c.experts]);
  ctx.param(name + ".bias_vl", [c.experts]); // present in the VL checkpoint, text takes bias
  const scores = ctx.mark(
    name + ".scores",
    map(logits, (x) => Math.sqrt(Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))))),
  );
  const rows = x.shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const routing =
    logits.data === null
      ? []
      : Array.from({ length: rows }, (_, r) =>
          route(
            Array.from(data(logits).slice(r * c.experts, (r + 1) * c.experts)),
            Array.from(data(bias)),
            c.k,
            c.scale,
          ),
        );
  ctx.mark(
    name + ".indices",
    make([...x.shape.slice(0, -1), c.k], [x], () => routing.flatMap((r) => r.chosen)),
  );
  ctx.mark(
    name + ".weights",
    make([...x.shape.slice(0, -1), c.k], [x], () => routing.flatMap((r) => r.weights)),
  );
  return { scores, routing };
}
export function Expert(ctx: ref.Ctx, c: Config, name: string, x: ref.Tensor, weights?: readonly number[]) {
  const gate = map(linear(ctx, name + ".w1", x, c.inter), (v) => Math.min(c.limit, v));
  const up = map(linear(ctx, name + ".w3", x, c.inter), (v) => Math.max(-c.limit, Math.min(c.limit, v)));
  let a = ctx.mark(name + ".product", ref.mul(ref.silu(gate), up));
  if (weights) a = make(a.shape, [a], () => Array.from(data(a), (v, i) => v * weights[Math.floor(i / c.inter)]));
  return linear(ctx, name + ".w2", a, c.d);
}
export function MoE(ctx: ref.Ctx, c: Config, name: string, x: ref.Tensor) {
  const { routing } = Gate(ctx, c, name + ".gate", x);
  const rows = x.shape.slice(0, -1).reduce((a, b) => a * b, 1);
  let y = ref.tensor(x.shape, x.data === null ? null : new Float32Array(rows * c.d));
  // Request every stored expert even if not active: shape count is stored, not active count.
  for (let e = 0; e < c.experts; e++) {
    if (ctx.mode === "shape") {
      Expert(ctx, c, name + ".experts." + e, x);
      continue;
    }
    const selected = Array.from({ length: rows }, (_, r) => r).filter((r) => routing[r].chosen.includes(e));
    const tx = ref.tensor(
      [selected.length, c.d],
      selected.flatMap((r) => Array.from(data(x).slice(r * c.d, (r + 1) * c.d))),
    );
    const out = Expert(
      ctx,
      c,
      name + ".experts." + e,
      tx,
      selected.map((r) => routing[r].weights[routing[r].chosen.indexOf(e)]),
    );
    selected.forEach((r, j) => {
      for (let i = 0; i < c.d; i++) data(y)[r * c.d + i] += data(out)[j * c.d + i];
    });
  }
  y = ref.add(y, Expert(ctx, c, name + ".shared_experts", x));
  return ctx.mark(name + ".output", y);
}
export function Compressor(ctx: ref.Ctx, c: Config, name: string, x: ref.Tensor, ratio: number) {
  const [b, s] = x.shape;
  const count = Math.floor(s / ratio);
  const kv = linear(ctx, name + ".wkv", x, c.hd);
  let latent: ref.Tensor;
  if (ratio === 1) latent = kv;
  else {
    const scores = linear(ctx, name + ".wgate", x, c.hd);
    latent = make([b, count, c.hd], [kv, scores], () => {
      const out = new Float32Array(b * count * c.hd);
      for (let batch = 0; batch < b; batch++)
        for (let t = 0; t < count; t++)
          for (let d = 0; d < c.hd; d++) {
            let max = -Infinity;
            for (let r = 0; r < ratio; r++) max = Math.max(max, data(scores)[(batch * s + t * ratio + r) * c.hd + d]);
            let sum = 0,
              v = 0;
            for (let r = 0; r < ratio; r++) {
              const j = (batch * s + t * ratio + r) * c.hd + d;
              const w = Math.exp(data(scores)[j] - max);
              sum += w;
              v += w * data(kv)[j];
            }
            out[(batch * count + t) * c.hd + d] = v / sum;
          }
      return out;
    });
  }
  return ctx.mark(name + ".latent", norm(ctx, name + ".norm", latent, c.eps));
}
function rotate(x: ref.Tensor, c: Config, ratio: number, positionRatio = 1, inverse = false) {
  return rope(
    x,
    c.rd,
    ratio ? c.compressedBase : c.base,
    positionRatio,
    ratio ? c.original : 0,
    c.factor,
    c.fast,
    c.slow,
    inverse,
  );
}
export function Indexer(
  ctx: ref.Ctx,
  c: Config,
  name: string,
  x: ref.Tensor,
  qr: ref.Tensor,
  latent: ref.Tensor | null,
  i: number,
  state: State,
) {
  const [b, s] = x.shape,
    ratio = c.ratios[i];
  if (c.kvSources.includes(i)) {
    if (latent === null) throw Error("KV source has no compressor latent");
    state.key = rotate(norm(ctx, name + ".k_norm", linear(ctx, name + ".wk", latent, c.id), c.eps), c, ratio, ratio);
  }
  const q = rotate(ref.reshape(linear(ctx, name + ".wq_b", qr, c.ih * c.id), [b, s, c.ih, c.id]), c, ratio);
  const weights = ref.scale(linear(ctx, name + ".weights_proj", x, c.ih), (c.id * c.ih) ** -0.5);
  if (state.key === null) throw Error("Index source read before key owner");
  const key = state.key,
    n = key.shape[1];
  const scores = ctx.mark(
    name + ".scores",
    make([b, s, n], [q, key, weights], () => {
      const a = new Float32Array(b * s * n);
      for (let batch = 0; batch < b; batch++)
        for (let t = 0; t < s; t++)
          for (let k = 0; k < n; k++) {
            let sum = 0;
            for (let h = 0; h < c.ih; h++) {
              let dot = 0;
              for (let d = 0; d < c.id; d++)
                dot += data(q)[((batch * s + t) * c.ih + h) * c.id + d] * data(key)[(batch * n + k) * c.id + d];
              sum += Math.max(dot, 0) * data(weights)[(batch * s + t) * c.ih + h];
            }
            a[(batch * s + t) * n + k] = k < Math.floor((t + 1) / ratio) ? sum : -Infinity;
          }
      return a;
    }),
  );
  if (scores.data === null) {
    state.selected = null;
    return;
  }
  const rows = Array.from({ length: b }, (_, batch) =>
    Array.from({ length: s }, (_, t) => Array.from(data(scores).slice((batch * s + t) * n, (batch * s + t + 1) * n))),
  );
  if (i === c.candidateSource)
    state.candidates = rows.map((batch) =>
      batch.map((r, t) => candidateMask(r, Math.floor((t + 1) / ratio), c.candidateK, c.candidateBlock)),
    );
  state.selected = rows.map((batch, bi) =>
    batch.map((r, t) =>
      r
        .map((score, k) => ({ score, k }))
        .filter(({ score, k }) => score > -Infinity && (i <= c.candidateSource || state.candidates?.[bi][t][k]))
        .sort((a, b) => b.score - a.score || a.k - b.k)
        .slice(0, c.ik)
        .map((x) => x.k)
        .sort((a, b) => a - b),
    ),
  );
}
export function Attention(ctx: ref.Ctx, c: Config, name: string, x: ref.Tensor, i: number, state: State) {
  const [b, s] = x.shape,
    ratio = c.ratios[i];
  const qr = ctx.mark(name + ".qr", norm(ctx, name + ".q_norm", linear(ctx, name + ".wq_a", x, c.qr), c.eps));
  const q = ctx.mark(
    name + ".q",
    rotate(ref.reshape(linear(ctx, name + ".wq_b", qr, c.h * c.hd), [b, s, c.h, c.hd]), c, ratio),
  );
  const window = ctx.mark(
    name + ".window",
    rotate(norm(ctx, name + ".kv_norm", linear(ctx, name + ".wkv", x, c.hd), c.eps), c, ratio),
  );
  const sink = ctx.param(name + ".attn_sink", [c.h]);
  let latent: ref.Tensor | null = null;
  if (c.kvSources.includes(i)) latent = Compressor(ctx, c, name + ".compressor", x, ratio);
  if (c.indexSources.includes(i)) Indexer(ctx, c, name + ".indexer", x, qr, latent, i, state);
  if (latent !== null) state.kv = rotate(latent, c, ratio, ratio);
  const kv = ratio ? state.kv : null;
  const n = kv?.shape[1] ?? 0;
  const probabilities = q.data === null ? null : new Float32Array(b * s * c.h * (s + n + 1));
  const attn = make([b, s, c.h, c.hd], [q, window, sink, ...(kv ? [kv] : [])], () => {
    const out = new Float32Array(b * s * c.h * c.hd);
    for (let batch = 0; batch < b; batch++)
      for (let t = 0; t < s; t++)
        for (let h = 0; h < c.h; h++) {
          const reached: Array<{ values: Float32Array; offset: number; column: number }> = [];
          for (let k = Math.max(0, t - c.window + 1); k <= t; k++)
            reached.push({ values: data(window), offset: (batch * s + k) * c.hd, column: k });
          if (kv)
            for (const k of state.selected?.[batch][t] ?? [])
              reached.push({ values: data(kv), offset: (batch * n + k) * c.hd, column: s + k });
          const scores = reached.map((k) => {
            let dot = 0;
            for (let d = 0; d < c.hd; d++)
              dot += data(q)[((batch * s + t) * c.h + h) * c.hd + d] * k.values[k.offset + d];
            return dot / Math.sqrt(c.hd);
          });
          const max = Math.max(data(sink)[h], ...scores),
            exps = scores.map((v) => Math.exp(v - max)),
            sum = Math.exp(data(sink)[h] - max) + exps.reduce((a, b) => a + b, 0);
          const base = ((batch * s + t) * c.h + h) * (s + n + 1);
          reached.forEach((k, j) => {
            probabilities![base + k.column] = exps[j] / sum;
          });
          probabilities![base + s + n] = Math.exp(data(sink)[h] - max) / sum;
          for (let d = 0; d < c.hd; d++)
            out[((batch * s + t) * c.h + h) * c.hd + d] = reached.reduce(
              (v, k, j) => v + (exps[j] / sum) * k.values[k.offset + d],
              0,
            );
        }
    return out;
  });
  ctx.mark(name + ".probabilities", ref.tensor([b, s, c.h, s + n + 1], probabilities));
  const o = ctx.mark(name + ".inverse", rotate(attn, c, ratio, 1, true));
  const groupWidth = (c.h * c.hd) / c.groups;
  const w = ctx.param(name + ".wo_a.weight", [c.groups * c.or, groupWidth]);
  const projected = ctx.mark(
    name + ".grouped",
    make([b, s, c.groups * c.or], [o, w], () => {
      const out = new Float32Array(b * s * c.groups * c.or);
      for (let t = 0; t < b * s; t++)
        for (let g = 0; g < c.groups; g++)
          for (let r = 0; r < c.or; r++) {
            let v = 0;
            for (let d = 0; d < groupWidth; d++)
              v += data(o)[t * c.h * c.hd + g * groupWidth + d] * data(w)[(g * c.or + r) * groupWidth + d];
            out[t * c.groups * c.or + g * c.or + r] = v;
          }
      return out;
    }),
  );
  return ctx.mark(name + ".output", linear(ctx, name + ".wo_b", projected, c.d));
}
export function hcMixes(ctx: ref.Ctx, c: Config, name: string, x: ref.Tensor, which: "attn" | "ffn") {
  const [b, s] = x.shape,
    m = (2 + c.hc) * c.hc;
  const flat = ref.reshape(x, [b, s, c.hc * c.d]);
  const projection = ref.affine(flat, ctx.param(name + ".hc_" + which + "_fn", [m, c.hc * c.d]), null, "out_in");
  const scales = ctx.param(name + ".hc_" + which + "_scale", [3]),
    base = ctx.param(name + ".hc_" + which + "_base", [m]);
  const pre = make([b, s, c.hc], [x, projection, scales, base], () => new Float32Array(b * s * c.hc));
  const post = make([b, s, c.hc], [x, projection, scales, base], () => new Float32Array(b * s * c.hc));
  const comb = make([b, s, c.hc, c.hc], [x, projection, scales, base], () => new Float32Array(b * s * c.hc * c.hc));
  if (x.data !== null)
    for (let t = 0; t < b * s; t++) {
      let square = 0;
      for (let j = 0; j < c.hc * c.d; j++) square += data(x)[t * c.hc * c.d + j] ** 2;
      const inv = 1 / Math.sqrt(square / (c.hc * c.d) + c.eps);
      for (let j = 0; j < c.hc; j++) {
        data(pre)[t * c.hc + j] =
          sigmoid(data(projection)[t * m + j] * inv * data(scales)[0] + data(base)[j]) + c.hcEps;
        data(post)[t * c.hc + j] =
          2 * sigmoid(data(projection)[t * m + c.hc + j] * inv * data(scales)[1] + data(base)[c.hc + j]);
      }
      const logits = Array.from(
        { length: c.hc * c.hc },
        (_, j) => data(projection)[t * m + 2 * c.hc + j] * inv * data(scales)[2] + data(base)[2 * c.hc + j],
      );
      data(comb).set(sinkhorn(logits, c.hc, c.iterations, c.hcEps), t * c.hc * c.hc);
    }
  ctx.mark(name + "." + which + ".comb", comb);
  return { pre, post, comb };
}
export function hcPre(c: Config, x: ref.Tensor, pre: ref.Tensor) {
  const [b, s] = x.shape;
  return make([b, s, c.d], [x, pre], () => {
    const out = new Float32Array(b * s * c.d);
    for (let t = 0; t < b * s; t++)
      for (let d = 0; d < c.d; d++)
        for (let j = 0; j < c.hc; j++) out[t * c.d + d] += data(x)[(t * c.hc + j) * c.d + d] * data(pre)[t * c.hc + j];
    return out;
  });
}
export function hcPost(c: Config, y: ref.Tensor, residual: ref.Tensor, post: ref.Tensor, comb: ref.Tensor) {
  const [b, s] = residual.shape;
  return make(residual.shape, [y, residual, post, comb], () => {
    const out = new Float32Array(b * s * c.hc * c.d);
    for (let t = 0; t < b * s; t++)
      for (let target = 0; target < c.hc; target++)
        for (let d = 0; d < c.d; d++) {
          let v = data(post)[t * c.hc + target] * data(y)[t * c.d + d];
          for (let source = 0; source < c.hc; source++)
            v += data(comb)[(t * c.hc + source) * c.hc + target] * data(residual)[(t * c.hc + source) * c.d + d];
          out[(t * c.hc + target) * c.d + d] = v;
        }
    return out;
  });
}
export function Engram(
  ctx: ref.Ctx,
  c: Config,
  name: string,
  x: ref.Tensor,
  hashIds: readonly (readonly number[])[],
  rows: number,
) {
  const [b, s] = x.shape;
  const lookup = ref.embedding(ctx.param(name + ".embed.weight", [rows, c.hashDim]), hashIds);
  const value = ref.reshape(lookup, [b, s, c.hashColumns * c.hashDim]);
  const kv = linear(ctx, name + ".wkv", value, c.d * (c.hc + 1));
  const qw = ctx.param(name + ".q_weight", [c.hc, c.d]),
    kw = ctx.param(name + ".k_weight", [c.hc, c.d]);
  return ctx.mark(
    name + ".output",
    make(x.shape, [x, kv, qw, kw], () => {
      const out = Float32Array.from(data(x));
      for (let t = 0; t < b * s; t++)
        for (let h = 0; h < c.hc; h++) {
          let dot = 0,
            x2 = 0,
            k2 = 0;
          for (let d = 0; d < c.d; d++) {
            const a = data(x)[(t * c.hc + h) * c.d + d],
              k = data(kv)[t * c.d * (c.hc + 1) + h * c.d + d];
            dot += a * k * data(qw)[h * c.d + d] * data(kw)[h * c.d + d];
            x2 += a * a;
            k2 += k * k;
          }
          dot *= 1 / Math.sqrt(x2 / c.d + c.eps) / Math.sqrt(k2 / c.d + c.eps) / Math.sqrt(c.d);
          const gate = sigmoid(Math.sign(dot || 1) * Math.sqrt(Math.max(Math.abs(dot), f.clampEps.value)));
          for (let d = 0; d < c.d; d++)
            out[(t * c.hc + h) * c.d + d] += gate * data(kv)[t * c.d * (c.hc + 1) + c.hc * c.d + d];
        }
      return out;
    }),
  );
}
export function Block(ctx: ref.Ctx, c: Config, i: number, x: ref.Tensor, pre: ref.Tensor, state: State) {
  const name = "layers." + i,
    attn = hcMixes(ctx, c, name, x, "attn");
  const a = norm(ctx, name + ".attn_norm", hcPre(c, x, pre), c.eps);
  let h = hcPost(c, Attention(ctx, c, name + ".attn", a, i, state), x, attn.post, attn.comb);
  const ffn = hcMixes(ctx, c, name, h, "ffn");
  const input = norm(ctx, name + ".ffn_norm", hcPre(c, h, attn.pre), c.eps);
  h = ctx.mark(name + ".output", hcPost(c, MoE(ctx, c, name + ".ffn", input), h, ffn.post, ffn.comb));
  return { h, pre: ffn.pre };
}
export function forward(
  ctx: ref.Ctx,
  c: Config,
  ids: readonly (readonly number[])[],
  hashes?: Readonly<Record<number, readonly (readonly number[])[]>>,
) {
  const [b, s] = [ids.length, ids[0].length];
  const embed = ref.embedding(ctx.param("embed.weight", [c.vocab, c.d]), ids);
  let h = make([b, s, c.hc, c.d], [embed], () =>
    Array.from({ length: b * s * c.hc * c.d }, (_, i) => data(embed)[Math.floor(i / (c.hc * c.d)) * c.d + (i % c.d)]),
  );
  let pre = make([b, s, c.hc], [h], () => Array.from({ length: b * s * c.hc }, (_, i) => (i % c.hc === 0 ? 1 : 0)));
  const state: State = { kv: null, key: null, selected: null, candidates: null };
  for (let i = 0; i < c.layers; i++) {
    if (c.engramLayers.includes(i)) {
      if (ctx.mode === "value" && !hashes?.[i]) throw Error("Value mode requires explicit Engram hash IDs");
      const hashIds = hashes?.[i] ?? Array.from({ length: b }, () => Array(s * c.hashColumns).fill(0));
      h = Engram(ctx, c, "layers." + i + ".engram", h, hashIds, c.engramRows[c.engramLayers.indexOf(i)]);
    }
    ({ h, pre } = Block(ctx, c, i, h, pre, state));
  }
  const output = norm(ctx, "norm", hcPre(c, h, pre), c.eps);
  return ctx.mark("logits", linear(ctx, "head", output, c.vocab));
}
// Explicit synthetic lookup IDs, not an implementation of publisher int64 hashes.
export function toyHashes(c: Config, ids: readonly (readonly number[])[]) {
  return Object.fromEntries(
    c.engramLayers.map((layer, j) => [
      layer,
      ids.map((row) =>
        row.flatMap((id, t) => Array.from({ length: c.hashColumns }, (_, h) => (id + t + h + layer) % c.engramRows[j])),
      ),
    ]),
  );
}
