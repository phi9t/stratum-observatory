import { cfg, code, derive, type Fact } from "../../kit/lib/facts";
export const hidden = cfg("text_config.hidden_size", 5120);
export const layers = cfg("text_config.num_hidden_layers", 40);
export const heads = cfg("text_config.num_attention_heads", 64);
export const headDim = cfg("text_config.head_dim", 512);
export const ropeDim = cfg("text_config.qk_rope_head_dim", 64);
export const qRank = cfg("text_config.q_lora_rank", 1280);
export const oRank = cfg("text_config.o_lora_rank", 1024);
export const groups = cfg("text_config.o_groups", 8);
export const vocab = cfg("text_config.vocab_size", 129280);
export const inter = cfg("text_config.moe_intermediate_size", 2304);
export const experts = cfg("text_config.n_routed_experts", 384);
export const shared = cfg("text_config.n_shared_experts", 1);
export const topK = cfg("text_config.num_experts_per_tok", 6);
export const routeScale = cfg("text_config.routed_scaling_factor", 1.5);
export const window = cfg("text_config.sliding_window", 128);
export const ratios = cfg(
  "text_config.compress_ratios",
  [
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 0,
  ],
);
export const kvSources = cfg("text_config.kv_source_layer_ids", [2, 8, 14, 20]);
export const indexSources = cfg("text_config.index_source_layer_ids", [2, 8, 14, 20, 24, 28, 32, 36]);
export const candidateSource = cfg("text_config.candidate_source_layer_id", 20);
export const candidateTopK = cfg("text_config.candidate_topk_blocks", 2048);
export const candidateBlock = cfg("text_config.candidate_block_size", 8);
export const indexHeads = cfg("text_config.index_n_heads", 32);
export const indexDim = cfg("text_config.index_head_dim", 128);
export const indexTopK = cfg("text_config.index_topk", 512);
export const copies = cfg("text_config.hc_mult", 4);
export const iterations = cfg("text_config.hc_sinkhorn_iters", 20);
export const hcEps = cfg("text_config.hc_eps", 1e-6);
export const eps = cfg("text_config.rms_norm_eps", 1e-20);
export const limit = cfg("text_config.swiglu_limit", 10.0);
export const scoring = cfg("text_config.scoring_func", "sqrtsoftplus");
export const engramLayers = cfg("text_config.engram_layer_ids", [1, 14]);
export const engramRows = cfg("text_config.engram_num_embeddings", [384006168, 384016682]);
export const ngram = cfg("text_config.engram_max_ngram_size", 4);
export const hashHeads = cfg("text_config.engram_n_heads", 8);
export const hashDim = cfg("text_config.engram_head_dim", 256);
export const context = cfg("text_config.max_position_embeddings", 1048576);
export const ropeBase = cfg("text_config.rope_theta", 10000);
export const compressedBase = cfg("text_config.compress_rope_theta", 160000);
export const draftLayers = cfg("text_config.num_nextn_predict_layers", 3);
export const draftExperts = cfg("text_config.dspark_n_routed_experts", 128);
export const draftTopK = cfg("text_config.dspark_num_experts_per_tok", 3);
export const draftSize = cfg("text_config.dspark_block_size", 5);
export const draftTargets = cfg("text_config.dspark_target_layer_ids", [37, 38, 39]);
export const markovRank = cfg("text_config.dspark_markov_rank", 256);
export const visionLayers = cfg("vision_config.num_hidden_layers", 32);
export const visionDim = cfg("vision_config.hidden_size", 1024);
export const visionHeads = cfg("vision_config.num_attention_heads", 16);
export const visionQKV = derive(() => 3 * visionDim.value, [visionDim]);
export const visionInter = cfg("vision_config.intermediate_size", 2816);
export const patch = cfg("vision_config.patch_size", 14);
export const downsample = cfg("vision_config.downsample_ratio", 3);
export const factor = cfg("text_config.rope_scaling.factor", 16);
export const original = cfg("text_config.rope_scaling.original_max_position_embeddings", 65536);
export const fast = cfg("text_config.rope_scaling.beta_fast", 32);
export const slow = cfg("text_config.rope_scaling.beta_slow", 1);

export const delayedPre = code(true, {
  source: "reference",
  symbol: "Block.forward",
  contains: "self.hc_pre(x, pre_mix)",
});
export const gateTemperature = code(1, {
  source: "reference",
  symbol: "ModelArgs",
  contains: "gate_temp: float = 1.0",
});
export const quantBlock = code(32, {
  source: "reference",
  symbol: "Linear.__init__",
  contains: "in_features // fp4_block_size",
});
export const compressedQuantBlock = code(16, {
  source: "reference",
  symbol: "Attention._compress_kv",
  contains: "fp4_act_quant(latent, 16, True",
});
export const clampEps = code(1e-6, {
  source: "reference",
  symbol: "Engram.__init__",
  contains: "self.clamp_value = 1e-6",
});
export const toyTokens = derive(() => groups.value, [groups]);
export const demoContext = derive(() => window.value * 2, [window]);
export const residualWidth = derive(() => hidden.value * copies.value, [hidden, copies]);
export const qWidth = derive(() => heads.value * headDim.value, [heads, headDim]);
export const groupWidth = derive(() => qWidth.value / groups.value, [qWidth, groups]);
export const outputLatent = derive(() => groups.value * oRank.value, [groups, oRank]);
export const hashColumns = derive(() => (ngram.value - 1) * hashHeads.value, [ngram, hashHeads]);
// Explicit teaching sizes, derived from checkpoint quantities, not checkpoint facts.
export const toySpec = derive(
  () => ({
    d: groups.value,
    vocab: hashColumns.value,
    h: copies.value,
    hd: groups.value,
    rd: copies.value,
    qr: copies.value,
    groups: 2,
    or: copies.value,
    inter: topK.value,
    experts: groups.value,
    k: engramLayers.value.length,
    window: draftLayers.value,
    ih: engramLayers.value.length,
    id: copies.value,
    ik: engramLayers.value.length,
    candidateK: engramLayers.value.length,
    candidateBlock: engramLayers.value.length,
    engramRows: engramLayers.value.map(() => indexHeads.value),
    hashColumns: draftLayers.value,
    hashDim: engramLayers.value.length,
  }),
  [groups, hashColumns, copies, topK, engramLayers, draftLayers, indexHeads],
);
export const hashWidth = derive(() => hashColumns.value * hashDim.value, [hashColumns, hashDim]);
export const mixWidth = derive(() => (2 + copies.value) * copies.value, [copies]);
export const expertParams = derive(() => 3 * hidden.value * inter.value, [hidden, inter]);
export const activeFraction = derive(() => topK.value / experts.value, [topK, experts]);
export function kindOf(i: number) {
  return ratios.value[i] === 0
    ? "window"
    : kvSources.value.includes(i)
      ? "kv"
      : indexSources.value.includes(i)
        ? "index"
        : "reuse";
}
export function owner(i: number, sources: readonly number[]) {
  return sources.filter((x) => x <= i).at(-1);
}
export function layerDescription(i: number) {
  const r = ratios.value[i];
  return r === 0
    ? "Sliding-window only; no compressed state."
    : `Ratio ${r}; compressed KV owner ${owner(i, kvSources.value)}; index owner ${owner(i, indexSources.value)}${engramLayers.value.includes(i) ? "; Engram before attention" : ""}.`;
}
export const ledger = derive(() => {
  const d = hidden.value,
    h = headDim.value,
    q = qRank.value,
    g = groups.value,
    o = oRank.value,
    n = heads.value;
  const attn = d * q + q * n * h + d * h + ((n * h) / g) * (g * o) + g * o * d + q + h + n;
  const hyper = 2 * (mixWidth.value * residualWidth.value + mixWidth.value + 3) + 2 * d;
  const compressor = kvSources.value.reduce((s, i) => s + d * h * (ratios.value[i] > 1 ? 2 : 1) + h, 0);
  const index = indexSources.value.reduce(
    (s, i) =>
      s +
      q * indexHeads.value * indexDim.value +
      d * indexHeads.value +
      (kvSources.value.includes(i) ? h * indexDim.value + indexDim.value : 0),
    0,
  );
  const engram =
    engramRows.value.reduce((s, rows) => s + rows * hashDim.value, 0) +
    engramLayers.value.length * (hashWidth.value * d * (copies.value + 1) + 2 * copies.value * d);
  return [
    {
      label: "Token embedding + head",
      value: 2 * vocab.value * d,
      detail: "Untied vocabulary matrices",
      lane: "merged" as const,
    },
    {
      label: "Attention",
      value: layers.value * attn + compressor + index,
      detail: "Query / shared KV / grouped output / source modules",
      lane: "high" as const,
    },
    {
      label: "Routed experts",
      value: layers.value * experts.value * expertParams.value,
      detail: "All stored experts, not only activated ones",
      lane: "low" as const,
    },
    {
      label: "Shared experts + routers",
      value: layers.value * (shared.value * expertParams.value + experts.value * (d + 2)),
      detail: "Always-on FFN, router and text/image selection biases",
      lane: "low" as const,
    },
    {
      label: "Hyper-connections + norms",
      value: layers.value * hyper + d,
      detail: "Mixing projections, bases, scales and final RMSNorm",
      lane: "skip" as const,
    },
    {
      label: "Engram",
      value: engram,
      detail: "Hash tables, key/value projection and per-copy gains",
      lane: "instance" as const,
    },
  ];
}, [
  hidden,
  headDim,
  qRank,
  groups,
  oRank,
  heads,
  mixWidth,
  residualWidth,
  kvSources,
  ratios,
  indexSources,
  indexHeads,
  indexDim,
  engramRows,
  hashDim,
  engramLayers,
  hashWidth,
  copies,
  vocab,
  layers,
  experts,
  expertParams,
  shared,
]);
export const parameterTotal = derive(() => ledger.value.reduce((s, r) => s + r.value, 0), [ledger]);
export const draftParameters = derive(() => {
  const d = hidden.value,
    h = headDim.value,
    q = qRank.value,
    n = heads.value,
    g = groups.value,
    o = oRank.value;
  const attention = d * q + q * n * h + d * h + n * h * o + g * o * d + q + h + n;
  const hyper = 2 * (mixWidth.value * residualWidth.value + mixWidth.value + 3) + 2 * d;
  const blocks =
    draftLayers.value *
    (attention + hyper + (draftExperts.value + shared.value) * expertParams.value + draftExperts.value * (d + 2));
  const mainProjection = d * d * draftTargets.value.length + d;
  const finalNorm = d;
  const markov = 2 * vocab.value * markovRank.value;
  const confidence = d + markovRank.value;
  return blocks + mainProjection + finalNorm + markov + confidence;
}, [
  hidden,
  headDim,
  qRank,
  heads,
  groups,
  oRank,
  mixWidth,
  residualWidth,
  draftLayers,
  draftExperts,
  shared,
  expertParams,
  draftTargets,
  vocab,
  markovRank,
]);
export const visionParameters = derive(() => {
  const d = visionDim.value,
    i = visionInter.value,
    r = downsample.value;
  const patchProjection = d * 3 * patch.value ** 2 + d;
  const blocks = visionLayers.value * (4 * d * d + 4 * d + 3 * d * i + 2 * d);
  const finalNorm = d;
  const aligner = d * r * r * hidden.value + hidden.value + hidden.value ** 2 + hidden.value;
  const delimiters = 3 * hidden.value;
  return patchProjection + blocks + finalNorm + aligner + delimiters;
}, [visionDim, visionInter, downsample, patch, visionLayers, hidden]);
export const fullLogicalParameters = derive(
  () => parameterTotal.value + draftParameters.value + visionParameters.value,
  [parameterTotal, draftParameters, visionParameters],
);
export const parameterNote = `The logical text backbone excludes ${draftParameters.value} dSpark parameters and ${visionParameters.value} vision/aligner/delimiter parameters, totaling ${draftParameters.value + visionParameters.value}; adding these gives ${fullLogicalParameters.value}, exactly the hub total. Main embedding/head shared by draft stages are counted once. This reconciles logical weights, not packed storage bytes or scale metadata.`;
export function cacheScalars(batch: number, context: number) {
  const windowKV = layers.value * Math.min(context, window.value) * headDim.value;
  const compressed = kvSources.value.reduce((s, i) => s + Math.floor(context / ratios.value[i]) * headDim.value, 0);
  const index = kvSources.value.reduce((s, i) => s + Math.floor(context / ratios.value[i]) * indexDim.value, 0);
  const partial = kvSources.value.reduce(
    (s, i) => s + (ratios.value[i] > 1 ? 2 * ratios.value[i] * headDim.value : 0),
    0,
  );
  return {
    windowKV: batch * windowKV,
    compressed: batch * compressed,
    index: batch * index,
    partial: batch * partial,
    total: batch * (windowKV + compressed + index + partial),
  };
}
export const facts: Record<string, Fact> = {
  hidden,
  layers,
  heads,
  headDim,
  ropeDim,
  qRank,
  oRank,
  groups,
  vocab,
  inter,
  experts,
  shared,
  topK,
  routeScale,
  window,
  ratios,
  kvSources,
  indexSources,
  candidateSource,
  candidateTopK,
  candidateBlock,
  indexHeads,
  indexDim,
  indexTopK,
  copies,
  iterations,
  hcEps,
  eps,
  limit,
  scoring,
  engramLayers,
  engramRows,
  ngram,
  hashHeads,
  hashDim,
  context,
  ropeBase,
  compressedBase,
  draftLayers,
  draftExperts,
  draftTopK,
  draftSize,
  draftTargets,
  markovRank,
  visionLayers,
  visionDim,
  visionHeads,
  visionInter,
  visionQKV,
  patch,
  downsample,
  factor,
  original,
  fast,
  slow,
  delayedPre,
  gateTemperature,
  quantBlock,
  compressedQuantBlock,
  clampEps,
  toySpec,
  toyTokens,
  demoContext,
  residualWidth,
  qWidth,
  groupWidth,
  outputLatent,
  hashColumns,
  hashWidth,
  mixWidth,
  expertParams,
  activeFraction,
  ledger,
  parameterTotal,
  draftParameters,
  visionParameters,
  fullLogicalParameters,
};
