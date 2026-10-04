import { cfg, code, derive, normals, routeToken, type Fact, type Routing } from "../../kit";

export const hidden = cfg("hidden_size", 7168);
export const vocab = cfg("vocab_size", 129280);
export const layers = cfg("num_hidden_layers", 61);
export const heads = cfg("num_attention_heads", 128);
export const denseWidth = cfg("intermediate_size", 18432);
export const expertWidth = cfg("moe_intermediate_size", 2048);
export const experts = cfg("n_routed_experts", 256);
export const shared = cfg("n_shared_experts", 1);
export const topK = cfg("num_experts_per_tok", 8);
export const firstDense = cfg("first_k_dense_replace", 3);
export const moeFrequency = cfg("moe_layer_freq", 1);
export const groups = cfg("n_group", 8);
export const topGroups = cfg("topk_group", 4);
export const normalize = cfg("norm_topk_prob", true);
export const scoring = cfg("scoring_func", "sigmoid");
export const method = cfg("topk_method", "noaux_tc");
export const routeScale = cfg("routed_scaling_factor", 2.5);
export const qRank = cfg("q_lora_rank", 1536);
export const kvRank = cfg("kv_lora_rank", 512);
export const nope = cfg("qk_nope_head_dim", 128);
export const rotary = cfg("qk_rope_head_dim", 64);
export const valueDim = cfg("v_head_dim", 128);
export const epsilon = cfg("rms_norm_eps", 1e-6);
export const attentionBias = cfg("attention_bias", false);
export const dropout = cfg("attention_dropout", 0);
export const tied = cfg("tie_word_embeddings", false);
export const activation = cfg("hidden_act", "silu");
export const positions = cfg("max_position_embeddings", 163840);
export const theta = cfg("rope_theta", 10000);
export const ropeType = cfg("rope_scaling.type", "yarn");
export const ropeFactor = cfg("rope_scaling.factor", 40);
export const original = cfg("rope_scaling.original_max_position_embeddings", 4096);
export const fast = cfg("rope_scaling.beta_fast", 32);
export const slow = cfg("rope_scaling.beta_slow", 1);
export const mscale = cfg("rope_scaling.mscale", 1);
export const mscaleAll = cfg("rope_scaling.mscale_all_dim", 1);
export const weightBlock = cfg("quantization_config.weight_block_size", [128, 128]);
export const mtpLayers = cfg("num_nextn_predict_layers", 1);
export const bos = cfg("bos_token_id", 0);
export const eos = cfg("eos_token_id", 1);
export const dtype = cfg("torch_dtype", "bfloat16");
export const epSize = cfg("ep_size", 1);

export const matricesPerFfn = code(3, {
  source: "reference",
  symbol: "DeepseekV3MLP",
  contains: "self.down_proj = nn.Linear",
});
export const attentionMatrices = code(5, {
  source: "reference",
  symbol: "DeepseekV3Attention.__init__",
  contains: "self.o_proj = nn.Linear(",
});
export const groupPeakCount = code(2, { source: "reference", symbol: "MoEGate.forward", contains: ".topk(2, dim=-1)" });
export const routeEpsilon = code(1e-20, {
  source: "reference",
  symbol: "MoEGate.forward",
  contains: "keepdim=True) + 1e-20",
});
export const yarnCoefficient = code(0.1, {
  source: "reference",
  symbol: "yarn_get_mscale",
  contains: "0.1 * mscale * math.log(scale)",
});
export const normRule = code("float32 variance, learned gain, no bias", {
  source: "reference",
  symbol: "DeepseekV3RMSNorm.forward",
  contains: "hidden_states.pow(2).mean(-1, keepdim=True)",
});
export const correctedChoice = code("bias changes selection only", {
  source: "reference",
  symbol: "MoEGate.forward",
  contains: "topk_weight = scores.gather(1, topk_idx)",
});
export const layerRule = code("layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0", {
  source: "reference",
  symbol: "DeepseekV3DecoderLayer",
  contains: "layer_idx % config.moe_layer_freq == 0",
});
export const sharedRule = code("unrouted shared output is added", {
  source: "reference",
  symbol: "DeepseekV3MoE.forward",
  contains: "y = y + self.shared_experts(identity)",
});
export const expandedCache = code("expanded keys and values", {
  source: "reference",
  symbol: "DeepseekV3Attention.forward",
  contains: "key_states, value_states = past_key_value.update(",
});
export const paddedCache = code("values padded before cache update", {
  source: "reference",
  symbol: "DeepseekV3FlashAttention2.forward",
  contains: "value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])",
});
export const compressedCache = code("normalized latent and positional key", {
  source: "publisher",
  symbol: "MLA.forward",
  contains: "self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)",
});
export const fusedProjection = code("merged query and KV A projections", {
  source: "serving",
  symbol: "DeepseekV2MLAAttention",
  contains: "[self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim]",
});
export const mtpFusion = code("concat normalized embedding and hidden, then project", {
  source: "mtp",
  symbol: "MTPTransformerBlock.forward",
  contains: "torch.cat([self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1)",
});
export const fp8ScaleRule = code("ceil(out/block) × ceil(in/block)", {
  source: "publisher",
  symbol: "Linear",
  contains: "scale_out_features = (out_features + block_size - 1) // block_size",
});

export function kindOf(index: number): "dense" | "moe" {
  if (!Number.isInteger(index) || index < 0 || index >= layers.value) throw new RangeError("invalid main layer index");
  return index >= firstDense.value && index % moeFrequency.value === 0 ? "moe" : "dense";
}
const allLayers = () => Array.from({ length: layers.value }, (_, i) => i);
export const moeCount = derive(
  () => allLayers().filter((i) => kindOf(i) === "moe").length,
  [layers, firstDense, moeFrequency, layerRule],
);
export const denseCount = derive(() => layers.value - moeCount.value, [layers, moeCount]);
export const qDim = derive(() => nope.value + rotary.value, [nope, rotary]);
export const qWidth = derive(() => heads.value * qDim.value, [heads, qDim]);
export const kvAWidth = derive(() => kvRank.value + rotary.value, [kvRank, rotary]);
export const kvWidth = derive(() => heads.value * (nope.value + valueDim.value), [heads, nope, valueDim]);
export const outputWidth = derive(() => heads.value * valueDim.value, [heads, valueDim]);
export const perGroup = derive(() => experts.value / groups.value, [experts, groups]);
export const fusedAWidth = derive(() => qRank.value + kvAWidth.value, [qRank, kvAWidth]);
export const rotaryPairs = derive(() => rotary.value / 2, [rotary]);
const getMscale = (amount: number) => 1 + yarnCoefficient.value * amount * Math.log(ropeFactor.value);
export const attentionScale = derive(
  () => qDim.value ** -0.5 * getMscale(mscaleAll.value) ** 2,
  [qDim, ropeFactor, mscaleAll, yarnCoefficient],
);
export const rotaryAmplitude = derive(
  () => getMscale(mscale.value) / getMscale(mscaleAll.value),
  [mscale, mscaleAll, ropeFactor, yarnCoefficient],
);
const correction = (rotations: number) =>
  (rotary.value * Math.log(original.value / (rotations * 2 * Math.PI))) / (2 * Math.log(theta.value));
export const correctionLow = derive(
  () => Math.max(0, Math.floor(correction(fast.value))),
  [rotary, original, fast, theta],
);
export const correctionHigh = derive(
  () => Math.min(rotary.value - 1, Math.ceil(correction(slow.value))),
  [rotary, original, slow, theta],
);
export function yarnFrequency(pair: number) {
  if (!Number.isInteger(pair) || pair < 0 || pair >= rotaryPairs.value) throw new RangeError("invalid rotary pair");
  const ramp = Math.max(0, Math.min(1, (pair - correctionLow.value) / (correctionHigh.value - correctionLow.value)));
  const base = theta.value ** ((-2 * pair) / rotary.value);
  return base * (1 - ramp) + (base / ropeFactor.value) * ramp;
}

export function route(logits: readonly number[], bias: readonly number[]): Routing {
  if (logits.length !== experts.value || bias.length !== experts.value || ![...logits, ...bias].every(Number.isFinite))
    throw new RangeError("expected finite logits and correction biases for every routed expert");
  const scores = logits.map((x) => 1 / (1 + Math.exp(-x)));
  const corrected = scores.map((x, i) => x + bias[i]);
  const descending = (values: readonly number[]) =>
    values.map((score, index) => ({ score, index })).sort((a, b) => b.score - a.score || a.index - b.index);
  const groupScores = Array.from({ length: groups.value }, (_, g) =>
    descending(corrected.slice(g * perGroup.value, (g + 1) * perGroup.value))
      .slice(0, groupPeakCount.value)
      .reduce((a, b) => a + b.score, 0),
  );
  const chosenGroups = descending(groupScores)
    .slice(0, topGroups.value)
    .map((x) => x.index);
  const chosen = descending(corrected)
    .filter((x) => chosenGroups.includes(Math.floor(x.index / perGroup.value)))
    .slice(0, topK.value)
    .map((x) => x.index);
  const sum = chosen.reduce((s, i) => s + scores[i], 0) + routeEpsilon.value;
  return { scores, chosen, chosenGroups, weights: chosen.map((i) => (scores[i] / sum) * routeScale.value) };
}
export function syntheticRoute(seed: number, corrected: boolean) {
  if (!corrected)
    return routeToken({
      experts: experts.value,
      topK: topK.value,
      scoring: "sigmoid",
      seed,
      groups: groups.value,
      topGroups: topGroups.value,
      normalize: normalize.value,
      scale: routeScale.value,
    });
  return route(
    normals(seed, experts.value),
    corrected ? normals(seed + 1, experts.value).map((x) => x / 4) : Array(experts.value).fill(0),
  );
}

export const embeddingParams = derive(() => vocab.value * hidden.value, [vocab, hidden]);
export const headParams = derive(() => (tied.value ? 0 : embeddingParams.value), [tied, embeddingParams]);
export const attentionParams = derive(
  () =>
    hidden.value * qRank.value +
    qRank.value +
    qRank.value * qWidth.value +
    hidden.value * kvAWidth.value +
    kvRank.value +
    kvRank.value * kvWidth.value +
    outputWidth.value * hidden.value,
  [hidden, qRank, qWidth, kvAWidth, kvRank, kvWidth, outputWidth, attentionBias],
);
export const expertParams = derive(
  () => matricesPerFfn.value * hidden.value * expertWidth.value,
  [matricesPerFfn, hidden, expertWidth],
);
export const denseParams = derive(
  () => matricesPerFfn.value * hidden.value * denseWidth.value,
  [matricesPerFfn, hidden, denseWidth],
);
export const routerParams = derive(
  () => hidden.value * experts.value + experts.value,
  [hidden, experts, correctedChoice],
);
export const blockNormParams = derive(() => 2 * hidden.value, [hidden, normRule]);
export const routedParams = derive(
  () => moeCount.value * experts.value * expertParams.value,
  [moeCount, experts, expertParams],
);
export const sharedParams = derive(
  () => moeCount.value * shared.value * expertParams.value,
  [moeCount, shared, expertParams],
);
export const allAttentionParams = derive(() => layers.value * attentionParams.value, [layers, attentionParams]);
export const allDenseParams = derive(() => denseCount.value * denseParams.value, [denseCount, denseParams]);
export const allRouterParams = derive(() => moeCount.value * routerParams.value, [moeCount, routerParams]);
export const allNormParams = derive(
  () => layers.value * blockNormParams.value + hidden.value,
  [layers, blockNormParams, hidden],
);
export const moeParams = derive(
  () => (experts.value + shared.value) * expertParams.value + routerParams.value,
  [experts, shared, expertParams, routerParams],
);
export const parameterTotal = derive(
  () =>
    embeddingParams.value +
    headParams.value +
    allAttentionParams.value +
    allDenseParams.value +
    routedParams.value +
    sharedParams.value +
    allRouterParams.value +
    allNormParams.value,
  [
    embeddingParams,
    headParams,
    allAttentionParams,
    allDenseParams,
    routedParams,
    sharedParams,
    allRouterParams,
    allNormParams,
  ],
);
export const activeFfnParams = derive(
  () => (topK.value + shared.value) * expertParams.value,
  [topK, shared, expertParams],
);
export const storedFfnParams = derive(
  () => (experts.value + shared.value) * expertParams.value,
  [experts, shared, expertParams],
);
export const mtpUnique = derive(
  () =>
    mtpLayers.value *
    (attentionParams.value +
      blockNormParams.value +
      moeParams.value +
      2 * hidden.value +
      2 * hidden.value ** 2 +
      hidden.value),
  [mtpLayers, attentionParams, blockNormParams, moeParams, hidden, mtpFusion],
);
export const mtpDuplicates = derive(
  () => mtpLayers.value * (embeddingParams.value + headParams.value),
  [mtpLayers, embeddingParams, headParams],
);
const scale = (out: number, inp: number) =>
  Math.ceil(out / weightBlock.value[0]) * Math.ceil(inp / weightBlock.value[1]);
export const scaleScalars = derive(
  () =>
    (layers.value + mtpLayers.value) *
      (scale(qRank.value, hidden.value) +
        scale(qWidth.value, qRank.value) +
        scale(kvAWidth.value, hidden.value) +
        scale(kvWidth.value, kvRank.value) +
        scale(hidden.value, outputWidth.value)) +
    (moeCount.value + mtpLayers.value) *
      (experts.value + shared.value) *
      matricesPerFfn.value *
      scale(expertWidth.value, hidden.value) +
    denseCount.value * matricesPerFfn.value * scale(denseWidth.value, hidden.value),
  [
    layers,
    mtpLayers,
    qRank,
    hidden,
    qWidth,
    kvAWidth,
    kvWidth,
    kvRank,
    outputWidth,
    moeCount,
    experts,
    shared,
    matricesPerFfn,
    expertWidth,
    denseCount,
    denseWidth,
    weightBlock,
    fp8ScaleRule,
  ],
);
export const scaleTensorCount = derive(
  () =>
    (layers.value + mtpLayers.value) * attentionMatrices.value +
    ((moeCount.value + mtpLayers.value) * (experts.value + shared.value) + denseCount.value) * matricesPerFfn.value,
  [layers, mtpLayers, moeCount, experts, shared, denseCount, matricesPerFfn, attentionMatrices],
);
export const storedTotal = derive(
  () => parameterTotal.value + mtpUnique.value + mtpDuplicates.value + scaleScalars.value,
  [parameterTotal, mtpUnique, mtpDuplicates, scaleScalars],
);
export const storedDifference = derive(() => storedTotal.value - parameterTotal.value, [storedTotal, parameterTotal]);
export const parameterNote = `Main forward parameters: ${parameterTotal.value}. Hub stored total: ${storedTotal.value}; exact difference ${storedDifference.value} = ${mtpUnique.value} unique MTP parameters + ${mtpDuplicates.value} separately stored MTP embedding/head entries + ${scaleScalars.value} FP8 block-scale scalars. README_WEIGHTS documents MTP sharing; the pinned index lists separate embedding/head entries. Scale shapes use ceil(out/128) × ceil(in/128), including the nonaligned KV A projection. No weights downloaded.`;
export const eagerCacheWidth = derive(
  () => heads.value * (qDim.value + valueDim.value),
  [heads, qDim, valueDim, expandedCache],
);
export const flashCacheWidth = derive(() => heads.value * qDim.value * 2, [heads, qDim, paddedCache]);
export const cacheReduction = derive(() => eagerCacheWidth.value / kvAWidth.value, [eagerCacheWidth, kvAWidth]);
export type CacheMode = "compressed" | "eager" | "flash";
export function cacheScalars(batch: number, context: number, mode: CacheMode) {
  if (![batch, context].every((x) => Number.isInteger(x) && x > 0) || !(mode in { compressed: 1, eager: 1, flash: 1 }))
    throw new RangeError("invalid cache scenario");
  return (
    batch *
    context *
    layers.value *
    { compressed: kvAWidth.value, eager: eagerCacheWidth.value, flash: flashCacheWidth.value }[mode]
  );
}
// Teaching controls and storage assumption, not checkpoint architectural claims.
export const DEMO = {
  tokens: 24,
  query: 9,
  seed: 7,
  maxSeed: 999,
  batch: 1,
  maxBatch: 8,
  context: 256,
  pairs: [0, 8, 16, 31],
  bytes: 2,
  precision: 3,
} as const;

export const facts: Record<string, Fact> = {
  hidden,
  vocab,
  layers,
  heads,
  denseWidth,
  expertWidth,
  experts,
  shared,
  topK,
  firstDense,
  moeFrequency,
  groups,
  topGroups,
  normalize,
  scoring,
  method,
  routeScale,
  qRank,
  kvRank,
  nope,
  rotary,
  valueDim,
  epsilon,
  attentionBias,
  dropout,
  tied,
  activation,
  positions,
  theta,
  ropeType,
  ropeFactor,
  original,
  fast,
  slow,
  mscale,
  mscaleAll,
  weightBlock,
  mtpLayers,
  bos,
  eos,
  dtype,
  epSize,
  matricesPerFfn,
  groupPeakCount,
  attentionMatrices,
  routeEpsilon,
  yarnCoefficient,
  normRule,
  correctedChoice,
  layerRule,
  sharedRule,
  expandedCache,
  paddedCache,
  compressedCache,
  fusedProjection,
  mtpFusion,
  fp8ScaleRule,
  moeCount,
  denseCount,
  qDim,
  qWidth,
  kvAWidth,
  kvWidth,
  outputWidth,
  perGroup,
  fusedAWidth,
  rotaryPairs,
  attentionScale,
  rotaryAmplitude,
  correctionLow,
  correctionHigh,
  embeddingParams,
  headParams,
  attentionParams,
  expertParams,
  denseParams,
  routerParams,
  blockNormParams,
  routedParams,
  sharedParams,
  allAttentionParams,
  allDenseParams,
  allRouterParams,
  allNormParams,
  moeParams,
  parameterTotal,
  activeFfnParams,
  storedFfnParams,
  mtpUnique,
  mtpDuplicates,
  scaleScalars,
  scaleTensorCount,
  storedTotal,
  storedDifference,
  eagerCacheWidth,
  flashCacheWidth,
  cacheReduction,
};
