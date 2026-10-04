import { cfg, code, derive } from "../../kit";
import { attentionScale, type KimiConfig } from "./reference/model";
import fixture from "./reference/fixture.json";
import storage from "./storage-summary.json";
export const vocab = cfg("vocab_size", 163840);
export const hidden = cfg("hidden_size", 7168);
export const layers = cfg("num_hidden_layers", 61);
export const heads = cfg("num_attention_heads", 64);
export const qRank = cfg("q_lora_rank", 1536);
export const kvRank = cfg("kv_lora_rank", 512);
export const nope = cfg("qk_nope_head_dim", 128);
export const rope = cfg("qk_rope_head_dim", 64);
export const v = cfg("v_head_dim", 128);
export const dense = cfg("intermediate_size", 18432);
export const expert = cfg("moe_intermediate_size", 2048);
export const experts = cfg("n_routed_experts", 384);
export const shared = cfg("n_shared_experts", 1);
export const topK = cfg("num_experts_per_tok", 8);
export const firstDense = cfg("first_k_dense_replace", 1);
export const eps = cfg("rms_norm_eps", 1e-6);
export const theta = cfg("rope_theta", 50000.0);
export const scale = cfg("routed_scaling_factor", 2.827);
export const groups = cfg("n_group", 1);
export const topGroups = cfg("topk_group", 1);
export const frequency = cfg("moe_layer_freq", 1);
export const context = cfg("max_position_embeddings", 131072);
export const factor = cfg("rope_scaling.factor", 32.0);
export const original = cfg("rope_scaling.original_max_position_embeddings", 4096);
export const betaFast = cfg("rope_scaling.beta_fast", 1.0);
export const betaSlow = cfg("rope_scaling.beta_slow", 1.0);
export const mscale = cfg("rope_scaling.mscale", 1.0);
export const allDim = cfg("rope_scaling.mscale_all_dim", 1.0);
export const blocks = cfg("quantization_config.weight_block_size", [128, 128]);
export const scoring = cfg("scoring_func", "sigmoid");
export const method = cfg("topk_method", "noaux_tc");
export const normalize = cfg("norm_topk_prob", true);
export const tied = cfg("tie_word_embeddings", false);
export const activation = cfg("hidden_act", "silu");
export const dropout = cfg("attention_dropout", 0.0);
export const bos = cfg("bos_token_id", 163584);
export const eos = cfg("eos_token_id", 163585);
export const nextn = cfg("num_nextn_predict_layers", 0);
export const groupScoreCount = code(2, { source: "reference", symbol: "MoEGate", contains: "topk(2, dim=-1)" });
export const yarnCoefficient = code(0.1, { source: "reference", symbol: "yarn_get_mscale", contains: "0.1 * mscale" });
export const latentNormEps = code(1e-6, { source: "reference", symbol: "DeepseekV3RMSNorm", contains: "eps=1e-6" });
export const correctionDenominator = code(1e-20, { source: "reference", symbol: "MoEGate", contains: "+ 1e-20" });
export const real: KimiConfig = {
  vocab: vocab.value,
  hidden: hidden.value,
  layers: layers.value,
  heads: heads.value,
  qRank: qRank.value,
  kvRank: kvRank.value,
  nope: nope.value,
  rope: rope.value,
  v: v.value,
  dense: dense.value,
  expert: expert.value,
  experts: experts.value,
  shared: shared.value,
  topK: topK.value,
  firstDense: firstDense.value,
  eps: eps.value,
  theta: theta.value,
  factor: factor.value,
  original: original.value,
  betaFast: betaFast.value,
  betaSlow: betaSlow.value,
  mscale: mscale.value,
  allDim: allDim.value,
  scale: scale.value,
};
export const kindOf = (index: number) => (index >= firstDense.value && index % frequency.value === 0 ? "moe" : "dense");
export const moeLayers = derive(
  () => Array.from({ length: layers.value }, (_, i) => i).filter((i) => kindOf(i) === "moe").length,
  [layers, firstDense, frequency],
);
export const denseLayers = derive(() => layers.value - moeLayers.value, [layers, moeLayers]);
export const qWidth = derive(() => heads.value * (nope.value + rope.value), [heads, nope, rope]);
export const kvWidth = derive(() => heads.value * (nope.value + v.value), [heads, nope, v]);
export const qHead = derive(() => nope.value + rope.value, [nope, rope]);
export const valueWidth = derive(() => heads.value * v.value, [heads, v]);
export const latentWidth = derive(() => kvRank.value + rope.value, [kvRank, rope]);
export const expandedCache = derive(() => heads.value * (qHead.value + v.value), [heads, qHead, v]);
export const cacheRatio = derive(() => expandedCache.value / latentWidth.value, [expandedCache, latentWidth]);
export const selectedFraction = derive(() => topK.value / experts.value, [topK, experts]);
export const softmaxScale = derive(() => attentionScale(real), [qHead, factor, allDim, yarnCoefficient]);
export const embeddingParams = derive(() => vocab.value * hidden.value, [vocab, hidden]);
export const attentionPerLayer = derive(
  () =>
    hidden.value * qRank.value +
    qRank.value +
    qRank.value * qWidth.value +
    hidden.value * latentWidth.value +
    kvRank.value +
    kvRank.value * kvWidth.value +
    valueWidth.value * hidden.value,
  [hidden, qRank, qWidth, latentWidth, kvRank, kvWidth, valueWidth],
);
export const attentionParams = derive(() => layers.value * attentionPerLayer.value, [layers, attentionPerLayer]);
export const expertParams = derive(() => 3 * hidden.value * expert.value, [hidden, expert]);
export const routedParams = derive(
  () => moeLayers.value * experts.value * expertParams.value,
  [moeLayers, experts, expertParams],
);
export const sharedParams = derive(
  () => moeLayers.value * shared.value * expertParams.value,
  [moeLayers, shared, expertParams],
);
export const denseParams = derive(
  () => denseLayers.value * 3 * hidden.value * dense.value,
  [denseLayers, hidden, dense],
);
export const routerParams = derive(
  () => moeLayers.value * (experts.value * hidden.value + experts.value),
  [moeLayers, experts, hidden],
);
export const normParams = derive(() => (layers.value * 2 + 1) * hidden.value, [layers, hidden]);
export const parameterTotal = derive(
  () =>
    embeddingParams.value * 2 +
    attentionParams.value +
    routedParams.value +
    sharedParams.value +
    denseParams.value +
    routerParams.value +
    normParams.value,
  [embeddingParams, attentionParams, routedParams, sharedParams, denseParams, routerParams, normParams],
);
export const activeParams = derive(
  () => parameterTotal.value - routedParams.value + moeLayers.value * topK.value * expertParams.value,
  [parameterTotal, routedParams, moeLayers, topK, expertParams],
);
export const scaleScalars = derive(() => {
  const grid = (out: number, input: number) => Math.ceil(out / blocks.value[0]) * Math.ceil(input / blocks.value[1]);
  return (
    layers.value *
      (grid(qRank.value, hidden.value) +
        grid(qWidth.value, qRank.value) +
        grid(latentWidth.value, hidden.value) +
        grid(kvWidth.value, kvRank.value) +
        grid(hidden.value, valueWidth.value)) +
    denseLayers.value * (2 * grid(dense.value, hidden.value) + grid(hidden.value, dense.value)) +
    moeLayers.value *
      (experts.value + shared.value) *
      (2 * grid(expert.value, hidden.value) + grid(hidden.value, expert.value))
  );
}, [
  layers,
  blocks,
  qRank,
  hidden,
  qWidth,
  latentWidth,
  kvWidth,
  kvRank,
  valueWidth,
  denseLayers,
  dense,
  moeLayers,
  experts,
  shared,
  expert,
]);
export const savedFrequencyWidth = derive(() => storage.observed_frequency_widths[0], [rope]);
export const frequencyScalars = derive(() => layers.value * savedFrequencyWidth.value, [layers, savedFrequencyWidth]);
export const storedTotal = derive(
  () => parameterTotal.value + frequencyScalars.value,
  [parameterTotal, frequencyScalars],
);
export const parameterNote = `The hub total is ${storedTotal.value}: ${parameterTotal.value} learned parameters plus ${frequencyScalars.value} saved rotary inverse-frequency entries. Metadata-only headers of the dense and first expert shards show ${savedFrequencyWidth.value} entries per saved buffer; repeating that over ${layers.value} indexed buffers exactly accounts for the difference. The remaining buffer shapes were not individually inspected. The runtime reference rebuilds ${rope.value / 2} frequencies per layer as nonpersistent state. The hub total excludes weight_scale_inv metadata; the pinned index names these tensors and the config's block grid implies ${scaleScalars.value} additional inverse-scale scalars. No weight payload was fetched.`;
export const rotaryPairs = derive(() => [0, rope.value / 16, rope.value / 4 - 1, rope.value / 2 - 1], [rope]);

export const toy = fixture.config as KimiConfig;
export const demo = {
  tokens: 24,
  scaleTokens: 16,
  seedMax: 1000,
  batchMax: 16,
  defaultContext: 4096,
  bytes: 2,
  radDigits: 2,
  gain: 3,
};
export const cacheScalars = (batch: number, tokens: number, latent = true) => {
  if (!Number.isInteger(batch) || batch < 1 || !Number.isInteger(tokens) || tokens < 1 || tokens > context.value)
    throw new RangeError("Invalid cache scenario");
  return batch * tokens * layers.value * (latent ? latentWidth.value : expandedCache.value);
};
export const ledger = [
  {
    label: "Token embedding",
    detail: "Vocabulary lookup, untied",
    value: embeddingParams.value,
    lane: "merged" as const,
  },
  {
    label: "Attention",
    detail: "Low-rank projections, latent gains and output projection",
    value: attentionParams.value,
    lane: "high" as const,
  },
  { label: "Dense FFN", detail: "Gate, up and down in dense layers", value: denseParams.value, lane: "low" as const },
  {
    label: "Routed experts",
    detail: "All logical routed expert weights",
    value: routedParams.value,
    lane: "low" as const,
  },
  { label: "Shared experts", detail: "Always-on expert weights", value: sharedParams.value, lane: "low" as const },
  {
    label: "Router",
    detail: "Score matrix and selection correction bias",
    value: routerParams.value,
    lane: "instance" as const,
  },
  { label: "Stream norms", detail: "Sublayer gains and final gain", value: normParams.value, lane: "merged" as const },
  {
    label: "Output head",
    detail: "Separate vocabulary projection",
    value: embeddingParams.value,
    lane: "merged" as const,
  },
];
export const facts = {
  vocab,
  hidden,
  layers,
  heads,
  qRank,
  kvRank,
  nope,
  rope,
  v,
  dense,
  expert,
  experts,
  shared,
  topK,
  firstDense,
  eps,
  theta,
  scale,
  groups,
  topGroups,
  frequency,
  context,
  factor,
  original,
  betaFast,
  betaSlow,
  mscale,
  allDim,
  blocks,
  scoring,
  method,
  normalize,
  tied,
  activation,
  dropout,
  bos,
  eos,
  nextn,
  groupScoreCount,
  yarnCoefficient,
  latentNormEps,
  correctionDenominator,
  moeLayers,
  denseLayers,
  qWidth,
  kvWidth,
  qHead,
  valueWidth,
  latentWidth,
  expandedCache,
  cacheRatio,
  selectedFraction,
  softmaxScale,
  embeddingParams,
  attentionPerLayer,
  attentionParams,
  expertParams,
  routedParams,
  sharedParams,
  denseParams,
  routerParams,
  normParams,
  parameterTotal,
  activeParams,
  scaleScalars,
  savedFrequencyWidth,
  frequencyScalars,
  rotaryPairs,
  storedTotal,
};
