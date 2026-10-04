import { cfg, code, derive } from "../../kit/lib/facts";
export const hidden = cfg("hidden_size", 6144);
export const layers = cfg("num_hidden_layers", 78);
export const vocab = cfg("vocab_size", 154880);
export const heads = cfg("num_attention_heads", 64);
export const qRank = cfg("q_lora_rank", 2048);
export const kvRank = cfg("kv_lora_rank", 512);
export const nope = cfg("qk_nope_head_dim", 192);
export const rope = cfg("qk_rope_head_dim", 64);
export const valueDim = cfg("v_head_dim", 256);
export const denseWidth = cfg("intermediate_size", 12288);
export const expertWidth = cfg("moe_intermediate_size", 2048);
export const experts = cfg("n_routed_experts", 256);
export const shared = cfg("n_shared_experts", 1);
export const topK = cfg("num_experts_per_tok", 8);
export const scale = cfg("routed_scaling_factor", 2.5);
export const groups = cfg("n_group", 1);
export const topGroups = cfg("topk_group", 1);
export const normalize = cfg("norm_topk_prob", true);
export const indexHeads = cfg("index_n_heads", 32);
export const indexDim = cfg("index_head_dim", 128);
export const indexTopK = cfg("index_topk", 2048);
export const indexTypes = cfg("indexer_types", [
  "full",
  "full",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
  "full",
  "shared",
  "shared",
  "shared",
]);
export const mlpTypes = cfg("mlp_layer_types", [
  "dense",
  "dense",
  "dense",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
  "sparse",
]);
export const maxContext = cfg("max_position_embeddings", 1048576);
export const eps = cfg("rms_norm_eps", 1e-5);
export const theta = cfg("rope_parameters.rope_theta", 8000000);
export const denseFirst = cfg("first_k_dense_replace", 3);
export const indexFreq = cfg("index_topk_freq", 4);
export const indexOffset = cfg("index_skip_topk_offset", 3);
export const latentEps = code(1e-6, {
  source: "reference",
  symbol: "GlmMoeDsaRMSNorm.__init__",
  contains: "eps: float = 1e-6",
});
export const indexEps = code(1e-6, { source: "reference", symbol: "GlmMoeDsaIndexer.__init__", contains: "eps=1e-6" });
export const sigmoidRule = code("sigmoid", {
  source: "reference",
  symbol: "GlmMoeDsaTopkRouter.forward",
  contains: "router_logits.sigmoid()",
});
export const bytesPerScalar = derive(() => 2, [hidden]); // Scenario choice: BF16 payload bytes, not allocator usage.
export const qWidth = derive(() => heads.value * (nope.value + rope.value), [heads, nope, rope]);
export const vWidth = derive(() => heads.value * valueDim.value, [heads, valueDim]);
export const cacheWidth = derive(() => kvRank.value + rope.value, [kvRank, rope]);
export const expandedWidth = derive(
  () => heads.value * (nope.value + rope.value + valueDim.value),
  [heads, nope, rope, valueDim],
);
export const fullCount = derive(() => indexTypes.value.filter((x) => x === "full").length, [indexTypes]);
export const sparseCount = derive(() => mlpTypes.value.filter((x) => x === "sparse").length, [mlpTypes]);
export const embeddingParams = derive(() => 2 * vocab.value * hidden.value, [vocab, hidden]);
export const attentionParams = derive(
  () =>
    layers.value *
    (hidden.value * qRank.value +
      qRank.value +
      qRank.value * qWidth.value +
      hidden.value * cacheWidth.value +
      kvRank.value +
      kvRank.value * heads.value * (nope.value + valueDim.value) +
      vWidth.value * hidden.value),
  [layers, hidden, qRank, qWidth, cacheWidth, kvRank, heads, nope, valueDim, vWidth],
);
export const indexParams = derive(
  () =>
    fullCount.value *
    (qRank.value * indexHeads.value * indexDim.value +
      hidden.value * indexDim.value +
      2 * indexDim.value +
      hidden.value * indexHeads.value),
  [fullCount, qRank, indexHeads, indexDim, hidden],
);
export const denseParams = derive(
  () => (layers.value - sparseCount.value) * 3 * hidden.value * denseWidth.value,
  [layers, sparseCount, hidden, denseWidth],
);
export const expertParams = derive(
  () => sparseCount.value * 3 * hidden.value * expertWidth.value * (experts.value + shared.value),
  [sparseCount, hidden, expertWidth, experts, shared],
);
export const routerParams = derive(
  () => sparseCount.value * hidden.value * experts.value,
  [sparseCount, hidden, experts],
);
export const normParams = derive(() => (2 * layers.value + 1) * hidden.value, [layers, hidden]);
export const parameterTotal = derive(
  () =>
    embeddingParams.value +
    attentionParams.value +
    indexParams.value +
    denseParams.value +
    expertParams.value +
    routerParams.value +
    normParams.value,
  [embeddingParams, attentionParams, indexParams, denseParams, expertParams, routerParams, normParams],
);
export const demoTokens = derive(() => 12, [layers]);
export const seedMax = derive(() => 10000, [layers]);
export const batchMax = derive(() => topK.value, [topK]);
export const defaultContext = derive(() => indexTopK.value, [indexTopK]);
export const dialPairs = derive(() => [0, 1, rope.value / 4, rope.value / 2 - 1], [rope]);
export const kinds = [
  { id: "dense", label: "Dense · own indexer", lane: "merged" },
  { id: "full", label: "MoE · own indexer", lane: "high" },
  { id: "shared", label: "MoE · reused selection", lane: "low" },
] as const;
export const kindOf = (i: number) => (mlpTypes.value[i] === "dense" ? "dense" : indexTypes.value[i]);
export const selectionOwner = (i: number) => {
  while (i > 0 && indexTypes.value[i] === "shared") i--;
  return i;
};
export const cacheScalars = (batch: number, context: number) =>
  batch * context * (layers.value * cacheWidth.value + fullCount.value * indexDim.value);
export const facts = {
  hidden,
  layers,
  vocab,
  heads,
  qRank,
  kvRank,
  nope,
  rope,
  valueDim,
  denseWidth,
  expertWidth,
  experts,
  shared,
  topK,
  scale,
  groups,
  topGroups,
  normalize,
  indexHeads,
  indexDim,
  indexTopK,
  indexTypes,
  mlpTypes,
  maxContext,
  eps,
  theta,
  denseFirst,
  indexFreq,
  indexOffset,
  latentEps,
  indexEps,
  sigmoidRule,
  bytesPerScalar,
  qWidth,
  vWidth,
  cacheWidth,
  expandedWidth,
  fullCount,
  sparseCount,
  embeddingParams,
  attentionParams,
  indexParams,
  denseParams,
  expertParams,
  routerParams,
  normParams,
  parameterTotal,
  demoTokens,
  seedMax,
  batchMax,
  defaultContext,
  dialPairs,
};

export const mtpLayers = cfg("num_nextn_predict_layers", 1);
export const quantBlock = cfg("quantization_config.weight_block_size", [128, 128]);
export const mtpStructure = code("enorm, hnorm, eh_proj, shared_head", {
  source: "serving-mtp",
  symbol: "DeepseekV32MultiTokenPredictorLayer.__init__",
  contains: "self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)",
});
export const mtpParams = derive(
  () =>
    mtpLayers.value *
    (attentionParams.value / layers.value +
      indexParams.value / fullCount.value +
      3 * hidden.value * expertWidth.value * (experts.value + shared.value) +
      hidden.value * experts.value +
      5 * hidden.value +
      2 * hidden.value ** 2),
  [mtpLayers, mtpStructure, attentionParams, layers, indexParams, fullCount, hidden, expertWidth, experts, shared],
);
export const correctionBuffers = derive(
  () => (sparseCount.value + mtpLayers.value) * experts.value,
  [sparseCount, mtpLayers, experts],
);
export const hubAccounted = derive(
  () => parameterTotal.value + mtpParams.value + correctionBuffers.value,
  [parameterTotal, mtpParams, correctionBuffers],
);
export const parameterNote = `Hub total ${hubAccounted.value} = decoder trainable ${parameterTotal.value} + excluded MTP trainable ${mtpParams.value} + router correction buffers ${correctionBuffers.value}. Every tensor-index name is reconciled by constructor shape in reference/storage-audit.py; FP8 scale metadata is additional storage excluded from the Hub parameter total and reconciled separately against index metadata.total_size.`;
Object.assign(facts, { mtpLayers, quantBlock, mtpStructure, mtpParams, correctionBuffers, hubAccounted });

// Teaching choices, not released checkpoint dimensions. Structural modes remain live.
export const toyConfig = derive(
  () => ({
    hidden: 16,
    vocab: 24,
    heads: 2,
    qRank: 8,
    kvRank: 4,
    nope: 4,
    rope: 4,
    v: 4,
    indexHeads: 8,
    indexDim: 8,
    indexTopK: 3,
    experts: 4,
    topK: 2,
    shared: shared.value,
    denseWidth: 12,
    expertWidth: 6,
    eps: eps.value,
    theta: theta.value,
    scale: scale.value,
    indexTypes: ["full", "full", "full", "shared", "full", "shared"],
    mlpTypes: ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
  }),
  [hidden, shared, eps, theta, scale, indexTypes, mlpTypes],
);
Object.assign(facts, { toyConfig });
