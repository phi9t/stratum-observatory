import { cfg, code, derive, type Fact } from "../../kit/lib/facts";

export const hidden = cfg("text_config.hidden_size", 7168);
export const layers = cfg("text_config.num_hidden_layers", 93);
export const vocab = cfg("text_config.vocab_size", 163840);
export const heads = cfg("text_config.num_attention_heads", 96);
export const dim = cfg("text_config.linear_attn_config.head_dim", 128);
export const qRank = cfg("text_config.q_lora_rank", 1536);
export const kvRank = cfg("text_config.kv_lora_rank", 512);
export const nopeDim = cfg("text_config.qk_nope_head_dim", 128);
export const sharedKeyDim = cfg("text_config.qk_rope_head_dim", 64);
export const vDim = cfg("text_config.v_head_dim", 128);
export const experts = cfg("text_config.num_experts", 896);
export const topK = cfg("text_config.num_experts_per_token", 16);
export const shared = cfg("text_config.num_shared_experts", 2);
export const latent = cfg("text_config.routed_expert_hidden_size", 3584);
export const intermediate = cfg("text_config.moe_intermediate_size", 3072);
export const denseIntermediate = cfg("text_config.intermediate_size", 33792);
export const denseLayers = cfg("text_config.first_k_dense_replace", 1);
export const moeFreq = cfg("text_config.moe_layer_freq", 1);
export const fullLayers = cfg(
  "text_config.linear_attn_config.full_attn_layers",
  [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
);
export const kdaLayers = cfg(
  "text_config.linear_attn_config.kda_layers",
  [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41,
    42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79,
    81, 82, 83, 85, 86, 87, 89, 90, 91,
  ],
);
export const blockSize = cfg("text_config.attn_res_block_size", 12);
export const convKernel = cfg("text_config.linear_attn_config.short_conv_kernel_size", 4);
export const lowerBound = cfg("text_config.linear_attn_config.gate_lower_bound", -5.0);
export const eps = cfg("text_config.rms_norm_eps", 1e-5);
export const beta = cfg("text_config.activation_situ_beta", 4.0);
export const linearBeta = cfg("text_config.activation_situ_linear_beta", 25.0);
export const maxContext = cfg("text_config.max_position_embeddings", 1048576);
export const visionLayers = cfg("vision_config.vt_num_hidden_layers", 27);
export const visionWidth = cfg("vision_config.vt_hidden_size", 1024);
export const patch = cfg("vision_config.patch_size", 14);
export const visionMlp = cfg("vision_config.vt_intermediate_size", 4096);
export const visionQkv = cfg("vision_config.qkv_hidden_size", 1536);
export const visionPosH = cfg("vision_config.init_pos_emb_height", 64);
export const visionPosW = cfg("vision_config.init_pos_emb_width", 64);
export const mergeKernel = cfg("vision_config.merge_kernel_size", [2, 2]);
export const normLatentEps = code(1e-6, { source: "reference", symbol: "KimiRMSNorm.__init__", contains: "eps=1e-6" });
export const routingEps = code(1e-20, { source: "reference", symbol: "KimiMoEGate.forward", contains: "+ 1e-20" });
export const rgb = code(3, {
  source: "multimodal",
  symbol: "MoonVision3dPatchEmbed.__init__",
  contains: "in_dim: int = 3",
});
export const convBias = code(false, {
  source: "conv-module",
  symbol: "ShortConvolution.__init__",
  contains: "bias: bool = False",
});
export const noTextRope = code(true, {
  source: "reference",
  symbol: "KimiMLAAttention.__init__",
  contains: "self.rotary_emb = None",
});
export const oneBased = code(true, {
  source: "configuration",
  symbol: "KimiLinearConfig.is_kda_layer",
  contains: "(layer_idx + 1)",
});
export const qDim = derive(() => nopeDim.value + sharedKeyDim.value, [nopeDim, sharedKeyDim]);
export const wide = derive(() => heads.value * dim.value, [heads, dim]);
export const fullCount = derive(() => fullLayers.value.length, [fullLayers]);
export const kdaCount = derive(() => kdaLayers.value.length, [kdaLayers]);
export const moeCount = derive(
  () => Array.from({ length: layers.value }, (_, i) => i).filter(isMoe).length,
  [layers, denseLayers, moeFreq],
);
export const sharedWidth = derive(() => intermediate.value * shared.value, [intermediate, shared]);
export const candidateCount = (i: number) => Math.ceil(i / blockSize.value);
export function isKda(i: number) {
  return kdaLayers.value.includes(i + 1);
}
export function isMoe(i: number) {
  return i >= denseLayers.value && i % moeFreq.value === 0;
}
export function kindOf(i: number) {
  return `${isKda(i) ? "kda" : "mla"}${isMoe(i) ? "-moe" : "-dense"}`;
}
export const logicalCacheWidth = derive(() => heads.value * (qDim.value + vDim.value), [heads, qDim, vDim]);
export const absorbedCacheWidth = derive(() => kvRank.value + sharedKeyDim.value, [kvRank, sharedKeyDim]);
export const recurrenceScalars = derive(() => heads.value * dim.value * dim.value, [heads, dim]);
export const convolutionScalars = derive(() => 3 * wide.value * convKernel.value, [wide, convKernel]);
export const bankBlocks = derive(() => Math.ceil(layers.value / blockSize.value), [layers, blockSize]);
export const embeddingParams = derive(() => 2 * vocab.value * hidden.value, [vocab, hidden]);
export const mlaParams = derive(
  () =>
    hidden.value * qRank.value +
    qRank.value +
    qRank.value * heads.value * qDim.value +
    hidden.value * absorbedCacheWidth.value +
    kvRank.value +
    kvRank.value * heads.value * (nopeDim.value + vDim.value) +
    2 * wide.value * hidden.value,
  [hidden, qRank, heads, qDim, absorbedCacheWidth, kvRank, nopeDim, vDim, wide],
);
export const kdaParams = derive(
  () =>
    5 * hidden.value * wide.value +
    3 * wide.value * convKernel.value +
    heads.value +
    hidden.value * dim.value +
    dim.value * wide.value +
    wide.value +
    hidden.value * heads.value +
    dim.value,
  [hidden, wide, convKernel, heads, dim],
);
export const expertParams = derive(
  () => 3 * latent.value * intermediate.value * experts.value,
  [latent, intermediate, experts],
);
export const routerParams = derive(() => hidden.value * experts.value + experts.value, [hidden, experts]);
export const sharedParams = derive(() => 3 * hidden.value * sharedWidth.value, [hidden, sharedWidth]);
export const latentParams = derive(() => 2 * hidden.value * latent.value + latent.value, [hidden, latent]);
export const denseParams = derive(() => 3 * hidden.value * denseIntermediate.value, [hidden, denseIntermediate]);
export const normsParams = derive(() => 6 * layers.value * hidden.value + 3 * hidden.value, [layers, hidden]);
export const ledger = derive(
  () => [
    {
      label: "Embedding and untied head",
      value: embeddingParams.value,
      detail: "Separate vocabulary × stream matrices",
      lane: "merged" as const,
    },
    {
      label: "KDA attention",
      value: kdaCount.value * kdaParams.value,
      detail: "Projections, convolution, decay, output norm",
      lane: "high" as const,
    },
    {
      label: "Full attention",
      value: fullCount.value * mlaParams.value,
      detail: "Query/KV latent projections and output gate",
      lane: "high" as const,
    },
    {
      label: "Routed expert matrices",
      value: moeCount.value * expertParams.value,
      detail: "All resident experts, not only selected experts",
      lane: "low" as const,
    },
    {
      label: "Shared and dense FFN",
      value: moeCount.value * sharedParams.value + denseLayers.value * denseParams.value,
      detail: "Always-on full-stream feed-forward paths",
      lane: "low" as const,
    },
    {
      label: "Router and latent bridge",
      value: moeCount.value * (routerParams.value + latentParams.value),
      detail: "Router/bias; compress, normalize sum, expand",
      lane: "instance" as const,
    },
    {
      label: "Depth scoring and norms",
      value: normsParams.value,
      detail: "Sublayer norms, candidate scoring gain and projection, final norms",
      lane: "skip" as const,
    },
  ],
  [
    embeddingParams,
    kdaCount,
    kdaParams,
    fullCount,
    mlaParams,
    moeCount,
    expertParams,
    sharedParams,
    denseLayers,
    denseParams,
    routerParams,
    latentParams,
    normsParams,
  ],
);
export const parameterTotal = derive(() => ledger.value.reduce((a, r) => a + r.value, 0), [ledger]);
export const visionMerged = derive(
  () => visionWidth.value * mergeKernel.value.reduce((a, b) => a * b, 1),
  [visionWidth, mergeKernel],
);
export const visionParams = derive(
  () =>
    rgb.value * patch.value ** 2 * visionWidth.value +
    visionPosH.value * visionPosW.value * visionWidth.value +
    visionLayers.value *
      (4 * visionWidth.value * visionQkv.value + 2 * visionWidth.value * visionMlp.value + 2 * visionWidth.value) +
    visionWidth.value +
    visionMerged.value ** 2 +
    visionMerged.value * hidden.value +
    hidden.value,
  [rgb, patch, visionWidth, visionPosH, visionPosW, visionLayers, visionQkv, visionMlp, visionMerged, hidden],
);
// Hub API metadata is reported separately, not an architecture fact or a fabricated reconciliation.
export const hubTotal = 2779931837184;
export const unexplainedDifference = hubTotal - parameterTotal.value - visionParams.value;
export const DEMO = { tokens: 8, seedMax: 1000, batchMax: 16, contextDefault: 1024, grid: 16, bytes: 2 };
export const TOY = {
  hidden: 12,
  layers: 5,
  vocab: 24,
  heads: 2,
  dim: 4,
  qRank: 6,
  kvRank: 4,
  nopeDim: 4,
  sharedKeyDim: 2,
  vDim: 4,
  experts: 6,
  topK: 2,
  latent: 6,
  intermediate: 8,
  denseIntermediate: 16,
  shared: 2,
  blockSize: 2,
  convKernel: 4,
  lowerBound: -5,
  eps: eps.value,
  normLatentEps: normLatentEps.value,
  beta: beta.value,
  linearBeta: linearBeta.value,
  kda: [1, 2, 4],
  denseLayers: 1,
};
export const serializationPadding = derive(() => kdaCount.value * (dim.value - heads.value), [kdaCount, dim, heads]);
export const facts: Record<string, Fact> = {
  serializationPadding,
  hidden,
  layers,
  vocab,
  heads,
  dim,
  qRank,
  kvRank,
  nopeDim,
  sharedKeyDim,
  vDim,
  experts,
  topK,
  shared,
  latent,
  intermediate,
  denseIntermediate,
  denseLayers,
  moeFreq,
  fullLayers,
  kdaLayers,
  blockSize,
  convKernel,
  lowerBound,
  eps,
  beta,
  linearBeta,
  maxContext,
  visionLayers,
  visionWidth,
  patch,
  visionMlp,
  visionQkv,
  visionPosH,
  visionPosW,
  mergeKernel,
  normLatentEps,
  routingEps,
  rgb,
  convBias,
  noTextRope,
  oneBased,
  qDim,
  wide,
  fullCount,
  kdaCount,
  moeCount,
  sharedWidth,
  logicalCacheWidth,
  absorbedCacheWidth,
  recurrenceScalars,
  convolutionScalars,
  bankBlocks,
  embeddingParams,
  mlaParams,
  kdaParams,
  expertParams,
  routerParams,
  sharedParams,
  latentParams,
  denseParams,
  normsParams,
  ledger,
  parameterTotal,
  visionMerged,
  visionParams,
};
export const parameterNote = `Publisher text decoder: ${parameterTotal.value} logical parameters. Hub total ${hubTotal} exceeds this by ${hubTotal - parameterTotal.value}: vision tower/projector ${visionParams.value}, leaving a remainder of ${serializationPadding.value}. That remainder is an inference, not a verified fact: a comment in SGLang KimiK3DeltaAttention says the checkpoint stores A_log as head_dim entries and narrows to num_heads, and (${dim.value}-${heads.value}) × ${kdaCount.value} = ${serializationPadding.value} matches it, but the publisher code allocates num_heads entries and no tensor index is pinned. Packed storage bytes are not parameter count.`;
