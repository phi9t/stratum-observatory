import { cfg, code, derive, type Fact } from "../../kit/lib/facts";

export const architecture = cfg("architectures", ["Glm5NextForConditionalGeneration"]);
export const modelType = cfg("model_type", "glm5_next");
export const textModelType = cfg("text_config.model_type", "glm5_next_text");
export const hidden = cfg("text_config.hidden_size", 4096);
export const vocab = cfg("text_config.vocab_size", 154880);
export const layers = cfg("text_config.num_hidden_layers", 45);
export const nextn = cfg("text_config.num_nextn_predict_layers", 1);
export const heads = cfg("text_config.num_attention_heads", 64);
export const kvHeads = cfg("text_config.num_key_value_heads", 64);
export const qRank = cfg("text_config.q_lora_rank", 1536);
export const kvRank = cfg("text_config.kv_lora_rank", 512);
export const nopeDim = cfg("text_config.qk_nope_head_dim", 256);
export const ropeDim = cfg("text_config.qk_rope_head_dim", 0);
export const qkDim = cfg("text_config.qk_head_dim", 256);
export const valueDim = cfg("text_config.v_head_dim", 256);
export const mlaUseNope = cfg("text_config.mla_use_nope", true);
export const denseIntermediate = cfg("text_config.intermediate_size", 12288);
export const moeIntermediate = cfg("text_config.moe_intermediate_size", 2048);
export const denseFirst = cfg("text_config.first_k_dense_replace", 3);
export const experts = cfg("text_config.n_routed_experts", 288);
export const topK = cfg("text_config.num_experts_per_tok", 8);
export const sharedExperts = cfg("text_config.n_shared_experts", 1);
export const scoring = cfg("text_config.scoring_func", "sigmoid");
export const routerScale = cfg("text_config.routed_scaling_factor", 2.5);
export const routerGroups = cfg("text_config.n_group", 1);
export const routerTopGroups = cfg("text_config.topk_group", 1);
export const normTopk = cfg("text_config.norm_topk_prob", true);
export const routerDtype = cfg("text_config.moe_router_dtype", "float32");
export const swigluLimit = cfg("text_config.swiglu_limit", 10.0);
export const linearHeads = cfg("text_config.linear_attn_config.num_heads", 64);
export const linearHeadDim = cfg("text_config.linear_attn_config.head_dim", 128);
export const convKernel = cfg("text_config.linear_attn_config.short_conv_kernel_size", 4);
export const gateLowerBound = cfg("text_config.linear_attn_config.gate_lower_bound", -5.0);
export const kdaLayers = cfg(
  "text_config.linear_attn_config.kda_layers",
  [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40,
    41, 42, 44,
  ],
);
export const sparseLayers = cfg(
  "text_config.linear_attn_config.full_attn_layers",
  [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
);
export const layerTypes = cfg("text_config.layer_types", [
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "deepseek_sparse_attention",
  "linear_attention",
]);
export const mlpLayerTypes = cfg("text_config.mlp_layer_types", [
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
]);
export const indexerTypes = cfg("text_config.indexer_types", [
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
  "full",
]);
export const indexHeads = cfg("text_config.index_n_heads", 32);
export const indexHeadDim = cfg("text_config.index_head_dim", 128);
export const indexTopK = cfg("text_config.index_topk", 2048);
export const indexKpool = cfg("text_config.index_kpool", 4);
export const indexKpoolCompress = cfg("text_config.index_kpool_compress", true);
export const indexTail = cfg("text_config.index_kpool_always_select_tail", true);
export const indexShareForMtp = cfg("text_config.index_share_for_mtp_iteration", true);
export const mhc = cfg("text_config.mhc", true);
export const hcMult = cfg("text_config.hc_mult", 4);
export const hcEps = cfg("text_config.hc_eps", 1e-6);
export const hcSinkhorn = cfg("text_config.hc_sinkhorn_iters", 20);
export const rmsNormEps = cfg("text_config.rms_norm_eps", 1e-5);
export const maxContext = cfg("text_config.max_position_embeddings", 1048576);
export const tieWordEmbeddings = cfg("text_config.tie_word_embeddings", false);
export const visionDepth = cfg("vision_config.depth", 24);
export const visionHidden = cfg("vision_config.hidden_size", 1024);
export const visionHeads = cfg("vision_config.num_heads", 16);
export const visionPatch = cfg("vision_config.patch_size", 14);
export const visionTemporalPatch = cfg("vision_config.temporal_patch_size", 2);
export const visionChannels = cfg("vision_config.in_channels", 3);
export const visionIntermediate = cfg("vision_config.intermediate_size", 4096);
export const visionOut = cfg("vision_config.out_hidden_size", 4096);
export const visionProjectionIntermediate = cfg("vision_config.projection_intermediate_size", 10240);
export const visionMerge = cfg("vision_config.spatial_merge_size", 2);

export const routerUsesSigmoid = code("sigmoid", {
  source: "reference",
  symbol: "Glm5NextTextTopkRouter.forward",
  contains: "scores = router_logits.sigmoid()",
});
export const routerUsesCorrectionBias = code(true, {
  source: "reference",
  symbol: "Glm5NextTextTopkRouter.forward",
  contains: "scores_for_choice = scores + self.e_score_correction_bias",
});
export const sharedExpertBypassesRouter = code(true, {
  source: "reference",
  symbol: "Glm5NextTextMoE.forward",
  contains: "hidden_states = hidden_states + self.shared_experts(residuals)",
});
export const indexPoolWeighted = code(true, {
  source: "reference",
  symbol: "Glm5NextTextIndexer.get_pooled_states",
  contains: "logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]",
});
export const textNopeForward = code(true, {
  source: "reference",
  symbol: "Glm5NextTextModel.forward",
  contains: "position_embeddings=None",
});
export const boundedKdaGate = code(true, {
  source: "reference",
  symbol: "Glm5NextTextForgetGate.forward",
  contains: "return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)",
});
export const mhcProjectionRule = code(true, {
  source: "reference",
  symbol: "Glm5NextTextHyperConnection.__init__",
  contains: "concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult",
});
export const vllmTailBuffer = code(true, {
  source: "serving",
  symbol: "Glm5NextModel.__init__",
  contains: "buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)",
});
export const vllmMhcFusion = code(true, {
  source: "serving",
  symbol: "Glm5NextDecoderLayer.forward",
  contains: "hc_post with this layer's attn hc_pre into one kernel",
});
export const mtpTopkReuse = code(true, {
  source: "serving_mtp",
  symbol: "Glm5NextMultiTokenPredictor.set_skip_topk",
  contains: "index_share_for_mtp_iteration: step 0 computes top-k, steps 1+ reuse.",
});

export const bytesPerBf16 = derive(() => 2, [hidden]);
export const qWidth = derive(() => heads.value * qkDim.value, [heads, qkDim]);
export const valueWidth = derive(() => heads.value * valueDim.value, [heads, valueDim]);
export const kdaWidth = derive(() => linearHeads.value * linearHeadDim.value, [linearHeads, linearHeadDim]);
export const sparseAttentionCount = derive(() => sparseLayers.value.length, [sparseLayers]);
export const kdaCount = derive(() => kdaLayers.value.length, [kdaLayers]);
export const denseLayerCount = derive(
  () => mlpLayerTypes.value.filter((kind) => kind === "dense").length,
  [mlpLayerTypes],
);
export const moeLayerCount = derive(
  () => mlpLayerTypes.value.filter((kind) => kind === "sparse").length,
  [mlpLayerTypes],
);
export const allSparseFullIndexer = derive(
  () => sparseLayers.value.every((layer) => indexerTypes.value[layer] === "full"),
  [sparseLayers, indexerTypes],
);
export const indexPoolsSelected = derive(() => indexTopK.value / indexKpool.value, [indexTopK, indexKpool]);
export const indexOutputWidth = derive(() => indexTopK.value + indexKpool.value - 1, [indexTopK, indexKpool]);
export const mHcStreamWidth = derive(() => hidden.value * hcMult.value, [hidden, hcMult]);
export const mHcMixWidth = derive(() => (2 + hcMult.value) * hcMult.value, [hcMult]);
export const mHcSavedPerLayerBytesPerToken = derive(
  () => hcMult.value * hidden.value * bytesPerBf16.value,
  [hcMult, hidden, bytesPerBf16],
);
export const mHcSavedBytesPerToken = derive(
  () => layers.value * hcMult.value * hidden.value * bytesPerBf16.value,
  [layers, hcMult, hidden, bytesPerBf16],
);
export const mHcSavedMiBPerToken = derive(() => mHcSavedBytesPerToken.value / 1024 ** 2, [mHcSavedBytesPerToken]);
export const currentIndexerScoreScalars = derive(() => indexHeads.value / indexKpool.value, [indexHeads, indexKpool]);

export function isSparseAttentionLayer(index: number): boolean {
  return layerTypes.value[index] === "deepseek_sparse_attention";
}

export function isKdaLayer(index: number): boolean {
  return layerTypes.value[index] === "linear_attention";
}

export function isMoELayer(index: number): boolean {
  return mlpLayerTypes.value[index] === "sparse";
}

export function layerKind(index: number): "kda-dense" | "kda-moe" | "sparse-moe" {
  if (isSparseAttentionLayer(index)) {
    return "sparse-moe";
  }
  return isMoELayer(index) ? "kda-moe" : "kda-dense";
}

export const embeddingParameters = derive(() => 2 * vocab.value * hidden.value, [vocab, hidden]);
export const kdaPerLayerParameters = derive(
  () =>
    3 * hidden.value * kdaWidth.value +
    3 * kdaWidth.value * convKernel.value +
    hidden.value * linearHeadDim.value +
    linearHeadDim.value * kdaWidth.value +
    kdaWidth.value +
    linearHeads.value +
    hidden.value * linearHeads.value +
    hidden.value * linearHeadDim.value +
    linearHeadDim.value * kdaWidth.value +
    linearHeadDim.value +
    kdaWidth.value * hidden.value,
  [hidden, kdaWidth, convKernel, linearHeadDim, linearHeads],
);
export const kdaParameters = derive(
  () => kdaCount.value * kdaPerLayerParameters.value,
  [kdaCount, kdaPerLayerParameters],
);
export const mlaPerLayerParameters = derive(
  () =>
    hidden.value * qRank.value +
    qRank.value +
    qRank.value * heads.value * (nopeDim.value + ropeDim.value) +
    hidden.value * (kvRank.value + ropeDim.value) +
    kvRank.value +
    kvRank.value * heads.value * (nopeDim.value + valueDim.value) +
    heads.value * valueDim.value * hidden.value,
  [hidden, qRank, heads, nopeDim, ropeDim, kvRank, valueDim],
);
export const sparseMlaParameters = derive(
  () => sparseAttentionCount.value * mlaPerLayerParameters.value,
  [sparseAttentionCount, mlaPerLayerParameters],
);
export const indexerPerLayerParameters = derive(
  () =>
    qRank.value * indexHeads.value * indexHeadDim.value +
    hidden.value * indexHeadDim.value +
    2 * indexHeadDim.value +
    hidden.value * indexHeads.value +
    indexHeadDim.value * hidden.value +
    indexKpool.value * indexHeadDim.value,
  [qRank, indexHeads, indexHeadDim, hidden, indexKpool],
);
export const indexerParameters = derive(
  () => sparseAttentionCount.value * indexerPerLayerParameters.value,
  [sparseAttentionCount, indexerPerLayerParameters],
);
export const expertMlpParameters = derive(() => 3 * hidden.value * moeIntermediate.value, [hidden, moeIntermediate]);
export const routedExpertParameters = derive(
  () => moeLayerCount.value * experts.value * expertMlpParameters.value,
  [moeLayerCount, experts, expertMlpParameters],
);
export const sharedExpertParameters = derive(
  () => moeLayerCount.value * sharedExperts.value * expertMlpParameters.value,
  [moeLayerCount, sharedExperts, expertMlpParameters],
);
export const denseMlpParameters = derive(
  () => denseLayerCount.value * 3 * hidden.value * denseIntermediate.value,
  [denseLayerCount, hidden, denseIntermediate],
);
export const routerStoredTensors = derive(
  () => moeLayerCount.value * experts.value * (hidden.value + 1),
  [moeLayerCount, experts, hidden],
);
export const mhcParameters = derive(
  () => layers.value * 2 * (mHcMixWidth.value * mHcStreamWidth.value + mHcMixWidth.value + 3),
  [layers, mHcMixWidth, mHcStreamWidth],
);
export const normParameters = derive(() => (2 * layers.value + 1) * hidden.value, [layers, hidden]);
export const textBackboneStoredParameters = derive(
  () =>
    embeddingParameters.value +
    kdaParameters.value +
    sparseMlaParameters.value +
    indexerParameters.value +
    denseMlpParameters.value +
    routedExpertParameters.value +
    sharedExpertParameters.value +
    routerStoredTensors.value +
    mhcParameters.value +
    normParameters.value,
  [
    embeddingParameters,
    kdaParameters,
    sparseMlaParameters,
    indexerParameters,
    denseMlpParameters,
    routedExpertParameters,
    sharedExpertParameters,
    routerStoredTensors,
    mhcParameters,
    normParameters,
  ],
);
export const mtpParameters = derive(
  () =>
    nextn.value *
    (experts.value * expertMlpParameters.value +
      sharedExperts.value * expertMlpParameters.value +
      mlaPerLayerParameters.value +
      indexerPerLayerParameters.value +
      hidden.value * (2 * hidden.value) +
      3 * hidden.value +
      experts.value * (hidden.value + 1) +
      2 * hidden.value),
  [nextn, experts, expertMlpParameters, sharedExperts, mlaPerLayerParameters, indexerPerLayerParameters, hidden],
);
export const visionPatchParameters = derive(
  () =>
    visionHidden.value * visionChannels.value * visionTemporalPatch.value * visionPatch.value ** 2 + visionHidden.value,
  [visionHidden, visionChannels, visionTemporalPatch, visionPatch],
);
export const visionAttentionPerBlock = derive(
  () =>
    visionHidden.value * (3 * visionHidden.value) +
    3 * visionHidden.value +
    visionHidden.value * visionHidden.value +
    visionHidden.value +
    2 * (visionHidden.value / visionHeads.value),
  [visionHidden, visionHeads],
);
export const visionMlpPerBlock = derive(
  () =>
    2 * visionHidden.value * visionIntermediate.value +
    2 * visionIntermediate.value +
    visionIntermediate.value * visionHidden.value +
    visionHidden.value,
  [visionHidden, visionIntermediate],
);
export const visionBlockParameters = derive(
  () => visionDepth.value * (2 * visionHidden.value + visionAttentionPerBlock.value + visionMlpPerBlock.value),
  [visionDepth, visionHidden, visionAttentionPerBlock, visionMlpPerBlock],
);
export const visionDownsampleParameters = derive(
  () => visionOut.value * visionHidden.value * visionMerge.value ** 2 + visionOut.value,
  [visionOut, visionHidden, visionMerge],
);
export const visionMergerParameters = derive(
  () =>
    visionOut.value * visionOut.value +
    2 * visionOut.value +
    2 * visionOut.value * visionProjectionIntermediate.value +
    visionProjectionIntermediate.value * visionOut.value,
  [visionOut, visionProjectionIntermediate],
);
export const visionParameters = derive(
  () =>
    visionPatchParameters.value +
    visionBlockParameters.value +
    visionHidden.value +
    visionDownsampleParameters.value +
    visionMergerParameters.value,
  [visionPatchParameters, visionBlockParameters, visionHidden, visionDownsampleParameters, visionMergerParameters],
);
export const parameterLedger = derive(
  () => [
    {
      label: "Embedding and untied head",
      detail: "Separate token embedding and lm_head matrices",
      value: embeddingParameters.value,
      lane: "merged" as const,
    },
    {
      label: "KDA attention",
      detail: "34 linear-attention layers with conv, forget gate and output gate",
      value: kdaParameters.value,
      lane: "high" as const,
    },
    {
      label: "Sparse MLA and indexers",
      detail: "11 NoPE MLA layers plus one full indexer per sparse layer",
      value: sparseMlaParameters.value + indexerParameters.value,
      lane: "high" as const,
    },
    {
      label: "Base routed experts",
      detail: "42 MoE layers store all 288 experts",
      value: routedExpertParameters.value,
      lane: "low" as const,
    },
    {
      label: "Dense, shared and router",
      detail: "Dense prefix, shared experts, router weights and correction buffers",
      value: denseMlpParameters.value + sharedExpertParameters.value + routerStoredTensors.value,
      lane: "low" as const,
    },
    {
      label: "mHC and norms",
      detail: "Two mHC modules per base layer plus layer/final norms",
      value: mhcParameters.value + normParameters.value,
      lane: "skip" as const,
    },
    {
      label: "MTP layer",
      detail: "One next-token sparse/MoE draft layer, no base-layer mHC",
      value: mtpParameters.value,
      lane: "instance" as const,
    },
    {
      label: "Vision tower",
      detail: "Patch embed, 24 ViT blocks, downsample and merger",
      value: visionParameters.value,
      lane: "instance" as const,
    },
  ],
  [
    embeddingParameters,
    kdaParameters,
    sparseMlaParameters,
    indexerParameters,
    routedExpertParameters,
    denseMlpParameters,
    sharedExpertParameters,
    routerStoredTensors,
    mhcParameters,
    normParameters,
    mtpParameters,
    visionParameters,
  ],
);
export const parameterTotal = derive(
  () => parameterLedger.value.reduce((sum, row) => sum + row.value, 0),
  [parameterLedger],
);
export const parameterNote =
  "This reconciles stored tensors in the BF16 checkpoint, including router correction buffers, MTP tensors and the vision tower. It is not an active-per-token count.";

const tt = (path: string, commit: string) => `torchtitan: ${path} @ ${commit}`;

export const trainingSources = {
  paramCounts:
    "torchtitan local: .scratch/glm53-flash-sft-ladder/evidence/t07b/full_size_param_counts.json (not git-tracked)",
  task02: tt(".scratch/glm53-flash-sft-ladder/sdd/task-02-report.md", "968462a4f"),
  task05: tt(".scratch/glm53-flash-sft-ladder/sdd/task-05-report.md", "5eccb1492"),
  task06: tt(".scratch/glm53-flash-sft-ladder/sdd/task-06-report.md", "1e707e0b6"),
  task13: tt(".scratch/glm53-flash-sft-ladder/sdd/task-13-report.md", "3aa4f8048"),
  task13b: tt(".scratch/glm53-flash-sft-ladder/sdd/task-13b-report.md", "2228c3877"),
  task13c: tt(".scratch/glm53-flash-sft-ladder/sdd/task-13c-report.md", "d4da5c208"),
  task17: tt(".scratch/glm53-flash-sft-ladder/sdd/task-17-report.md", "23e39799e"),
  task18: tt(".scratch/glm53-flash-sft-ladder/sdd/task-18-report.md", "4d55df899"),
  taskN01b: tt(".scratch/glm53-flash-sft-ladder/sdd/task-n01b-report.md", "3fca80581"),
  progress: tt(".scratch/glm53-flash-sft-ladder/sdd/progress.md", "d9ec5c537"),
  t08fLogs: "torchtitan local: experiments/glm53_flash_sft_ladder/results/t08f-20261014T213158Z/attempt-01/logs/",
} as const;

export const memoryEvidence = {
  epDegree: 8,
  routedExpertParams: 304405807104,
  routedExpertBf16GiB: 567,
  routedExpertEp8GiB: 70.875,
  otherTextParams: 8921004862,
  rank16LoraTrainableParams: 67190784,
  rank16OptimizerBytesPerParam: 16,
  rank16LoraShardGiB: 0.125152587890625,
  staticFloorEp8GiB: 73.04594492865726,
  staticPlusAllGatherEp8GiB: 86.85580561636016,
  measuredPeakEp8GiB: 81.46,
  expertLoraTrainableParams: 2780000000,
  expertLoraShardGiB: 5.1781535148620605,
  staticFloorEp4GiB: 146.09188985731453,
  b200CapacityGiB: 178.35,
  activationBufferReserveGiB: 6,
} as const;

export const epExpertsPerRank = derive(() => experts.value / memoryEvidence.epDegree, [experts]);
export const activationBudgetEp8GiB = derive(
  () => memoryEvidence.b200CapacityGiB - memoryEvidence.staticFloorEp8GiB - memoryEvidence.activationBufferReserveGiB,
  [bytesPerBf16],
);
export const fullFinetuneTb = derive(
  () => (textBackboneStoredParameters.value * (16 + bytesPerBf16.value)) / 10 ** 12,
  [textBackboneStoredParameters, bytesPerBf16],
);

export const parallelEvidence = [
  {
    label: "FSDP loop",
    scope: "first 8 layers",
    msPerStep: 110000,
    memoryGiB: 67.1171875,
    source: trainingSources.task13,
  },
  {
    label: "FSDP grouped",
    scope: "first 8 layers",
    msPerStep: 29579.53493,
    memoryGiB: 66.9765625,
    source: trainingSources.task13b,
  },
  {
    label: "EP8 grouped",
    scope: "first 8 layers",
    msPerStep: 5652.82139,
    memoryGiB: 17.3046875,
    source: trainingSources.task13b,
  },
  {
    label: "EP8 grouped",
    scope: "45 layers",
    msPerStep: 29013.4,
    memoryGiB: 81.46,
    source: trainingSources.task13c,
  },
  {
    label: "EP8 rung 4 train",
    scope: "45 layers, 94 steps",
    msPerStep: 37000,
    memoryGiB: 81.46,
    source: trainingSources.t08fLogs,
  },
] as const;

export const deepEpEvidence = {
  commit: "01dc3aaac82068020353dce2c302e38153c0bfaa",
  allToAllSharedLoadMs: 7820.945087,
  qwenSmokeOnly: true,
  glmDeepEpStatus: "blocked after buffer fix; no GLM DeepEP loss/grad/ms claim",
  source: trainingSources.task18,
} as const;

export const longContextEstimates = [
  { label: "today", minTokens: 16384, maxTokens: 24576, source: "current implementation estimate" },
  { label: "chunked indexer + loss", minTokens: 32768, maxTokens: 49152, source: "derived implementation estimate" },
  {
    label: "8-way context parallelism",
    minTokens: 262144,
    maxTokens: 262144,
    source: "derived implementation estimate",
  },
  {
    label: "CPU activation offload",
    minTokens: 1048576,
    maxTokens: 1048576,
    source: "derived implementation estimate",
  },
] as const;

export const rung4Evidence = {
  run: "T08f",
  steps: 94,
  seed42InitialLoss: 1.67393,
  seed42FinalLoss: 0.005254015326499939,
  seed43FinalLoss: 0.007782145403325558,
  peakGiB: 81.46,
  secondsPerStepApprox: 37,
  resumeSteps: "63-94",
  resumeMaxAbsDelta: 0,
  exportTensorCount: 906,
  evalStatus: "Countdown eval pending; no pass@1 claim",
  source: trainingSources.t08fLogs,
} as const;

export const sftLadder = [
  {
    rung: "0",
    model: "Qwen3 0.6B / 1.7B",
    status: "dev pass@1 0.6B 0.004 -> 0.254/0.326; 1.7B 0.024 -> 0.370/0.228",
    source: trainingSources.task02,
  },
  {
    rung: "1",
    model: "Qwen3 8B",
    status: "dev pass@1 0.028 -> 0.474/0.488",
    source: trainingSources.task05,
  },
  {
    rung: "2",
    model: "Qwen3 30B-A3B",
    status: "dev pass@1 0.006 -> attention 0.440/0.476; attention+expert 0.600; exact HDFS resume; no router collapse",
    source: trainingSources.task06,
  },
  {
    rung: "4",
    model: "GLM-5.3-Flash EP8",
    status:
      "T08f trained two 94-step seeds; seed42 loss 1.67393 -> 0.00525, seed43 final 0.00778; step-62 HDFS resume exact for steps 63-94, max |delta| 0.0; Countdown eval pending",
    source: trainingSources.t08fLogs,
  },
] as const;

export const demo = {
  tokens: 12,
  batch: 1,
  layerDefault: 3,
  routerSeedMax: 10000,
  sequenceOptions: longContextEstimates.map((row) => row.minTokens),
} as const;

export const facts: Record<string, Fact> = {
  architecture,
  modelType,
  textModelType,
  hidden,
  vocab,
  layers,
  nextn,
  heads,
  kvHeads,
  qRank,
  kvRank,
  nopeDim,
  ropeDim,
  qkDim,
  valueDim,
  mlaUseNope,
  denseIntermediate,
  moeIntermediate,
  denseFirst,
  experts,
  topK,
  sharedExperts,
  scoring,
  routerScale,
  routerGroups,
  routerTopGroups,
  normTopk,
  routerDtype,
  swigluLimit,
  linearHeads,
  linearHeadDim,
  convKernel,
  gateLowerBound,
  kdaLayers,
  sparseLayers,
  layerTypes,
  mlpLayerTypes,
  indexerTypes,
  indexHeads,
  indexHeadDim,
  indexTopK,
  indexKpool,
  indexKpoolCompress,
  indexTail,
  indexShareForMtp,
  mhc,
  hcMult,
  hcEps,
  hcSinkhorn,
  rmsNormEps,
  maxContext,
  tieWordEmbeddings,
  visionDepth,
  visionHidden,
  visionHeads,
  visionPatch,
  visionTemporalPatch,
  visionChannels,
  visionIntermediate,
  visionOut,
  visionProjectionIntermediate,
  visionMerge,
  routerUsesSigmoid,
  routerUsesCorrectionBias,
  sharedExpertBypassesRouter,
  indexPoolWeighted,
  textNopeForward,
  boundedKdaGate,
  mhcProjectionRule,
  vllmTailBuffer,
  vllmMhcFusion,
  mtpTopkReuse,
  bytesPerBf16,
  qWidth,
  valueWidth,
  kdaWidth,
  sparseAttentionCount,
  kdaCount,
  denseLayerCount,
  moeLayerCount,
  allSparseFullIndexer,
  indexPoolsSelected,
  indexOutputWidth,
  mHcStreamWidth,
  mHcMixWidth,
  mHcSavedPerLayerBytesPerToken,
  mHcSavedBytesPerToken,
  mHcSavedMiBPerToken,
  currentIndexerScoreScalars,
  embeddingParameters,
  kdaPerLayerParameters,
  kdaParameters,
  mlaPerLayerParameters,
  sparseMlaParameters,
  indexerPerLayerParameters,
  indexerParameters,
  expertMlpParameters,
  routedExpertParameters,
  sharedExpertParameters,
  denseMlpParameters,
  routerStoredTensors,
  mhcParameters,
  normParameters,
  textBackboneStoredParameters,
  mtpParameters,
  visionPatchParameters,
  visionAttentionPerBlock,
  visionMlpPerBlock,
  visionBlockParameters,
  visionDownsampleParameters,
  visionMergerParameters,
  visionParameters,
  epExpertsPerRank,
  activationBudgetEp8GiB,
  fullFinetuneTb,
  parameterLedger,
  parameterTotal,
};
