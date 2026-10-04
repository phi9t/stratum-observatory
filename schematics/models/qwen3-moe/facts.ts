import { cfg, code, derive, type Fact } from "../../kit/lib/facts";
export const vocab = cfg("vocab_size", 151936);
export const hidden = cfg("hidden_size", 4096);
export const layers = cfg("num_hidden_layers", 94);
export const heads = cfg("num_attention_heads", 64);
export const kvHeads = cfg("num_key_value_heads", 4);
export const headDim = cfg("head_dim", 128);
export const experts = cfg("num_experts", 128);
export const topK = cfg("num_experts_per_tok", 8);
export const expertWidth = cfg("moe_intermediate_size", 1536);
export const denseWidth = cfg("intermediate_size", 12288);
export const positions = cfg("max_position_embeddings", 40960);
export const theta = cfg("rope_theta", 1000000);
export const normEps = cfg("rms_norm_eps", 1e-6);
export const normTopK = cfg("norm_topk_prob", true);
export const sparseStep = cfg("decoder_sparse_step", 1);
export const denseLayers = cfg("mlp_only_layers", [] as number[]);
export const tied = cfg("tie_word_embeddings", false);
export const attentionBias = cfg("attention_bias", false);
export const sliding = cfg("use_sliding_window", false);
export const ropeScaling = cfg("rope_scaling", null);
export const dropout = cfg("attention_dropout", 0);
export const activation = cfg("hidden_act", "silu");
export const dtype = cfg("torch_dtype", "bfloat16");
export const bos = cfg("bos_token_id", 151643);
export const eos = cfg("eos_token_id", 151645);
export const cacheEnabled = cfg("use_cache", true);
export const auxOutput = cfg("output_router_logits", false);
export const auxCoefficient = cfg("router_aux_loss_coef", 0.001);
export const scoring = code("softmax", {
  source: "reference",
  symbol: "Qwen3MoeSparseMoeBlock.forward",
  contains: "F.softmax(router_logits, dim=1, dtype=torch.float)",
});
export const layerRule = code("not in mlp_only_layers and (layer + 1) % decoder_sparse_step == 0", {
  source: "reference",
  symbol: "Qwen3MoeDecoderLayer",
  contains: "(layer_idx + 1) % config.decoder_sparse_step == 0",
});
export const normPlacement = code("pre-norm", {
  source: "reference",
  symbol: "Qwen3MoeDecoderLayer.forward",
  contains: "hidden_states = self.input_layernorm(hidden_states)",
});
export const sharedExperts = code(0, {
  source: "reference",
  symbol: "Qwen3MoeSparseMoeBlock",
  contains: "self.experts = nn.ModuleList(",
});
export const mlpBias = code(false, {
  source: "reference",
  symbol: "Qwen3MoeMLP",
  contains: "self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)",
});
export const matricesPerExpert = code(3, {
  source: "reference",
  symbol: "Qwen3MoeMLP",
  contains: "self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)",
});
export const qkNormCount = code(2, {
  source: "reference",
  symbol: "Qwen3MoeAttention",
  contains: "self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)",
});
export const cacheParts = code(2, {
  source: "reference",
  symbol: "Qwen3MoeAttention.forward",
  contains: "past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)",
});
export const fused = code("QKVParallelLinear", {
  source: "serving",
  symbol: "Qwen3MoeAttention",
  contains: "self.qkv_proj = QKVParallelLinear(hidden_size,",
});
export const renormalized = code("divide by selected sum", {
  source: "reference",
  symbol: "Qwen3MoeSparseMoeBlock.forward",
  contains: "routing_weights /= routing_weights.sum(dim=-1, keepdim=True)",
});
export const qWidth = derive(() => heads.value * headDim.value, [heads, headDim]);
export const kvWidth = derive(() => kvHeads.value * headDim.value, [kvHeads, headDim]);
export const kvReuse = derive(() => heads.value / kvHeads.value, [heads, kvHeads]);
export const qExpansion = derive(() => qWidth.value / hidden.value, [qWidth, hidden]);
export const inactive = derive(() => experts.value - topK.value, [experts, topK]);
export const expertFraction = derive(() => topK.value / experts.value, [topK, experts]);
export const expertParams = derive(
  () => matricesPerExpert.value * hidden.value * expertWidth.value,
  [matricesPerExpert, hidden, expertWidth, mlpBias],
);
export const expertsPerLayer = derive(() => experts.value * expertParams.value, [experts, expertParams]);
export const attentionPerLayer = derive(
  () => hidden.value * (2 * qWidth.value + 2 * kvWidth.value) + qkNormCount.value * headDim.value,
  [hidden, qWidth, kvWidth, qkNormCount, headDim, attentionBias],
);
export const routerPerLayer = derive(() => hidden.value * experts.value, [hidden, experts]);
export const normsPerLayer = derive(() => 2 * hidden.value, [hidden, normPlacement]);
export const embeddingParams = derive(() => hidden.value * vocab.value, [hidden, vocab]);
export const expertTotal = derive(
  () => layers.value * expertsPerLayer.value,
  [layers, expertsPerLayer, layerRule, sparseStep, denseLayers],
);
export const attentionTotal = derive(() => layers.value * attentionPerLayer.value, [layers, attentionPerLayer]);
export const routerTotal = derive(() => layers.value * routerPerLayer.value, [layers, routerPerLayer]);
export const normTotal = derive(
  () => layers.value * normsPerLayer.value + hidden.value,
  [layers, normsPerLayer, hidden],
);
export const parameterTotal = derive(
  () => 2 * embeddingParams.value + expertTotal.value + attentionTotal.value + routerTotal.value + normTotal.value,
  [embeddingParams, tied, expertTotal, attentionTotal, routerTotal, normTotal],
);
export const activeMatrixParams = derive(
  () =>
    layers.value *
      (topK.value * expertParams.value + attentionPerLayer.value + routerPerLayer.value + normsPerLayer.value) +
    embeddingParams.value +
    hidden.value,
  [layers, topK, expertParams, attentionPerLayer, routerPerLayer, normsPerLayer, embeddingParams, hidden],
);
export const cachePerTokenLayer = derive(() => cacheParts.value * kvWidth.value, [cacheParts, kvWidth]);
export const cachePerToken = derive(() => layers.value * cachePerTokenLayer.value, [layers, cachePerTokenLayer]);
// BF16 has a sign bit, eight exponent bits and seven fraction bits: two bytes.
// Payload assumption, not a claim about vLLM allocations or measured runtime.
export const BF16_BYTES = 2;
export const GiB = 2 ** 30;
export const PERCENT = 100;
export const DECIMALS = 3;
export const DISPLAY_TOKENS = 24;
export const DEMO_CONTEXT = 2048;
export const MAX_BATCH = 16;
export const MAX_SEED = 9999;
export const DEFAULT_SEED = 17;
export const GAIN = 4;
export const ROPE_PAIRS = [0, 1, Math.floor(headDim.value / 4), headDim.value / 2 - 1];
export const TOY = {
  vocab: 32,
  hidden: 12,
  layers: 2,
  heads: 4,
  kvHeads: 2,
  headDim: 4,
  experts: 4,
  topK: 2,
  expertWidth: 8,
  normEps: normEps.value,
  theta: theta.value,
  tokens: 6,
};
export function kindOf(i: number): string {
  return !denseLayers.value.includes(i) && experts.value > 0 && (i + 1) % sparseStep.value === 0 ? "moe" : "dense";
}
export function cacheScalars(batch: number, context: number): number {
  return batch * context * cachePerToken.value;
}
export const facts: Record<string, Fact> = {
  vocab,
  hidden,
  layers,
  heads,
  kvHeads,
  headDim,
  experts,
  topK,
  expertWidth,
  denseWidth,
  positions,
  theta,
  normEps,
  normTopK,
  sparseStep,
  denseLayers,
  tied,
  attentionBias,
  sliding,
  ropeScaling,
  dropout,
  activation,
  dtype,
  bos,
  eos,
  cacheEnabled,
  auxOutput,
  auxCoefficient,
  scoring,
  layerRule,
  normPlacement,
  sharedExperts,
  mlpBias,
  matricesPerExpert,
  qkNormCount,
  cacheParts,
  fused,
  renormalized,
  qWidth,
  kvWidth,
  kvReuse,
  qExpansion,
  inactive,
  expertFraction,
  expertParams,
  expertsPerLayer,
  attentionPerLayer,
  routerPerLayer,
  normsPerLayer,
  embeddingParams,
  expertTotal,
  attentionTotal,
  routerTotal,
  normTotal,
  parameterTotal,
  activeMatrixParams,
  cachePerTokenLayer,
  cachePerToken,
};
