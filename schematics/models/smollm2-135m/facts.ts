import { cfg, code, derive, type Fact } from "../../kit/lib/facts";
export const vocab = cfg("vocab_size", 49152);
export const hidden = cfg("hidden_size", 576);
export const layers = cfg("num_hidden_layers", 30);
export const heads = cfg("num_attention_heads", 9);
export const kvHeads = cfg("num_key_value_heads", 3);
export const headDim = derive(() => hidden.value / heads.value, [hidden, heads]);
export const mlpWidth = cfg("intermediate_size", 1536);
export const positions = cfg("max_position_embeddings", 8192);
export const normEps = cfg("rms_norm_eps", 1e-5);
export const theta = cfg("rope_theta", 100000);
export const activation = cfg("hidden_act", "silu");
export const tied = cfg("tie_word_embeddings", true);
export const attentionBias = cfg("attention_bias", false);
export const ropeScaling = cfg("rope_scaling", null);
export const dropout = cfg("attention_dropout", 0);
export const dtype = cfg("torch_dtype", "bfloat16");
export const mlpBias = code(false, {
  source: "configuration",
  symbol: "LlamaConfig.__init__",
  contains: "mlp_bias=False",
});
export const cacheOrder = code("rotated K, unrotated V, before repeat", {
  source: "reference",
  symbol: "LlamaAttention.forward",
  contains: "past_key_value.update(key_states, value_states",
});
export const mlpProjections = code(3, {
  source: "reference",
  symbol: "LlamaMLP",
  contains: "self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)",
});
export const kvParts = code(2, {
  source: "reference",
  symbol: "LlamaAttention.forward",
  contains: "key_states, value_states = past_key_value.update",
});
export const fused = code("fused QKV", {
  source: "serving",
  symbol: "LlamaAttention.forward",
  contains: "qkv, _ = self.qkv_proj(hidden_states)",
});
export const rotaryPairs = derive(() => headDim.value / 2, [headDim]);
export const qWidth = derive(() => heads.value * headDim.value, [heads, headDim]);
export const kvWidth = derive(() => kvHeads.value * headDim.value, [kvHeads, headDim]);
export const groups = derive(() => heads.value / kvHeads.value, [heads, kvHeads]);
export const tokenEmbedParams = derive(() => vocab.value * hidden.value, [vocab, hidden]);
export const attentionParams = derive(
  () => hidden.value * (qWidth.value + 2 * kvWidth.value + qWidth.value),
  [hidden, qWidth, kvWidth, headDim],
);
export const mlpParams = derive(
  () => mlpProjections.value * hidden.value * mlpWidth.value,
  [mlpProjections, hidden, mlpWidth],
);
export const blockNormParams = derive(() => 2 * hidden.value, [hidden]);
export const finalNormParams = derive(() => hidden.value, [hidden]);
export const blockParams = derive(
  () => attentionParams.value + mlpParams.value + blockNormParams.value,
  [attentionParams, mlpParams, blockNormParams],
);
export const parameterTotal = derive(
  () => tokenEmbedParams.value + layers.value * blockParams.value + finalNormParams.value,
  [tokenEmbedParams, layers, blockParams, finalNormParams, tied],
);
export function cacheScalars(batch: number, context: number) {
  return batch * context * layers.value * kvParts.value * kvWidth.value;
}
export const DISPLAY = {
  tokens: 24,
  startQuery: 9,
  startRope: 12,
  batchMax: 16,
  context: 256,
  bytes: 2,
  seedMax: 999,
  runGain: 4,
  idStride: 7,
  parityTolerance: 1e-4,
  pairs: [0, 1, 16, 31],
  digits: 2,
};
export const TOY = {
  vocab: 40,
  positions: 16,
  hidden: 24,
  layers: 2,
  heads: 6,
  kvHeads: 2,
  headDim: 4,
  inner: 48,
  tokens: 8,
};
export const facts: Record<string, Fact> = {
  vocab,
  hidden,
  layers,
  heads,
  kvHeads,
  headDim,
  mlpWidth,
  positions,
  normEps,
  theta,
  activation,
  tied,
  attentionBias,
  ropeScaling,
  dropout,
  dtype,
  mlpBias,
  cacheOrder,
  mlpProjections,
  kvParts,
  fused,
  rotaryPairs,
  qWidth,
  kvWidth,
  groups,
  tokenEmbedParams,
  attentionParams,
  mlpParams,
  blockNormParams,
  finalNormParams,
  blockParams,
  parameterTotal,
};
