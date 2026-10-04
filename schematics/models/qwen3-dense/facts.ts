import { cfg, code, derive, type Fact } from "../../kit/lib/facts";
export const vocab = cfg("vocab_size", 151936);
export const hidden = cfg("hidden_size", 4096);
export const layers = cfg("num_hidden_layers", 36);
export const heads = cfg("num_attention_heads", 32);
export const kvHeads = cfg("num_key_value_heads", 8);
export const headDim = cfg("head_dim", 128);
export const mlpWidth = cfg("intermediate_size", 12288);
export const positions = cfg("max_position_embeddings", 40960);
export const normEps = cfg("rms_norm_eps", 1e-6);
export const theta = cfg("rope_theta", 1000000);
export const activation = cfg("hidden_act", "silu");
export const tied = cfg("tie_word_embeddings", false);
export const attentionBias = cfg("attention_bias", false);
export const sliding = cfg("use_sliding_window", false);
export const window = cfg("sliding_window", null);
export const ropeScaling = cfg("rope_scaling", null);
export const dropout = cfg("attention_dropout", 0);
export const dtype = cfg("torch_dtype", "bfloat16");
export const qkNorm = code("headwise RMSNorm", {
  source: "reference",
  symbol: "Qwen3Attention",
  contains: "self.q_norm = Qwen3RMSNorm(self.head_dim",
});
export const cacheOrder = code("rotated K, unrotated V, before repeat", {
  source: "reference",
  symbol: "Qwen3Attention.forward",
  contains: "past_key_value.update(key_states, value_states",
});
export const mlpProjections = code(3, {
  source: "reference",
  symbol: "Qwen3MLP",
  contains: "self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)",
});
export const kvParts = code(2, {
  source: "reference",
  symbol: "Qwen3Attention.forward",
  contains: "key_states, value_states = past_key_value.update",
});
export const fused = code("fused QKV", {
  source: "serving",
  symbol: "Qwen3Attention.forward",
  contains: "qkv, _ = self.qkv_proj(hidden_states)",
});
export const rotaryPairs = derive(() => headDim.value / 2, [headDim]);
export const qWidth = derive(() => heads.value * headDim.value, [heads, headDim]);
export const kvWidth = derive(() => kvHeads.value * headDim.value, [kvHeads, headDim]);
export const groups = derive(() => heads.value / kvHeads.value, [heads, kvHeads]);
export const tokenEmbedParams = derive(() => vocab.value * hidden.value, [vocab, hidden]);
export const attentionParams = derive(
  () => hidden.value * (qWidth.value + 2 * kvWidth.value + qWidth.value) + 2 * headDim.value,
  [hidden, qWidth, kvWidth, headDim, qkNorm],
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
  () => 2 * tokenEmbedParams.value + layers.value * blockParams.value + finalNormParams.value,
  [tokenEmbedParams, layers, blockParams, finalNormParams, tied],
);
export function cacheScalars(batch: number, context: number) {
  return batch * context * layers.value * kvParts.value * kvWidth.value;
}
export const DISPLAY = {
  tokens: 24,
  batchMax: 16,
  context: 256,
  bytes: 2,
  seedMax: 999,
  runGain: 4,
  idStride: 7,
  parityTolerance: 1e-4,
  pairs: [0, 1, 16, 63],
  digits: 2,
};
export const TOY = {
  vocab: 40,
  positions: 16,
  hidden: 24,
  layers: 2,
  heads: 4,
  kvHeads: 2,
  headDim: 6,
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
  sliding,
  window,
  ropeScaling,
  dropout,
  dtype,
  qkNorm,
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
