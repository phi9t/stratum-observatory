// Every number the page shows. Section files take numbers only from here.
import { cfg, code, derive, type Fact } from "../../kit/lib/facts";

// ---- config.json ----------------------------------------------------------
export const vocab = cfg("vocab_size", 50257);
export const positions = cfg("n_positions", 1024);
export const hidden = cfg("n_embd", 768);
export const layers = cfg("n_layer", 12);
export const heads = cfg("n_head", 12);
export const activation = cfg("activation_function", "gelu_new");
export const normEps = cfg("layer_norm_epsilon", 1e-5);
export const eosToken = cfg("eos_token_id", 50256);

// ---- read from code -------------------------------------------------------
// config.json has no n_inner, so the block falls back to four times the width.
export const mlpRatio = code(4, {
  source: "reference",
  symbol: "GPT2Block",
  contains: "config.n_inner if config.n_inner is not None else 4 * hidden_size",
});
// One projection produces q, k and v together, then splits three ways.
export const qkvParts = code(3, {
  source: "reference",
  symbol: "GPT2Attention",
  contains: "Conv1D(3 * self.embed_dim, self.embed_dim)",
});
export const preNorm = code("pre-norm", {
  source: "reference",
  symbol: "GPT2Block.forward",
  contains: "hidden_states = self.ln_1(hidden_states)",
});
export const scaling = code("head_dim ** -0.5", {
  source: "reference",
  symbol: "GPT2Attention",
  contains: "self.scaling = self.head_dim**-0.5",
});
export const tiedHead = code("lm_head.weight is transformer.wte.weight", {
  source: "reference",
  symbol: "GPT2LMHeadModel",
  contains: '_tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}',
});
export const headBias = code(false, {
  source: "reference",
  symbol: "GPT2LMHeadModel",
  contains: "nn.Linear(config.n_embd, config.vocab_size, bias=False)",
});
export const conv1dWeight = code("weight [in, out], y = x @ W + b", {
  source: "utils",
  symbol: "Conv1D",
  contains: "self.weight = nn.Parameter(torch.empty(nx, nf))",
});
export const geluFormula = code("0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))", {
  source: "activations",
  symbol: "NewGELUActivation",
  contains: "0.044715 * torch.pow(input, 3.0)",
});
// The checkpoint stores a causal-mask buffer per layer that current code no longer loads.
export const ignoredMaskBuffer = code("attn.bias", {
  source: "reference",
  symbol: "GPT2PreTrainedModel",
  contains: '_keys_to_ignore_on_load_unexpected = ["attn.bias", "crossattention.bias"]',
});
export const servingFusedQkv = code("QKVParallelLinear + chunk(3)", {
  source: "serving",
  symbol: "GPT2Attention.forward",
  contains: "q, k, v = qkv.chunk(chunks=3, dim=-1)",
});
export const servingNoLayerScale = code("assert not config.scale_attn_by_inverse_layer_idx", {
  source: "serving",
  symbol: "GPT2Model",
  contains: "assert not config.scale_attn_by_inverse_layer_idx",
});

// ---- derived --------------------------------------------------------------
export const headDim = derive(() => hidden.value / heads.value, [hidden, heads]);
export const qkvWidth = derive(() => qkvParts.value * hidden.value, [qkvParts, hidden]);
export const mlpWidth = derive(() => mlpRatio.value * hidden.value, [mlpRatio, hidden]);

const linear = (inW: number, outW: number): number => inW * outW + outW;
const norm = (width: number): number => width + width;

export const tokenEmbedParams = derive(() => vocab.value * hidden.value, [vocab, hidden]);
export const positionEmbedParams = derive(() => positions.value * hidden.value, [positions, hidden]);
export const attentionParams = derive(
  () => linear(hidden.value, qkvWidth.value) + linear(hidden.value, hidden.value),
  [hidden, qkvWidth],
);
export const mlpParams = derive(
  () => linear(hidden.value, mlpWidth.value) + linear(mlpWidth.value, hidden.value),
  [hidden, mlpWidth],
);
export const blockNormParams = derive(() => norm(hidden.value) + norm(hidden.value), [hidden]);
export const blockParams = derive(
  () => attentionParams.value + mlpParams.value + blockNormParams.value,
  [attentionParams, mlpParams, blockNormParams],
);
export const finalNormParams = derive(() => norm(hidden.value), [hidden]);

/** Parameters of GPT2LMHeadModel; the head adds none because it is tied to wte. */
export const parameterTotal = derive(
  () => tokenEmbedParams.value + positionEmbedParams.value + layers.value * blockParams.value + finalNormParams.value,
  [tokenEmbedParams, positionEmbedParams, layers, blockParams, finalNormParams, tiedHead],
);

export const maskBufferScalars = derive(
  () => layers.value * positions.value * positions.value,
  [layers, positions, ignoredMaskBuffer],
);
export const checkpointScalars = derive(
  () => parameterTotal.value + maskBufferScalars.value,
  [parameterTotal, maskBufferScalars],
);

export const parameterNote =
  `The hub total is ${String(checkpointScalars.value)}: the ${String(parameterTotal.value)} parameters counted here ` +
  `plus ${String(maskBufferScalars.value)} scalars of causal-mask buffers (one ${String(positions.value)} x ${String(positions.value)} ` +
  `"attn.bias" per layer, ${String(layers.value)} layers) that the checkpoint file stores and current transformers ignores on load.`;

/** Key/value cache of one decode stream: per layer, K and V of [heads, S, head_dim]. */
export function cacheScalars(batch: number, context: number): number {
  return batch * layers.value * 2 * heads.value * context * headDim.value;
}
// float32 storage, as the hub's safetensors index reports (parameters: { F32: … }).
export const BYTES_PER_SCALAR = Float32Array.BYTES_PER_ELEMENT;

export const facts: Record<string, Fact> = {
  vocab,
  positions,
  hidden,
  layers,
  heads,
  activation,
  normEps,
  eosToken,
  mlpRatio,
  qkvParts,
  preNorm,
  scaling,
  tiedHead,
  headBias,
  conv1dWeight,
  geluFormula,
  ignoredMaskBuffer,
  servingFusedQkv,
  servingNoLayerScale,
  headDim,
  qkvWidth,
  mlpWidth,
  tokenEmbedParams,
  positionEmbedParams,
  attentionParams,
  mlpParams,
  blockNormParams,
  blockParams,
  finalNormParams,
  parameterTotal,
  maskBufferScalars,
  checkpointScalars,
};
