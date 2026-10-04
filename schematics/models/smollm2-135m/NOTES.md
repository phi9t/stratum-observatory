# Source reading notes

Exact checkpoint: HuggingFaceTB/SmolLM2-135M (base), commit in sources.json.
Config identifies LlamaForCausalLM, not custom remote modeling code. The pinned
Transformers v4.51.3 reference is newer than config creation metadata (4.40.1);
it is the implementation used for this page, not the historical training executable.

## Reference modules and forward order

- LlamaRMSNorm: input to float32; final-axis mean square; rsqrt with epsilon;
  cast back; learned gain. No mean subtraction or bias.
- LlamaRotaryEmbedding: rope_scaling=null selects default frequencies. Buffer is
  nonpersistent. Position times inverse frequency, duplicate halves, cos/sin in
  float32, cast back. LlamaModel shares these tables across layers. rotate_half
  pairs i with i+D/2; values bypass rotation.
- LlamaMLP: bias-free gate/up/down; config omits mlp_bias so configuration class
  supplies false. SiLU(gate(x))*up(x), then down. No cross-position operation.
- LlamaAttention: hidden/query-heads fallback gives head width. Separate Q/K/V
  projections, view [B,S,H,D], transpose [B,H,S,D], rotate Q/K. No head RMSNorm.
  Cache.update stores rotated K/raw V per layer before repetition. Eager path
  repeats each KV head consecutively, dots QK transpose, scales D^-0.5, adds mask,
  softmax in float32/cast back, dropout (zero), mixes V, transposes/flattens, o_proj.
  SDPA/Flash/flex are execution backends; page draws eager logical arithmetic.
- LlamaDecoderLayer: save residual; input norm; attention; add; save residual;
  post-attention norm; MLP; add. Same circuit at every index.
- LlamaPreTrainedModel: initialization/capability metadata, excluded from graph.
- LlamaModel: input embedding or supplied vectors; optional DynamicCache; absolute
  cache positions; cache-aware causal/padding mask; shared rotary tables; ordered
  decoder layers; final RMSNorm. Static cache/padding branches exist but executable
  teaching reference uses unpadded prefill.
- KwargsForCausalLM: typing, excluded.
- LlamaForCausalLM: model then bias-free vocabulary projection; post_init ties
  embedding/head through framework because tie_word_embeddings=true. Logit slicing
  and optional label loss are outside the all-position forward-to-logits scope.
- Sequence classification, token classification and question-answering: alternative
  task heads absent from architectures; excluded explicitly in coverage.json.

Shapes: residual [B,S,576], Q [B,9,S,64], K/V [B,3,S,64], MLP [B,S,1536]. Each
three consecutive query heads share one KV head. Cache per token/layer: 2*3*64
scalars. Shared vocabulary [49152,576] counted once, not as an independent head.

## Independent serving implementation

vLLM LlamaMLP packs gate/up, SiluAndMul, row-parallel down. LlamaAttention packs
QKV, splits widths, rotates Q/K, delegates cache and attention to backend Attention,
row-parallel output. KV heads partitioned or replicated under divisibility checks;
not every tensor-parallel size is supported. LlamaDecoderLayer defers residual sums
and fuses with RMSNorm. LlamaModel completes the last sum at final norm; pipeline
stages carry hidden/residual separately. LlamaForCausalLM ties the vocabulary head
and skips separate head loading. Quantization, LoRA, GGUF and Mistral remapping are
excluded branches. Logical cache budgets exclude pages, replication and metadata.

## Training comparison

Publisher README explicitly names Nanotron for SmolLM2. Its later SmolLM3 statements
are unused. Pinned current Nanotron is a comparison, not evidence of the historical
training configuration. MLP packs gate/up and GLUActivation splits/SiLU/multiplies.
CausalSelfAttention packs/splits grouped heads; uses [B,S,H,D], then flattened
unpadded token rows. Training runs Flash rotary and variable-length Flash attention;
rope_interleaved=false uses half-split pairs. CoreAttention has an optional muP
scale, so not every configuration computes this page's scale. Decoder keeps pre-norm
and residual order. Embedding is sequence-first; LlamaModel uses pipeline blocks,
final norm/head and float32 logits. Loss, LossWithZLoss, initialization, tied-name
metadata and FLOP estimates are outside scope. Current noninterleaved inference
branch appears to rotate twice; it is not drawn or used as a parity oracle.

## Numerical evidence

Fixture imports exact cached modeling source with matching Transformers 4.51.3,
constructs a random CPU float32 model, perturbs every unique parameter, saves unique
named parameters (tied head omitted by identity). Toy: two layers, width 24, six Q
heads, two KV heads, head width 4; preserves sharing ratio. Tests compare logits,
attention, parameter names, source hash/revision, causality, rotary arithmetic and
absence of invented head weights. Browser uses seeded random weights with gain 4.
No trained checkpoint weights, GPU runs, quality measurements or throughput claims.
