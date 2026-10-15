# Reading notes — Qwen/Qwen3-235B-A22B

Exact checkpoint/config and code revisions are in sources.json. Reference: transformers
v4.51.3, close to the checkpoint's declared v4.51.0. Read the entire modeling file,
configuration class, default RoPE helper, vLLM file, and Megatron bridge before drawing.
The parent Stratum docs named in the supplied contract do not exist in this subtree;
README explicitly defines this as a separate application with its own gates.

## Reference module tree and forwards

- Qwen3MoeForCausalLM: model then biasless, untied lm_head [151936,4096].
  logits_to_keep optionally slices positions before projection. labels and optional
  router auxiliary loss are training/output branches, outside the clean decoder.
- Qwen3MoeModel: embedding [151936,4096]; constructs DynamicCache when requested;
  derives absolute positions from past length; builds causal/padding mask; computes
  cos/sin once for all layers; runs 94 decoder layers; final RMSNorm.
  Mask helpers handle eager, SDPA and FlashAttention, static and sliding caches.
  Checkpoint disables sliding windows; diagrams use explicit eager arithmetic.
- Qwen3MoeDecoderLayer: save residual; input RMSNorm; attention; residual add;
  save residual; post-attention RMSNorm; sparse MoE; residual add. Exact sparse
  condition: layer not in mlp_only_layers AND num_experts > 0 AND
  (layer+1) % decoder_sparse_step == 0. Empty list, step 1 => every layer MoE.
  Constructor assigns attention twice and creates/replaces a dense MLP; these
  overwritten objects are not live modules or checkpoint parameters.
- Qwen3MoeAttention: q projection 4096→8192; reshape [B,S,64,128]; RMSNorm on
  last/head axis then transpose. k 4096→512; reshape [B,S,4,128]; same head norm.
  v 4096→512; reshape/transpose only. q_norm and k_norm each have a shared
  [128] gain, NOT one gain per head. Apply full-head split-half RoPE to q/k;
  append rotated k and unrotated v to cache per layer BEFORE KV repetition.
  Eager helper repeats each KV head 16 times; q·kᵀ times 128^-0.5; additive
  causal/padding mask; float32 softmax then input-dtype cast; dropout is zero;
  weights·v; transpose/merge to [B,S,8192]; o projection to [B,S,4096].
- Qwen3MoeMLP: biasless gate_proj and up_proj 4096→1536 per expert;
  SiLU(gate) * up; down_proj 1536→4096. Config intermediate_size 12288 belongs
  only to the disabled dense branch, not these experts.
- Qwen3MoeSparseMoeBlock: flatten B*S tokens; biasless gate 4096→128;
  float32 softmax over all experts; select top 8; divide selected probabilities
  by their sum; cast to hidden dtype; allocate output zeros; one-hot dispatch;
  gather each expert's tokens; run expert; multiply by selected weight;
  index_add contributions; restore [B,S,4096]. No groups, bias correction,
  extra scaling, capacity clipping or shared experts in this reference.
- Qwen3MoeRMSNorm: float32 mean squares and reciprocal square root with 1e-6;
  cast normalised tensor back, multiply by learned gain. No centering or bias.
- Qwen3MoeRotaryEmbedding: default because rope_scaling=null. Helper uses
  theta^-arange(0,head_dim,2)/head_dim. Duplicate frequencies then cos/sin;
  rotate_half pairs the first half with the second half, not adjacent channels.
  inv_freq is nonpersistent, not a parameter; shared positions are not a learned table.
- Qwen3MoePreTrainedModel: initialization/backend/cache capabilities only.
- KwargsForCausalLM: typing only.
- Qwen3MoeForSequenceClassification: backbone then score; choose last nonpadding
  token; optional classification/regression loss. Not the checkpoint's head.
- Qwen3MoeForTokenClassification: backbone, optional classifier dropout, score,
  optional task loss. Not the checkpoint's head.
- Qwen3MoeForQuestionAnswering: backbone, qa_outputs, split start/end, optional
  span loss. Not the checkpoint's head.
- load_balancing_loss_func: concatenate router logits, softmax, top-k one-hot;
  combine average assignment and average probability, accounting for padding.
  Optional auxiliary output/training loss; not default inference.

## Serving comparison (vLLM v0.8.5)

Read all classes (MLP, SparseMoeBlock, Attention, DecoderLayer, Model, ForCausalLM)
and loading methods. Fused QKVParallelLinear with asymmetric q/k/v split, separate
q/k native per-head norms, RoPE before Attention. KV heads partition when TP fits;
replicate when TP exceeds KV-head count. Backend Attention owns paged cache; the
model file delegates rather than explicitly materializing repeated KV or scores.
FusedMoE replaces Python one-hot/gather/index_add loop; replicated unquantized
router, renormalize=config.norm_topk_prob; collective reduction with TP.
Gate/up fused and checkpoint loader stacks q/k/v and expert weights. Block carries
(residual, update) separately and fuses addition into the next RMSNorm, including
final norm. Vocabulary parallel embedding/head and pipeline intermediate tensors;
compute_logits separated from forward. No claim of throughput or memory allocation
beyond logical KV payload. Backends and kernels are not locally GPU-validated.

## Training bridge

NVIDIA-NeMo/Megatron-Bridge Qwen3MoEBridge explicitly names this checkpoint, maps
Q/K/V and gate/up weights, sets RMSNorm, gated linear unit, qk_layernorm,
alltoall dispatcher and grouped GEMM; derives pre_softmax as !norm_topk_prob.
This is conversion/provider configuration, not independently executed training math.
No training parity or training recipe performance claim is made.

## Decisions

Full unpadded prefill to logits is the executable clean reference; cache is drawn
and its payload calculated separately. Toy keeps two MoE layers, grouped queries,
explicit head_dim independent of hidden/head count, and top-k above one. Numerical
fixture uses local CPU torch and an isolated cache-local transformers v4.51.3,
checks its modeling/config/RoPE bytes against pinned files, perturbs all parameters,
and downloads no trained weights. No global installation or home-directory edit.

## Validation and review refinements

Numerical parity succeeded for the pinned reference, config and default-RoPE bytes.
Tests compare logits, attention matrices, router logits, exact chosen expert ids and
normalized selected weights and require exact parameter-name coverage. A negative
head-axis test first failed because fractional KV repetition was accepted; the local
operator now rejects nonpositive/noninteger repeat counts. Another negative test
changes a future token and requires earlier logits/routes to stay identical.
Fixture fingerprints are checked against cached pinned source bytes on every test run.

Shape mode cannot determine data-dependent expert loads. Its expert activation traces
use the all-token upper bound; value mode dispatches exact selected token subsets.
Inspectors use symbolic Ne rather than a fabricated zero load. Parameter dimensions,
output shapes and total are exact in shape mode.

Visual round one found clipped heading in the kit's toy-count MoeRouter, and excessive
mobile inspector heights from full parameter names in max-content label columns.
The model-local toy chart uses wider named expert bars, and labels are shortened with
their owning namespace preserved in the inspector. No shared kit changes.

Operational deviation: dependency files were installed only under the ignored
.sources-cache/qwen3-moe/python, but pip was invoked without --no-cache-dir and used
~/.cache/pip for downloads. That unintended home-cache write
violates the run's no-home-write constraint. No global installation or home source/
configuration edit was made; no files outside the authorized directory were removed
in an attempt to conceal or undo the cache side effect. Future reproduction commands
must use PIP_NO_CACHE_DIR=1 and PIP_DISABLE_PIP_VERSION_CHECK=1.
