# Replay dependencies and shared-cache queries

The Decoder and CSA2 teaching additions follow the approved schematic → maths → code
format. [zartbot’s architecture walkthrough](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html)
is credited as secondary teaching inspiration. The diagrams, explanations and Python
are newly authored. No article artwork or prose is bundled; no explicit reuse license
was visible on the article. That observation is not a complete licensing audit.

## Primary-source boundaries

Publisher code uses Flash revision `dba1be0a40aa45a94ad051997016db3960a90277`:
[model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
and [kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py).

- `Attention._window_kv` projects each layer’s local history; `_compress_kv` resolves
  shared global-cache ownership. One does not reconstruct the other.
- `Attention.forward` and `sparse_attn_kernel` combine local and selected global KV
  under one normalization, including the learned sink. The sink contributes to the
  denominator with zero value.
- `Indexer.forward` scores positions before candidate masking in the pinned reference.
  `select_candidate_blocks` reduces to block maxima and ensures the newest eligible
  block is retained. The candidate-producing selector takes its own final top-k from
  the visible range; subsequent selectors apply the candidate mask. A bounded candidate
  set does not establish constant computation in this implementation.

The [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)
describes bounded replay as an approximation. The pinned
[SGLang model](https://github.com/sgl-project/sglang/blob/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/models/deepseek_v4.py)
implements feature-gated replay in `DeepseekV4Model._forward_layers_hc_pre_from_prev`,
with an `is_extend_without_speculative()` guard. It is not the speculative verifier.
PDF source acceptance remains blocked by the existing UTF-8 evidence pipeline.

## What the examples establish

The replay explanation isolates local causal dependencies. For a window containing the
current token and its preceding W−1 tokens, D layers can depend on inputs as far
back as D(W−1) positions. This is an inclusive-index dependency calculation, not a
claim that the full V4.1 network has only a local receptive field. Truncating the
replayed prefix removes dependency paths; the global cache alone does not supply
missing layer-local activations.

The query example fixes an illustrative KV dictionary and changes only the query.
Its CPU checks compare the actual local `sparse_attention` function with independent
softmax arithmetic, including the sink. This demonstrates query-dependent weighting
without mutating the cache. Real decoder layers also compute fresh local KV, residual
updates and expert outputs, so the example does not describe the whole decoder as a
query-only network. CPU agreement is not checkpoint parity or GPU certification.

## Computed reader traces

The replay dependency diagram and its accompanying tensor table now consume the
same saved CPU trace. Each depth row shows the actual residual result; omitted local
positions are null rather than zero. Full history and truncated replay call the same
sparse-attention core with identical global KV and sink. A scalar oracle independently
checks every attention and residual intermediate and the final target difference.

The selection example computes candidate-source scores, causal masks, per-block
maxima, newest-block pinning and candidate masks through the same helper used by
selection. It shows the source layer's direct top-k separately from the later query's
masked top-k. The query example shows scaled logits, joint weights (including sink),
weighted contributions and outputs. All three use deliberately small input vectors,
not checkpoint activations, and do not vary with page hardware/scenario controls.
