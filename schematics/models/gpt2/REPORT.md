# GPT-2 schematic — report

Checkpoint `openai-community/gpt2`, pinned in `sources.json`. Reference:
transformers `modeling_gpt2.py`. Serving: vllm `gpt2.py`.

## Drawn
- Whole forward pass (`GPT2Model`, `GPT2LMHeadModel`), including the tied head.
- Block (`GPT2Block`): 8 ops. Attention (`GPT2Attention`, `eager_attention_forward`):
  12 ops including the KV cache. MLP (`GPT2MLP`): 5 ops.
- Causal mask grid, to-scale tensor widths, parameter ledger, KV-cache size with batch
  and context controls, serving differences.

## Excluded
- Classification, token-classification, question-answering and double-heads models and
  `GPT2SequenceSummary`: the checkpoint's `architectures` is `GPT2LMHeadModel`.
- Cross-attention branch (`add_cross_attention` unset) and
  `_upcast_and_reordered_attn` (`reorder_and_upcast_attn` unset): mentioned in
  inspector notes, not drawn.
- Tokenisation, sampling, training losses.

## Parameters
Derived 124439808. Hub `safetensors.total` 137022720. Difference 12582912 = 12 layers ×
1024 × 1024 causal-mask buffers stored in the file as `attn.bias` and ignored on load
(`_keys_to_ignore_on_load_unexpected`).

## Not verified
- sglang has no GPT-2 model file; only vllm was compared.
- The float32 cache-size figure assumes the stored dtype; a server may use another.

## Gates
`npm run check -- gpt2`: pins, typecheck, literals, format, tests, facts, coverage, parameters, build,
layout, play all pass.

## Clean reference
`reference/model.ts` reproduces `transformers.GPT2LMHeadModel` at toy width (2 layers,
width 24, 3 heads): logits within 1e-4, layer-0 attention within 1e-5, and it consumes
exactly the 28 tensors upstream exports. In shape mode at the real config it counts
124439808 parameters, equal to `parameterTotal`.
