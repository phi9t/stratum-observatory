# V4.1 readable PyTorch modules

The user requested straight-line PyTorch implementations in the five existing V4.1
chapters. `src/pytorch-*.mjs` owns the executable teaching text. The builder serializes
these records as data, so Python imports cannot be confused with JavaScript module
imports. The UI, Python downloads, CPU checks and scientific consumer inventory read
the same records. The encoder and decoder reuse the same mHC/flat-stack and MoE records.

Each record names upstream symbols, immutable source links and its simplifications.
Exact source-symbol byte anchors are recorded per example in the observatory manifest.
These are derived consumers awaiting independent source acceptance, not new accepted
source snapshots. CPU checks run only our authored adaptations, never upstream code.

Numerical normalization, routing and clamp settings are explicit caller inputs where
needed. Tests supply illustrative dimensions, scalars and weights. Attention examples
isolate already projected/rotated latent contractions, with the sink retained in the
normalization. They do not supply quantization, ring-buffer management or checkpoint
loading. The flat CED stack injects causal attention operators and exposes the boundary
residual for inspection; it does not claim a separate cross-attention path or certify
encoder-only prefill. Engram receives hash-row identities from the upstream hash layout.

The V4.1 dSpark draft attention and output heads are distinct from the DeepSpec
single-sequence Qwen3/Gemma4 verifier. Its probability correction, contiguous acceptance,
confidence threshold and cropped state are shown for a nonterminal round. Stop tokens,
output-budget truncation and production scheduling are outside the excerpt.

Native disclosures support keyboard access. Each download includes attribution,
source URLs and any declared dependencies. Printing opens both code and MIT notices
and restores the previous disclosure states afterward. Expanded code is measured by
the aesthetics gate; readable text uses existing tokens and scales.

## Typed modules and code presentation

The module names now describe their role (`RMSNorm`, `TransformerBlock`,
`CausalEncoderDecoder`, `MixtureOfExperts`, `EngramMemory`) without teaching prefixes.
Parameters and return values are annotated. Standard-library `Annotated` metadata names
tensor axes; adjacent comments explain contractions, broadcasting, gathers, normalization
and cache ownership. The shape metadata documents the contract; explicit guards validate
selected input boundaries. It does not claim to be a symbolic shape checker.

Reduced-precision checks exercise BF16 parameters, CPU autocast, output dtype and finite
backpropagation. FP32-sensitive contractions disable autocast locally. The examples retain
plain PyTorch operations and explicit scalar inputs, without configuration switches for
unused architectures. They remain unfused reference modules, with checkpoint loading,
quantization and production serving integration bounded by each record's scope.

The offline Python display lexer is lossless and escapes HTML. It supplies theme-aware
syntax tokens and nonselectable CSS line numbers. The native Clipboard API copies the
standalone program with attribution; if clipboard permission is unavailable, the UI
points to the Python download. Neither the lexer nor the page executes Python.
