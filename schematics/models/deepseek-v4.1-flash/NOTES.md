# Source reading notes

Checkpoint: deepseek-ai/DeepSeek-V4.1-Flash at 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
The HF config nests text/vision fields; publisher inference/config.json maps them to ModelArgs names. Main layers use zero-based IDs. No Pro facts are used.

## Publisher reference: inference/model.py

- ModelArgs: runtime cache limits are separate from architecture. Defaults are a small test model; released dimensions come from config, not defaults. get_moe_config changes counts only after the backbone, for dSpark.
- ParallelEmbedding: vocab-sharded lookup, mask non-local IDs, reduce partial rows. Main embedding and output head are untied.
- Linear / ColumnParallelLinear / RowParallelLinear: bias-free projections; column shards outputs, row shards reductions and reduces in fp32. FP8 has block scales; FP4 logically [out,in] is packed [out,in/2], with [out,in/32] scales. These storage tensors are not the logical parameter ledger.
- RMSNorm: fp32 mean square and weight product, cast at end.
- ParallelEngramEmbedding: fp8 table plus per-row scales; lookup, reshape into scale blocks, multiply scales, flatten, mask and reduce. Engram: flatten hash rows, wkv, split keys for each residual copy and a shared value. Per-copy normalized dot with q_weight*k_weight, signed square root then sigmoid, optional image mask, gated add. Runs before Block, not inside its forward.
- Compressor: ratio one is projection+norm; ratio two projects KV and gate separately, groups consecutive tokens, softmax across tokens independently per channel, weighted sum, RMSNorm. Incomplete groups retain KV/score state and are unavailable until completion.
- Indexer: only KV sources own key projection/norm/cache. Keys consume the unrotated compressed latent before Attention modifies it. Query comes from normalized query latent; project to index heads, rotate tail, fake FP4 quantize. Key likewise. ReLU dot products times signed projected head weights, sum heads, causal completed-group mask. Candidate source selects blocks by max score and pins newest reachable block; later indexers mask to those candidates. Top-k positions are sorted into positional order. Other layers reuse indices.
- Attention: wq_a -> q_norm -> wq_b -> heads -> tail RoPE. Window: wkv -> kv_norm -> tail RoPE -> fake FP8 -> ring write. Compressed source: compressor -> indexer on pre-RoPE latent -> tail RoPE -> fake FP4 -> compressed cache. Non-source layers read the last source's state. Concatenate window and selected compressed indices. sparse_attn uses shared KV as both K and V, scaled scores and a learned per-head sink in denominator only. Inverse tail RoPE on output -> grouped wo_a einsum -> flatten -> wo_b. Pure SWA uses unscaled base RoPE; compressed layers use compressed base and YaRN for all paths.
- Gate: fp32 projection / gate_temp (default one), sqrt(softplus). Text or image correction bias selects top-k only. Gather original scores, normalize with epsilon, scale. No group filter in this reference.
- Expert: w1 gate and w3 up; upper clamp gate, symmetric clamp up, SiLU*up, optional routing weight BEFORE w2. MoE: flatten, gate, dispatch selected routed experts, accumulate and reduce, add separate always-on shared expert, restore shape.
- Block: hc_mixes flattens all copies, normalized projection, split pre/post/matrix; Sinkhorn matrix. Attention consumes incoming pre, produces its own pre for FFN. FFN consumes attention pre and produces next block's pre. Each hc_post expands sublayer output and mixes residual through matrix. No ordinary x+f(x) substitution.
- ParallelHead: fp32 projection, last position by default, optional full logits, vocabulary gather.
- DSparkAttention: pure window, seeds main KV on prefill; during draft reads main ring plus all draft slots (not ordinary causal draft mask). Same query/output low rank path. DSparkMarkovHead: small embedding+head. DSparkConfidenceHead: concatenate final hidden and Markov embedding -> scalar.
- DSparkBlock: first stage projects concatenated target hidden means; last stage has norm, Markov and confidence heads. Shared main embedding/head. Noise-filled draft input; sequential Markov bias and sampling. Prefill only seeds cache. Speculative acceptance loop is not implemented by publisher generate.py.
- SharedAttentionRuntime: distinct slots for compressed KV, index keys, indices and candidates; each source overwrites before consumers.
- Transformer: ngram hashes -> token embedding -> optional image-span replacement -> repeat residual copies -> initial one-hot pre -> Engram before selected blocks -> collect target layer means -> blocks -> collapse with final pre -> norm -> head -> sampling. Sampling is outside page's logits boundary.

## Additional publisher files

engram.py: normalize tokenizer strings, compressed ID map; layer-seeded odd integer multipliers, disjoint prime buckets; rolling XOR products -> modulo+offset. Image tokens terminate lookback; history persists across decode. Hash arithmetic needs int64, not JS number.
vision.py: PatchEmbed biased RGB projection; per-image bidirectional attention with biased qkv/out and 2D RoPE; pre-RMSNorm residual blocks with bias-free SwiGLU; final norm; Aligner spatial pad and non-overlapping unfold, biased linear/GELU/linear. Image delimiters have learned language-width vectors.
kernel.py: sinkhorn begins row softmax+epsilon, column division then repeated row/column division. Sparse attention adds exp(sink) only to denominator. FP4 and FP8 rounding affect numerical values.

## Independent serving

SGLang deepseek_v4.py explicitly distinguishes model_type deepseek_v41; low ratios construct DeepseekV41Compressor/Indexer only at config source IDs. MqaAttentionBase can fuse wq_a+wkv. MQALayer._compute_kv_to_cache fuses RMSNorm, RoPE and paged cache writes, avoiding the intermediate. Output inverse RoPE may fuse into grouped wo_a. Decoder disables same-sublayer post/pre fusion when hc_pre_from_prev_sublayer is enabled. Platform paths use different packing/rounding; these are not a numerical parity proof. serving-sparse preserves completed-group compression and indexer arithmetic. Full accelerator/backend portability is outside this page's verification.

## Training support boundary

Pinned torchtitan DeepSeekV4TransformerBlock uses same-sublayer pre coefficients, unlike this V4.1 reference's delayed pre. DeepSeekV4Model has a learned HC head, different compression flavors and auxiliary MTP training. It is a family comparison, NOT an executable training implementation for this exact checkpoint. No claim of V4.1 training coverage.

## Implementation and scope decisions

- The runnable model keeps all backbone depth and ownership boundaries, even at teaching widths. Toy dimensions are explicitly derived teaching sizes in facts.ts; they are never presented as checkpoint dimensions. Synthetic lookup IDs are supplied at the tokenizer boundary, because reimplementing the publisher's rolling int64 tokenizer hashes with JavaScript numbers would silently lose precision.
- The executable is text-only prefill to full logits, ideal floating arithmetic. It does not round FP8/FP4 values, implement paged/ring decode state, run distributed collectives, or implement vision/dSpark/sampling. The page draws those auxiliary paths structurally and labels their numerical status.
- Main compressed KV uses FP4 groups of 16 with E4M3 scales; indexer queries/keys use groups of 32 with E8M0 scales. Window vectors use fake FP8 after tail rotation. These distinct code facts are shown; no packed-byte cache budget is inferred from them.
- Candidate admission is a small controlled counterexample with descending scores: the latest reachable block must survive even when its scores lose. The same candidateMask implementation is used inside the executable indexer; the demo explicitly reduces the budget and block width.
- Random weight seed 4 gives visibly different token dispatch at the default layer, without changing the checkpoint rule. The seed controls still permit other, sometimes uniform, legitimate random examples.
- First routing implementation was preceded by a failing behavioral test that attempted the missing route helper. The tests reject weighting with correction bias, ordinary top-k that drops the newest candidate block, and replacing Sinkhorn with identity. Later tests check completed-group/window causality and sink normalization in the clean program; none of these is labelled upstream parity.

## Parameter reconciliation

Independent logical text ledger: 748494684784. Shape-mode parameter requests: exactly 748494684784. Routed experts contribute 543581798400; Engram contributes 196928504320. The remaining text terms are token embedding/head, attention, shared experts/routers, HC coefficients and RMSNorm gains.

The hub's logical total is 763205315794. Draft-only logical parameters are 14225362530: draft blocks with their smaller expert bank and text/image router biases; first-stage target-feature projection and norm; final-stage norm, Markov embedding/head, confidence projection. Main token embedding and vocabulary head are shared and counted once. Vision/aligner/delimiter parameters are 485268480: patch embedding, vision blocks and final norm, spatial aligner biased projections, and language-width delimiter vectors. Thus:

748494684784 + 14225362530 + 485268480 = 763205315794.

This reconciles logical scalars, not quantized packed tensor elements, scale tensors, file bytes, active compute or peak memory.

## Visual review decisions

Compared the sections to the skill's circuit, block, expert, tensor-scale and widget examples. Corrected the cramped eight-expert router by using a model-local wider bar chart; did not alter the kit. Corrected false serial arrows between compressor value/gate projections and draft feature/embedding inputs. Added saved residual/coefficient side paths rather than omitting residual edges. Applied orange to FFN/router graphs, violet to state, teal to residual mixing. Rotary dials now execute the layer's actual base/YaRN frequency formula. Sparse attention heatmaps display the reached raw/compressed positions and the denominator sink. Mobile preserves diagram type size with the kit's horizontal scrolling instead of shrinking labels; this navigation is described on the page.
