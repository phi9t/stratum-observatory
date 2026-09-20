# DeepSeek V4.1 Flash deep-dive source map

Research scope: encoder, decoder, Engram, CSA2, and dSpark. The user clarified that “despark” means dSpark speculative decoding/MTP. Publisher code spells the classes `DSpark`; this is not a separate architecture inferred from a spelling similarity.

## Evidence status

This map inspects six existing locally cached upstream files. All six files reproduce the SHA-256 recorded in `verification/manifests/sources.json`. That confirms local byte identity, not independent scientific review or present upstream availability. The web reader could not open the pinned publisher files, but a subsequent direct HTTP check returned 200 for both model.py and README.md and reproduced their cached hashes. No source bytes, manifests, trust roots, runtime receipts, or acceptance decisions were changed. Publication remains BLOCKED and runtime experiments remain NOT_RUN.

## Exact inventory

- **dsv41-config** — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/config.json); revision `dba1be0a40aa45a94ad051997016db3960a90277`; SHA-256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`. Local cache verified.
- **dsv41-code** — [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py); revision `dba1be0a40aa45a94ad051997016db3960a90277`; SHA-256 `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`. Local cache verified.
- **dsv41-engram** — [inference/engram.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/inference/engram.py); revision `dba1be0a40aa45a94ad051997016db3960a90277`; SHA-256 `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897`. Local cache verified.
- **dsv41-kernel** — [inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py); revision `dba1be0a40aa45a94ad051997016db3960a90277`; SHA-256 `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`. Local cache verified.
- **dsv41-card** — [README.md](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/README.md); revision `dba1be0a40aa45a94ad051997016db3960a90277`; SHA-256 `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251`. Local cache verified.
- **dsv41-sglang** — [python/sglang/srt/models/deepseek_v4.py](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/models/deepseek_v4.py); revision `ac1a56987c5c552f9f4284f5a98db5d65541fafa`; SHA-256 `f801fc032dc4fae476ace0de365741472ba3ae12ae96cd360224817d643fc216`. Local cache verified.

## Encoder

Card line 47 explicitly identifies the causal encoder/decoder split. Config `/text_config/num_hidden_layers` and `/text_config/compress_ratios` establish the layer inventory, but the ratio boundary alone does not prove the architectural name (see existing T13a ruling in `ANCHOR_RESIDUE.md`). `Transformer.forward`, `Block.forward`, `Attention`, and `Compressor` establish token processing and compressed-cache production.

Show causal token flow, depth versus token time, local windows, and global-cache owners. The first encoder layers have zero compression ratio; remaining encoder entries have ratio two. Derive displayed numbers from model fields. Do not draw a bidirectional encoder or imply every layer owns global KV.

## Decoder

Card line 47 describes global KV projected from final encoder states. `Attention._compress_kv` (740–763) publishes global KV only at configured sources. `/text_config/kv_source_layer_ids` makes layer 20 the final producer; subsequent decoder layers reuse it while retaining local windows.

Show boundary projection → shared global memory → later readers, alongside local state and output generation. SGLang lines 3537–3552 and 3665–3689 implement feature-gated bounded replay after the final producer. This path requires `is_extend_without_speculative()`; do not claim it automatically applies to dSpark verification. The reference backbone loop is not a measured demonstration of the card's efficiency claims.

## Engram

`EngramLayout`, `build_compressed_token_map`, and `NgramHashState.forward` establish normalized token IDs, layer-specific n-gram hashes, and causal lookback. Image/dead tokens stop history. `model.py:Engram.forward` (350–366) fetches rows, projects per-residual-copy keys and a shared value, computes a normalized content gate, and adds gated memory to the residual. `Transformer.forward` excludes image positions.

Show token history → normalized IDs → hash buckets → table reads → projection → content gate → residual update. Config `/text_config/engram_layer_ids`, `engram_num_embeddings`, `engram_max_ngram_size`, `engram_n_heads`, and `engram_head_dim` supply exact dimensions. Engram is learned conditional memory, distinct from request KV or expert routing. Rounded card parameter counts do not anchor exact derived table totals.

## CSA2

Card line 49 names Full, Reindex, and Reuse. Config `/text_config/kv_source_layer_ids` and `index_source_layer_ids`, plus `Attention.__init__`, `_compress_topk_idxs`, and `_compress_kv`, establish owners: Full produces global KV/index keys/selection; Reindex recomputes selection against shared state; Reuse consumes existing selection. Local-window attention remains separate.

`Indexer.forward` (529–588) constructs index keys before main-cache rotation/quantization; `select_candidate_blocks` (591–618) builds reachable candidates and includes the newest block. Decoder candidate capacities come from `candidate_source_layer_id`, `candidate_topk_blocks`, `candidate_block_size`, and `index_topk` under `text_config`.

Show ownership modes and hierarchical candidate selection separately. Main-cache quantization and indexer quantization differ. The reference indexer constructs full score tensors before masking: candidate restriction alone is not evidence that every operation is context-independent. Global bytes/token excludes local windows and runtime allocation; no latency claim follows from it.

## dSpark / MTP

`DSparkBlock.forward_embed` initializes seed-plus-noise drafts from projected captured backbone features. `Transformer.forward` captures target attention inputs, not outputs. `DSparkAttention` uses local history and draft positions, with compression disabled. `DSparkBlock.forward_head` adds sequential Markov bias, samples proposals, and returns confidence scores. Prefill seeds draft history; `forward_spec` connects the draft stages.

Exact code anchors: `get_dspark_topk_idxs` 1021–1030; `DSparkAttention` 1032–1074; Markov/confidence heads 1077–1097; `DSparkBlock` 1100–1156; `forward_spec` 1275–1282. Config `/text_config/num_nextn_predict_layers`, `dspark_block_size`, `dspark_target_layer_ids`, `dspark_markov_rank`, and dSpark-specific expert fields define scope; do not substitute backbone counts.

**Missing:** reference lines 129–130 explicitly say the implementation supplies draft forward, no caller of `forward_spec`, and no acceptance loop. Card line 51 names confidence-scheduled verification without specifying its policy. Scoped SGLang code captures auxiliary features, not the complete coordinator. Target verification/acceptance may be shown only as a labeled conceptual interface. Thresholds, rollback, exact acceptance rules, speedups, and distribution guarantees remain unsupported.

Captured-feature reuse and decoder global-KV reuse are separately supported. Their coexistence does not establish coordinator buffer reuse or rollback. That is the blocker to an end-to-end dSpark execution tutorial, not to a substantive draft-path chapter. All five chapters should preserve parameter/cache scope and derive quantities from source-bound model fields. No wider corpus was searched.


## Published report checked after the user's source-grounding clarification

The pinned [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf) returned HTTP 200 with 1,809,802 PDF bytes and SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`. The raw endpoint returned a Git LFS pointer; the resolved PDF matches its OID. Raw responses and pdftotext extraction are preserved in the ignored research cache. The UTF-8-only evidence verifier does not accept a PDF as a code/text snapshot, so this report observation is not promoted into an accepted source hash or scientific PASS.

Sections 2.2 and 2.3.1 agree with the CED and CSA2 ownership map above. Section 2.4.2 explicitly omits the original Engram short causal convolution. Section 2.4.3 describes confidence-based verification-length scheduling: conditional acceptance estimates form prefix-survival estimates; those combine with profiled engine throughput curves and current load to select per-request verification length. This is report-level design, not an observed STRATUM runtime or a complete published coordinator implementation. The reference confidence-head scalar and the report's intended acceptance probability must be distinguished; calibration is not implemented in that head. Exact acceptance/rejection and rollback rules remain unavailable in the inspected code.
