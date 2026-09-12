# DeepSeek V4.1 serving research

Observed 2026-09-12. Bounded primary-source research; no model weights fetched, no upstream code executed, no runtime certification. This note does not modify source manifests, accepted evidence, or reports.

## Corpus and immutable identities

Searched existing design notes, source manifests and the local source cache first. Existing notes treated framework support as absent. The research cap was twelve article/source files, plus Git ref/tree metadata; all twelve file requests returned HTTP 200. GitHub recursive tree responses were complete (`truncated: false`). Raw responses were preserved unmodified in temporary research files and checksummed; they are candidate observations, not accepted evidence-cache entries.

- SGLang `main`: `21289cfd50a8cf7550c1786ac71733bec9396d4d`, resolved with `git ls-remote`.
- SGLang `dsv4.1`: `ac1a56987c5c552f9f4284f5a98db5d65541fafa`, resolved with `git ls-remote --heads`; used for exact V4.1 implementation evidence.
- Miles `main`: `8760515851ec4f76815171d812c836bf8caa0aca`, resolved with `git ls-remote`; exact plugin files were found on the development branch.
- Miles `zczeng/dsv41-rl-support`: `6c6858a42b61459467814edc1404b1d9bfa38471`, resolved with `git ls-remote --heads`.
- Miles ownership was discovered through the pinned official SGLang README link to `radixark/miles`; it was not guessed from the project name.

| ID | Source and immutable URL | Commit / status | SHA-256 |
|---|---|---|---|
| R1 | [LMSYS implementation-author report](https://www.lmsys.org/blog/2026-09-10-deepseek-v41/) | dated HTML snapshot; HTTP 200 | `4211c83c29bff40053b0fbed749ef225ea63e9d17d81fda183bda27a5264cf33` |
| R2 | [SGLang ownership/README](https://raw.githubusercontent.com/sgl-project/sglang/21289cfd50a8cf7550c1786ac71733bec9396d4d/README.md) | 21289cfd50a8cf7550c1786ac71733bec9396d4d; HTTP 200 | `ee6908559f95fe54c3f6773ed23e163d76c984f4750445830a343eac6edca1d0` |
| R3 | [SGLang main model (not exact V4.1 proof)](https://raw.githubusercontent.com/sgl-project/sglang/21289cfd50a8cf7550c1786ac71733bec9396d4d/python/sglang/srt/models/deepseek_v4.py) | 21289cfd50a8cf7550c1786ac71733bec9396d4d; HTTP 200 | `65b299966c0c24b674c55c562056cc82000779e9b98733ea2b1661b576b9f5e3` |
| R4 | [SGLang main configuration](https://raw.githubusercontent.com/sgl-project/sglang/21289cfd50a8cf7550c1786ac71733bec9396d4d/python/sglang/srt/configs/deepseek_v4.py) | 21289cfd50a8cf7550c1786ac71733bec9396d4d; HTTP 200 | `9adf7030a4f6a39459934d28bdd3e7fb8def1f524bb91ce786212f4d404c326f` |
| R5 | [SGLang main cache](https://raw.githubusercontent.com/sgl-project/sglang/21289cfd50a8cf7550c1786ac71733bec9396d4d/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) | 21289cfd50a8cf7550c1786ac71733bec9396d4d; HTTP 200 | `2a7ec00d54abc69a4115f90d12c04a80a3f4f2cc72b5f3ade98e0bc92f765f70` |
| R6 | [SGLang launch documentation](https://raw.githubusercontent.com/sgl-project/sglang/21289cfd50a8cf7550c1786ac71733bec9396d4d/docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx) | 21289cfd50a8cf7550c1786ac71733bec9396d4d; HTTP 200 | `8c6a63cf4eae472fb5e642bb8dc4b0ddc5939f4fe811a2ff3c81982bce42046e` |
| R7 | [SGLang exact branch model](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/models/deepseek_v4.py) | ac1a56987c5c552f9f4284f5a98db5d65541fafa; HTTP 200 | `f801fc032dc4fae476ace0de365741472ba3ae12ae96cd360224817d643fc216` |
| R8 | [SGLang exact V4.1 configuration](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/configs/deepseek_v41.py) | ac1a56987c5c552f9f4284f5a98db5d65541fafa; HTTP 200 | `23ffdb2c0201b449963f83dca9b9783decceb57f5a1ab5bf49bbae77faddf63e` |
| R9 | [SGLang Engram layouts](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/layers/engram.py) | ac1a56987c5c552f9f4284f5a98db5d65541fafa; HTTP 200 | `557bec6d297697ad7b16ae3456ecffd5e8eee1871662effa6914e76376125ebd` |
| R10 | [SGLang exact branch cache](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) | ac1a56987c5c552f9f4284f5a98db5d65541fafa; HTTP 200 | `e87817052053fef8db323eae94dc1b8300e097ef03002102b84f2021b9176dce` |
| R11 | [Miles V4.1 Megatron plugin](https://raw.githubusercontent.com/radixark/miles/6c6858a42b61459467814edc1404b1d9bfa38471/miles_plugins/models/deepseek_v41/deepseek_v41.py) | 6c6858a42b61459467814edc1404b1d9bfa38471; HTTP 200 | `5e6db411d89102890e045264e3e7ffd53cf2a5cfbe3ecda26b8d26fc78995f7c` |
| R12 | [Miles V4.1 launch script](https://raw.githubusercontent.com/radixark/miles/6c6858a42b61459467814edc1404b1d9bfa38471/scripts/run_deepseek_v41.py) | 6c6858a42b61459467814edc1404b1d9bfa38471; HTTP 200 | `cb3b36c9684c1afc1ccb4f753156987d9d9a53a3165f6d0e6b25594d22c8b2fe` |

## What the article establishes

The [implementation-author report](https://www.lmsys.org/blog/2026-09-10-deepseek-v41/) describes opt-in bounded replay: layers 0–20 process full prompt chunks; layers 21–39 process each request’s final at-most-128 tokens. Encoder replay reconstructs a prefix tail while retaining compressed KV and index keys. The authors explicitly call replay approximate because local attention is truncated at the reconstruction boundary. They report deployment and training evaluations; those are reported results, not STRATUM reproductions. Encoder replay excludes speculative decoding; decoder replay excludes prompt logprobs and full prompt-hidden capture. The article describes FP4 indexer reads, alternative host Engram layouts, and a Miles Megatron plugin using SGLang rollouts. No benchmark number from the article is promoted into a STRATUM performance claim.

## Exact V4.1 identity and release scope

[R8](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/configs/deepseek_v41.py) `normalize_deepseek_v41_config` flattens text/vision configuration, preserves `model_type=deepseek_v41`, and translates `DeepseekV41ForCausalLM` to `DeepseekV4ForCausalLM`. `DeepseekV41Config` deliberately inherits a configuration accepting low compression ratios rather than the native V4 configuration that rejects them. [R7](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/models/deepseek_v4.py) registers `DeepseekV4ForCausalLM` in `EntryClass`, imports V4.1-specific compressor/indexer implementations, and branches on `deepseek_v41`. The shared filename is therefore corroborated by an explicit V4.1 adapter and execution branches; it is not evidence borrowed from an unrelated V4 checkpoint.

[R6](https://raw.githubusercontent.com/sgl-project/sglang/21289cfd50a8cf7550c1786ac71733bec9396d4d/docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx) names DeepSeek-V4.1-Flash and says support has not shipped in a release, using development container tags. Finding development-branch source establishes implementation availability at that revision, not main-branch merge state, a released package, or compatibility of an arbitrary deployment.

## Bounded replay: source-level behavior

[R7](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/models/deepseek_v4.py) supplies these reviewable Python anchors:

- `DeepseekV4Model.__init__`: `late_layer_start` remains absent unless decoder replay is enabled; when enabled it is `max(kv_source_layer_ids)+1` and late compression ratios are validated.
- `DeepseekV4Model._forward_layers_hc_pre_from_prev`: applies tail slicing only for the enabled, non-speculative extend path; at `late_layer_start`, it switches backend metadata and slices hidden states, predecessor coefficients, IDs and positions. It still executes later layers on the selected rows.
- `DeepseekV4Model._check_late_layer_tail_readers`: rejects full prompt-hidden capture and prompt-token logprob requests that require discarded rows.

Applied to STRATUM’s source-layer IDs [2, 8, 14, 20], the split starts at layer 21: twenty-one full-context layers and nineteen tail layers. This is a serving optimization over token participation, not removal of nineteen layers from the model. The exact tail-index builder and encoder-side scheduling implementation were not fetched within the source cap; the article owns the 128-token limit and encoder-mode claims in this note.

## FP4 indexer storage: 170 B/token can be independently reconciled

[R10](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) contains the complete logical storage chain:

| Python anchor | What it establishes |
|---|---|
| `get_dsv4_indexer_bytes_per_token` | FP4 uses `D//2 + D//32`: packed payload plus scale bytes per compressed index token. |
| `DeepSeekV4TokenToKVPool._collect_sources_by_ratio` | Ratios one/two allocate source storage only for declared KV-source layers, not every consumer/index-source layer. |
| `DeepSeekV4TokenToKVPool.source_layer_of` | A consumer resolves to the nearest preceding source with the same ratio. |
| `DeepSeekV4TokenToKVPool._init_compressed_pools` | Index pools have `len(sources)` layers per ratio; low-ratio pools force FP4 and preserve `loc//ratio` addressing. |
| `DeepSeekV4TokenToKVPool._make_indexer_pool` | Forced FP4 ignores deployment toggles and enables reference round-to-nearest-even for low-ratio keys. |
| `DeepSeekV4IndexerPool._create_buffer` | CUDA stores byte-packed pages; the alternate HIP layout splits payload and scales. Both carry the same logical bytes. |
| `DeepSeekV4TokenToKVPool._init_compressed_layer_mapping` | Source-index mapping prevents counting reused storage again for consumers. |

For the existing checkpoint facts, index width 128 means 64 payload bytes plus four scale bytes per compressed token. Three source layers compress by two and one by one:

```text
indexer slope = (128/2 + 128/32) × (3/2 + 1) = 170 B/input token
main-KV slope = 720 B/input token   [existing STRATUM architectural derivation]
combined slope = 720 + 170 = 890 B/input token
```

This now reconciles the reported total through independently identified storage code rather than selecting a precision to fit a residual. The aggregate uses STRATUM’s existing checkpoint dimensions; the new evidence supplies the indexer representation and source ownership. It is a logical slope for completed compression groups, not exact allocator occupancy at every context length.

Allocation remains larger: `_init_compressed_pools` includes reserved full-page capacity before ratio division; `dsv41_index_page_size` chooses 64 or 128 compressed slots; `_create_buffer` rounds and reserves pages. Ratio-two compression retains pending-pair state (`get_compress_state_ring_size`). Odd contexts, speculative rollback buffers, candidate metadata, scale representation inside other caches, page fragmentation and request state require separate allocation accounting. Do not call `170×S` exact runtime memory for arbitrary odd S, or treat 890 B/token as an HBM-fit guarantee.

## Engram: ownership changes communication

[R9](https://raw.githubusercontent.com/sgl-project/sglang/ac1a56987c5c552f9f4284f5a98db5d65541fafa/python/sglang/srt/layers/engram.py) `EngramEmbedding` allocates FP8 table rows with per-block scales. Its default device layout shards rows over TP ranks. `_HostTable` and `EngramEmbedding._init_host_table` provide two host layouts:

- `private`: one anonymous mapping of owned rows per rank. `_lookup` gathers owned rows and retains TP all-reduce.
- `shared`: one memfd with all rows mapped by the TP group. `forward` directly gathers the complete requested rows and returns without that lookup all-reduce.

`_HostTable.choose_layout` selects among supported layouts using transparent-huge-page policy. `_open_shared_fd` requires TP ranks to share access through the owner’s `/proc/<pid>/fd`; it fails explicitly when this cannot be established. The source exposes memory topology and operating-system constraints, not a universal claim that host offload is faster. Neither residency nor network accounting should multiply a shared host table by every TP rank or erase all-reduce for private shards.

## Miles: a real plugin, with separate validation obligations

[R11](https://raw.githubusercontent.com/radixark/miles/6c6858a42b61459467814edc1404b1d9bfa38471/miles_plugins/models/deepseek_v41/deepseek_v41.py) is an exact V4.1 Megatron plugin: `apply_v41_config`, `V41Runtime`, `V41CarryPlan`, `V41HyperConnection`, `V41TransformerLayer` and `DeepSeekV41Attention`. It carries source-owned latent/index state, predecessor mHC data, and Engram inputs. Attention creates compressor/indexer components according to source roles; its forward invokes fake-quantized compressed KV, retrieves shared source state, combines sparse attention indices and requests FP32 gradient all-reduce on duplicated KV. Imports alone do not establish all helper semantics; helper kernels were outside this cap.

[R12](https://raw.githubusercontent.com/radixark/miles/6c6858a42b61459467814edc1404b1d9bfa38471/scripts/run_deepseek_v41.py) launches `--model-name deepseekv41` with SGLang rollout arguments and configurable parallelism, routing replay, optimizer/disk offload and Engram host placement. These are available scenario controls, not proof the article’s run used this exact current branch head. The plugin explicitly rejects virtual pipeline schedules outside its supported form and mHC recompute; preserve those limits instead of claiming unrestricted training support.

## Consequences for STRATUM and remaining gaps

- Replace unconditional “no framework implementation exists” wording with dated, branch-scoped implementation availability. Candidate code still needs the project’s source authority, selectors, consumer bindings and review process.
- Keep report-level encoder/decoder architecture distinct from full reference execution and optional SGLang replay. Sampling remains downstream of logits; replay changes which prompt rows reach later layers.
- Replace universal “indexer precision unknown” wording. A separately labeled logical indexer subtotal can now be derived and bound to the cache/source-ownership chain; it must retain allocation exclusions.
- If Engram placement is added later, represent host-sharded, host-shared and device-sharded ownership explicitly. Do not silently change the existing uniform-weight assumption into an observed placement.
- Miles can inform a dedicated post-training explanation, but generic Adam residency is still only the existing teaching baseline. No memory-fit, throughput, trainer–rollout equivalence or GPU execution receipt follows from these source reads.
- Unchecked within the cap: exact encoder replay scheduler, tail-index construction, FP4 scoring/fusion kernels, Miles helper/bridge implementations, branch merge/release history and full runtime reproduction. Main and development branches must not be conflated.

Stop condition reached: twelve source files/article checked; both repositories’ exact implementation branches identified; no further endpoint expansion, weights, code execution or manifest/report edits performed.
