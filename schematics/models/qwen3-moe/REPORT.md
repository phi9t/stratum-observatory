# Qwen3-235B-A22B schematic report

Built the interactive schematic in `models/qwen3-moe/` for **Qwen/Qwen3-235B-A22B**, checkpoint commit `8efa61729e24bd65b1d152b5ab5409052aa80e65`. The portable output is `dist/qwen3-moe.html`.

## Sources and decisions

`sources.json` pins checkpoint config and Transformers v4.51.3 reference/configuration/RoPE code at `5f4ecf2d9f867a1255131d2461d75793c0cf1db2`; vLLM v0.8.5 serving code at `3015d5634e74d59704e2b39bab0dbe2e6f86a38a`; and Megatron-Bridge conversion/provider code at `ec4ec358c3622134905144ca193516b194feddd5`. `symbols.json` supplies exact symbol links; `coverage.json` accounts for every reference class. No model weights were downloaded.

The pinned branch makes every layer MoE: `mlp_only_layers=[]` and `decoder_sparse_step=1`. Full causal GQA uses explicit head dimension 128, giving query width 8192 despite residual width 4096. The cache retains rotated keys and unrotated values before eager KV repetition. The decoder has no shared experts, sliding window, router bias correction, or tied embedding/head. The unused dense intermediate config does not enter the ledger. Layer selection changes the owning parameter namespace; it does not invent different layer kinds.

The complete reference source, serving implementation, configuration, default RoPE helper and training bridge were read. Megatron-Bridge establishes conversion/provider mappings, not executed training parity. vLLM's packed projections, fused expert dispatch, separate residual lifetime and backend cache ownership are explained with pinned links.

The ten sections cover whole model, data representation, stack, block, detailed attention, detailed routed experts, executable toy experiment, parameter accounting, serving and sources/glossary. Experiments execute the same clean reference arithmetic with seeded random toy weights. Shape-only expert gathers use an all-token upper bound because actual route loads require numerical execution; they never substitute missing evidence with zero.

## Verification on current bytes

`npm run check -- qwen3-moe` passes all 11 gates, with no failures or skips. It runs 11 tests, including upstream CPU parity, future-token invariance, causal masking, normalized attention/router sums, parameter shape/count and independent split-half rotary checks. Fractional KV repetition was first demonstrated as a failing negative test and then rejected locally.

| Gate | Result | Evidence |
| --- | --- | --- |
| pins | PASS | Pinned source bytes resolve |
| typecheck | PASS | Model-local TypeScript |
| literals | PASS | Config/code/derived facts centralized |
| format | PASS | Model-local formatter |
| tests | PASS | 11 passed, including clean reference parity |
| facts | PASS | Config/code/derivation bindings |
| coverage | PASS | Reference class inventory accounted for |
| parameters | PASS | 235093634560 equals pinned safetensors total |
| build | PASS | Deterministic, self-contained HTML |
| layout | PASS | 2573 text elements across three modes |
| play | PASS | 175 controls; 12/12 sampled controls changed page; URL restored |

The real-config shape pass registers 36945 tensors totaling **235093634560 parameters**. The CPU fixture has 45 parameter tensors, batch 2 and sequence length 6; its architecture preserves RMSNorm, QK norm, GQA, split-half RoPE, gated SiLU and normalized sparse routing while reducing dimensions. Its generator verifies installed upstream module bytes against the pinned source files before execution. No trained weights are involved.

Measured maximum absolute errors from `reference/validation.json`:

| Output | Maximum absolute error |
| --- | ---: |
| logits | 2.980232238769531e-7 |
| attention | 1.7881393432617188e-7 |
| router logits | 2.086162567138672e-7 |
| router mixture weights | 1.1920928955078125e-7 |
| selected expert IDs | 0 |

The validation file records SHA256 fingerprints of the current reference, ops, fixture and facts. Portable artifact: **528087 bytes**, SHA256 `dfcb9aa0cb4a4b8dec8e90c7fc72729b0b513cdd0896f329dec3847c85e72bfe`. Check report: `runs/check/qwen3-moe/report.json`.

`npm run shoot -- qwen3-moe` generated 33 PNGs (hero plus ten sections in each of desktop-light, desktop-dark and mobile-light) from this artifact. All final PNGs were opened and visually inspected. Browser: `/usr/bin/google-chrome`, **Google Chrome 145.0.7632.75**. Desktop viewport 1440×1000 at scale 1; mobile 390×844 at scale 2. Files are under `runs/shots/qwen3-moe/`. Four capture rounds refined the page; final review found no clipped/overlapping text. Wide diagrams/tables scroll inside their mobile panels. Visual review against the supplied examples led to shorter inspector parameter labels with namespace retained separately, a wider model-local toy-router graphic, and more balanced serving facts. No kit or shared CSS was edited. The small residual-norm ledger share is explicitly stated as nonzero alongside the kit's rounded display.

## Rubric self-review

| Item | Result | Evidence |
| --- | --- | --- |
| 1 | PASS | Exact checkpoint repo/commit appears in hero and sources. |
| 2 | PASS | Pins, facts, coverage and exact parameter gates pass. |
| 3 | PASS | Five randomly selected ops were checked against pinned lines, as detailed below. |
| 4 | PASS | Uniform MoE layer map shows exact dense-exclusion and sparse-step conditions. |
| 5 | PASS | Serving section cites projection packing, FusedMoE, residual and cache ownership differences. |
| 6 | PASS | Architectural quantities derive from facts; demo/layout constants are separately identified. |
| 7a | PASS | Clean sequential reference takes this checkpoint's branches and passes upstream CPU parity. |
| 7b | PASS | Real-config shape count equals 235093634560. |
| 7 | PASS | All prescribed levels appear across ten sections. |
| 8 | PASS | Detailed attention and MoE graphs expose arithmetic, routing and cache steps separately. |
| 9 | PASS | Inspectors provide meaning, formula, shapes, parameter information where applicable and source links. |
| 10 | PASS | Explains wider query path, shared head-width gains, cache-before-repeat and expert-bank dominance. |
| 11 | PASS | Layer controls change the block namespace and explanatory sentence. |
| 12 | PASS | Computed routing, causal GQA attention and positional rotation widgets use reference rules. |
| 13 | PASS | Control handlers were reviewed; automated sample changes 12/12 and reload preserves scenario hash. |
| 14 | PASS | All 33 final light/dark/mobile screenshots reviewed; mobile panel scrolling is deliberate. |
| 15 | PASS | Ink/blue/orange/violet/teal lane grammar matches the stated legend. |
| 16 | PASS | Section ledes and figure captions explain the mechanism and what to notice. |
| 17 | PASS | Shared kit typography/components/tokens throughout; no CSS override. |
| 18 | PASS | Data-path operations supply width gauges. |
| 19 | PASS | Every section ends with Takeaways; mechanism Checks require reasoning. |
| 20 | PASS | Derived GQA cache reduction, selected expert fraction and stored-weight accounting explain benefits. |
| 21 | PASS | Sources section defines 23 terms used in the exposition. |
| 22 | PASS | Explicit exclusions, unverified scope and operational deviation recorded here. |

Five-op spot check, sampled with seed 17: **router renormalize**, **router linear**, **attention softmax**, **output projection**, **expert SiLU**. All links below use the pinned Transformers reference:

- [Router linear and renormalize](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L252-L265): flattened `[N,4096]` enters biasless router `[128,4096]`; FP32 softmax precedes top-8 selection and selected-sum division.
- [Attention softmax](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L114-L134): scaled QK scores receive the mask before FP32 softmax, preserving `[B,64,S,T]`.
- [Output projection](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L218-L220): merged `[B,S,8192]` projects through `[4096,8192]` back to the residual width.
- [Expert SiLU](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L223-L236): configured SiLU transforms the gate projection `[Ne,1536]` before multiplication with the parallel up projection and down projection.

## Excluded and unverified scope

Tokenization, sampling, training losses/recipes, non-causal task heads, quantization and distributed kernel internals are excluded because the requested schematic traces this checkpoint's decoder through vocabulary logits. Base/configuration infrastructure and task-head exclusions are individually recorded in `coverage.json`. No language quality, trained-route or checkpoint-output claims are made.

Numerical parity covers unpadded CPU float32 toy prefill. Incremental decode/cache mutation and padding are source-inspected and drawn, not numerically parity-tested. GPU kernels, BF16 numerical parity, distributed serving/training parity, latency and throughput were **NOT_RUN**: no GPU experiment or trained weights were used. BF16 KV bytes are an explicit storage assumption and logical payload calculation, not a benchmark. Training source scope is the pinned bridge rather than imported distributed implementation execution.

No local check gates were skipped. The parent STRATUM engineering/science/runtime commands and named parent contract documents are absent in this self-contained subtree; README explicitly states parent gates do not cover it. They were **NOT_RUN**, and no publication/runtime approval is claimed.

**Constraint deviation:** the cache-local Python dependency install used pip's default download cache at `~/.cache/pip`, unintentionally permitting cache writes outside the allowed subtree. Dependencies themselves were installed only under ignored `.sources-cache/qwen3-moe/python`; no global package installation or home source/config edit occurred. Exact cache files written were not tracked, so their extent cannot be established. No out-of-scope cleanup was attempted to conceal this. Future reproduction must set `PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1`. This violates the run's filesystem constraint and is recorded rather than presented as full operational compliance.

No commits, pushes, branches or git configuration changes were made. Tracked worktree changes are confined to the new model directory; tool artifacts are ignored.
