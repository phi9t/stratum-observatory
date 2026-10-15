# GLM-5.3-Flash schematic report

Built the exact Hugging Face checkpoint `zai-org/GLM-5.3-Flash-BF16` at
revision `a5b45eb41df6402735dedc900be14a42e8d5e538` as a portable schematic page.
All authored files are under `models/glm-5.3-flash/`; generated outputs are under
the ignored `dist/`, `runs/` and `.sources-cache/` directories. No weights were
downloaded, no shared kit/tooling files were edited, and no git commit, branch,
push or configuration change was made.

## Artifact and source identity

- Page: `dist/glm-5.3-flash.html`, 523277 bytes, SHA-256
  `862eff9c6147d511dcfe0c8f64f8231e6b20af9a4336faed15dac9f5ac9b66ec`.
- Reference: `huggingface/transformers` commit
  `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`,
  `src/transformers/models/glm5_next/modeling_glm5_next.py`.
- Serving comparison: `vllm-project/vllm` commit
  `273a04964ed9d1e389c4a7cb9d660e8294782f8d`,
  `vllm/models/glm5next/common/{model.py,attention.py,kda.py,sparse_indexer.py,mtp.py}`.
- Local training evidence: TorchTitan citations are displayed as
  repo-relative paths plus commits where reports are tracked. They are
  read-only local evidence, not fetched public source entries, because the
  schematic fetcher accepts only Hugging Face and GitHub sources.

## What is drawn

The page has twelve sections: Circuit, Data, Layers, Block, KDA, Sparse, Experts,
Training, Run, Weights, Serving and Sources. It draws the released text decoder
with `45` base layers, a 3:1 KDA versus sparse-MLA schedule, dense MLP only in
layers `0..2`, MoE in layers `3..44`, four-stream mHC boundaries, NoPE sparse
MLA, IndexPool, sigmoid top-8-of-288 routing plus one shared expert, one stored
MTP layer, source-backed serving notes and TorchTitan training/scaling evidence.

The prompt mentioned possible sparse-layer top-k reuse. The pinned config and code
win: every base sparse layer has `indexer_types == "full"`. The page and tests
state that base sparse layers do not consume another layer's top-k. The only
checked reuse flag is MTP draft-iteration reuse in serving.

Training and scaling content separates derived estimates from measured campaign
rows. The page records the LoRA SFT memory ledger, measured full-depth EP8
readiness and rung-4 runs, first-8-layer FSDP/grouped/EP comparisons,
long-context estimates, the SFT ladder status, and the DeepEP boundary. Rung 2
now cites the completed T06 30B-A3B Countdown pass, and rung 4 cites T08f logs:
seed 42 trained 94 steps with loss `1.67393` to `0.00525`, seed 43 trained 94
steps with final loss `0.00778`, and the step-62 HDFS resume matched exactly for
steps `63..94` with max absolute delta `0.0`. GLM DeepEP did not complete, so
the page makes no GLM DeepEP loss, gradient or step-time claim.

## Parameter accounting

The page reconciles the Hugging Face `safetensors.total` exactly:
`321323031390` stored parameters. The ledger includes text backbone stored
tensors, MTP and vision tensors:

- text backbone stored tensors: `313326811966`
- MTP stored tensors: `7432592416`
- vision tower/projector tensors: `563627008`

Their sum is `321323031390`. This is stored-tensor accounting, not active
per-token compute, packed bytes, optimizer memory or allocator peak.

## Clean reference and tests

`reference/model.ts` is intentionally shape-only. It enumerates stored tensors,
layer schedules, major tensor shapes and parameter counts without loading model
weights. Shape mode at the real config equals `parameterTotal`.

Model-local tests verify:

- source pins, facts, coverage and parameter reconciliation
- the exact 3:1 KDA/sparse schedule
- no base cross-layer top-k reuse
- dense-prefix versus MoE schedule
- the corrected SFT ladder evidence and whole-model mHC activation estimate
- sparse-layer score and top-k shapes in the clean reference

Whole-model numerical parity against Transformers or vLLM is **unverified**. No
released weights were downloaded, and no logits or GPU-kernel equivalence is
claimed.

## Five sampled source operations

Sample method: deterministic LCG with seed `53` over attention, indexer and FFN
operation names. Each selected operation was reopened in the cached pinned
Transformers source.

| Operation | Pinned source lines | Verified behavior |
| --- | --- | --- |
| MoE routed plus shared | `Glm5NextTextMoE.forward` L199-206 | The gate returns top-k weights/indices, routed experts run on the flattened hidden states, and `shared_experts(residuals)` is added separately. |
| Router sigmoid and correction | `Glm5NextTextTopkRouter.forward` L157-182 | Router logits become sigmoid scores; `e_score_correction_bias` is added for selection, while gathered original sigmoid scores become normalized scaled weights. |
| Expert gate/up/down | `Glm5NextTextExperts.forward` L119-134 | Selected experts apply packed gate/up projection, clamp/SwiGLU, down projection and route-weight scaling before `index_add_`. |
| IndexPool pooled states | `Glm5NextTextIndexer.get_pooled_states` L935-1008 | Keys, gate scores and validity split from packed state; four-key pools are weighted by gate score plus APE and invalid pools are masked. |
| Sparse MLA KV expansion | `Glm5NextTextAttention.expand_kv` L1155-1172 | Latent KV is expanded through `kv_b_proj`, split into NoPE key/value heads, and the zero-width text RoPE slot is concatenated by shape. |

## Visual review

`CHROME=/usr/bin/google-chrome npm run shoot -- glm-5.3-flash` generated 39
PNGs: hero plus twelve sections in desktop light, desktop dark and mobile light.
Browser: Google Chrome 145.0.7632.75 at `/usr/bin/google-chrome`. The training
section PNGs were opened directly in all three modes after the final shoot:
`desktop-light-ms-training.png`, `desktop-dark-ms-training.png` and
`mobile-light-ms-training.png`.

Fixes made during screenshot review included:

- LoRA memory and long-context budget bars now put every label and value in a
  stable 336px SVG layout, so mobile, light and dark captures show the full
  labels instead of clipped bar-end values.
- The EP diagram no longer implies `4 x 36 = 288`: it shows four ranks plus an
  ellipsis labelled as ranks `4..7`, and states `EP8 = 8 ranks x 36 routed
  experts = 288 total experts`.
- The long-context chart labels the mHC saved residual term as whole-model,
  separately shows the per-layer estimate, labels the indexer score term per
  sparse layer, and includes the estimated activation room line
  `178.35 GiB - 73.05 static - 6.00 buffers = 99.30 GiB/GPU`.
- Evidence citations use short `torchtitan:` repo-relative paths and report
  commits instead of long absolute local paths where a commit exists.

Wide evidence tables with TorchTitan citations still compress on mobile, but the
custom training figures no longer rely on hidden horizontal overflow; the layout
gate independently checks text overflow and contrast.

## Exclusions and limits

- Vision tower internals are accounted for in the parameter ledger but not drawn
  op by op.
- Whole-model numerical parity, released-weight logits, tokenizer behavior,
  sampling, CUDA kernels, vLLM runtime parity and production cache behavior are
  unverified.
- TorchTitan evidence was only read; no command was run in the TorchTitan repo.
- The requested `task-08f-report.md` was absent, so rung 4 is cited from T08f
  logs and `progress.md`. T08f adapter export logs exist for both seeds, but no
  completed Countdown eval pass@1 result was found; eval remains pending on the
  page.
- Long-context numbers are estimates derived from config, not measurements.
- Root Stratum engineering/science/runtime gates are outside this self-contained
  schematics subtree and were not run.

## Rubric self-review

| Item | Result | Evidence |
| --- | --- | --- |
| 1 exact identity | PASS | HF repo and full checkpoint revision are pinned and shown. |
| 2 check gates | PASS | pins, facts, coverage and parameters pass. |
| 3 sampled source ops | PASS | Five reopened source operations are recorded above. |
| 4 layer kinds | PASS | Layer map distinguishes dense KDA, MoE KDA and sparse-MLA MoE using config-derived predicates. |
| 5 serving comparison | PASS | vLLM buffer/fusion/MTP differences are cited from pinned serving files. |
| 6 authored numbers | PASS | Reader-facing quantities come from `facts.ts`; literal gate passes. |
| 7a clean reference | SCOPE | Shape-only reference exists; numerical parity is explicitly unverified. |
| 7b real shape count | PASS | Shape mode equals `parameterTotal`. |
| 7 detail levels | PASS | Overview, data, layer map, block, mechanisms, execution, weights, serving and sources are present. |
| 8 detailed graphs | PASS | Block, KDA, sparse/indexer and expert paths expose projection/norm/cache/routing boundaries; vision is excluded. |
| 9 inspectors | PASS | Operation graphs have descriptions, shapes, widths and references. |
| 10 code-derived observations | PASS | No base top-k reuse, selection-only correction bias and shared-expert independence come from code. |
| 11 layer play | PASS | Layer selection changes the map/readout and run sentence. |
| 12 mechanism widgets | PASS | Layer circuit, mHC, IndexPool, sparse grid, router, parallelism and context controls run source rules or labelled estimates. |
| 13 URL behavior | PASS | Play gate sampled controls and reload persistence. |
| 14 screenshots | PASS | Final light/dark/mobile captures were reviewed; no observed clipped custom figures remain. |
| 15 lane colors | PASS | Attention, expert, state and residual lanes follow the kit palette consistently. |
| 16 ledes/captions | PASS | Section ledes and figure captions state the reading target. |
| 17 one design | PASS | Shared kit shell, cards, tables, figures and inspectors are used throughout. |
| 18 gauges | PASS | Data-path ops carry widths. |
| 19 teaching endings | PASS | Every section now ends with Takeaways; mechanism sections include Checks. |
| 20 mechanism benefits | PASS | IndexPool, MoE active route, mHC saved activation and EP memory/timing benefits have derived or measured quantities. |
| 21 glossary | PASS | Architecture terms introduced by the page are defined in Data and local explanations. |
| 22 honesty | PASS | Exclusions, unverified parity, missing ladder report and non-run root gates are listed here and on the page. |

## Final local check

`CHROME=/usr/bin/google-chrome npm run check -- glm-5.3-flash`

| Gate | Status | Details |
| --- | --- | --- |
| pins | PASS |  |
| typecheck | PASS |  |
| literals | PASS |  |
| format | PASS |  |
| tests | PASS | Tests 12 passed (12); runs the clean reference's tests |
| facts | PASS |  |
| coverage | PASS |  |
| parameters | PASS | `parameterTotal 321323031390 equals safetensors.total` |
| build | PASS | `dist/glm-5.3-flash.html 523277 bytes sha256 862eff9c6147d511dcfe0c8f64f8231e6b20af9a4336faed15dac9f5ac9b66ec` |
| layout | PASS | desktop-light, desktop-dark, mobile-light: 2087 text elements checked |
| play | PASS | 70 controls (47 button, 17 op, 4 range, 2 radio); 12 of 12 sampled changed the page; reload kept `#query=6&parallel=fsdp&route=5000&ctx=2` |

**11 pass, 0 fail, 0 skipped.** Receipt:
`runs/check/glm-5.3-flash/report.json`.
