# GLM-5.3 schematic completion report

## Identity and deliverable

Exact checkpoint: `zai-org/GLM-5.3` at `aca966e4e02791568aa6a4ced368624b3d897f42`. Reference: Transformers `469230357aab0f2b303b0d638c1f8d06edb14184`. Serving comparison: vLLM `155488d853a0bc42df227dbfc74005b3fd488e94`, confirmed through its architecture registry, not inferred from family resemblance. All eight public source files are pinned in `sources.json`.

Portable output: `dist/glm-5.3.html`. Ten sections cover the full decoder, token representations, all layer kinds, residual block, detailed MLA, sparse indexer, router and experts, numerical toy execution, parameter ledger, serving and source boundaries. Layer selection, operation stepping, position, toy layer/head/query/seed and cache batch/context controls change the page. URL state restores selections.

## Evidence and scope

`reference/model.ts` implements the checkpoint’s dense/MoE and full/shared-indexer branches. Real-size shape execution counts **743377000704** decoder trainable parameters. CPU fixture generation imports the pinned upstream implementation and configuration, perturbs every parameter and correction buffer, and records logits, latent norms, attention, selections and router weights. Tested logits and continuous intermediates differ by less than **0.0001**; valid selected-token sets agree and sorted route weights agree to five decimal places. Parameter names match exactly. Tied top-k ordering and masked future filler identities are backend-dependent and are not asserted as identical.

The installed Transformers release rejected the checkpoint’s layer-type vocabulary. Pinned support modules were installed only into ignored `runs/glm-5.3-python`; the fixture uses the pinned model source directly. Reproduction with the available CPU interpreter:

```sh
PYTHONPATH="$PWD/runs/glm-5.3-python" /data03/yph/venv/bin/python models/glm-5.3/reference/make_fixture.py
npm test -- models/glm-5.3
```

The numerical fixture uses reduced dimensions and eight index heads to avoid a degenerate all-zero ReLU top-k tie, while retaining full/shared attention and dense/MoE modes. Browser weights are seeded random teaching weights. No checkpoint weights were downloaded or bundled.

The storage audit independently infers constructor shapes for **118629** pinned tensor-index names. Hub total **753329940480** equals decoder trainable **743377000704** + auxiliary MTP trainable **9952920320** + correction buffers **19456**. Additional FP8 scale scalars **45853104** are excluded from that Hub parameter total. Including them gives **753375793584** stored scalars and exactly **755617140416** bytes, matching index `metadata.total_size`. Audit receipt binds all input bytes; tests reject drift. This is metadata accounting, not an inspection of released tensor contents.

Not verified or drawn: tokenizer behavior and sampling (outside tensor decoder scope); training losses and a checkpoint-specific training implementation (none verified); MTP execution (auxiliary serving path, accounted only in storage); quantized numerical checkpoint equivalence, incremental-cache numerical parity, GPU kernels, allocator memory and performance (no released weights or GPU experiments). Cache controls report logical scalar payload with hypothetical BF16 bytes and exclude allocator padding and scale metadata. Reference latent epsilon is retained as **0.000001**, while serving uses **0.00001**; no equivalence claim hides that difference.

The standalone subtree’s README defines `check` and `shoot` as its gates. Parent Stratum `verify:engineering`, scientific and runtime commands are absent from this package and were not run. No standalone check gate was skipped. No tools, kit, skill, README, git configuration or files outside the permitted model/output directories were intentionally edited. No commits or pushes were made.

## Five randomly sampled source operations

Reproducible sample: Python `random.Random('GLM-5.3-review').sample` over fifteen projection/norm/cache/indexer/FFN operation names. All five were reopened in the pinned reference and checked against constructors, ordering and inspectors:

| Operation | Pinned reference lines | Verified shape and ordering |
|---|---|---|
| Routed `gate_up_proj` | 523–530, 549–551 | Packed `[256,4096,6144]`; selected slice maps token width 6144 to gate/up halves of 2048 before SiLU multiplication. |
| `q_a_proj` | 315–357, 391–393 | `[2048,6144]`; query compression precedes latent RMSNorm and head expansion. |
| Routed `down_proj` | 523–530, 551–554 | Packed `[256,6144,2048]`; down projection follows gating, precedes routed weight multiplication and index-add. |
| Indexer ReLU | 236–241 | `[B,S,32,T]` similarities are rectified before the signed weighted head sum. |
| `kv_a_layernorm` | 315–357, 395–402 | Only latent content 512 is normalized; the positional 64 bypasses it and is rotated separately. |

Source: [pinned modeling_glm_moe_dsa.py](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py).

## Visual review and decisions

All skill example images were opened before design. Four screenshot rounds were generated and inspected in desktop light, desktop dark and mobile light. The initial stock router widget collided at toy dimensions; a spaced custom SVG replaced it. Mobile serving comparisons changed from a narrow table to responsive facts. Independent projection and shared-expert branches retain their actual dependencies; routed inspectors identify packed parameters. Final review corrected sequence-score gauges to the stated retained-key width rather than head or query-projection width. Symbolic shapes remain in inspectors.

Wide diagrams and source/ledger tables deliberately scroll inside their figure/table on mobile, preserving readable typography and complete geometry. Viewport crops at those scrollers are intentional, not lost content. Final screenshots have no observed overlapping text; the automated layout pass measures page overflow, typography and contrast independently. Operation selection is yellow; semantic lanes use ink/residual, blue/attention, orange/experts, violet/state and teal/bypasses consistently.

## Rubric self-review

| Item | Result | Evidence |
|---|---|---|
| 1 exact identity | PASS | Repository and full checkpoint commit appear in the hero. |
| 2 source/fact/coverage/parameter gates | PASS | All pass; numeric parameter boundary is exactly reconciled. |
| 3 five source ops | PASS | Reproducible sample and reopened source lines recorded above. |
| 4 layer kinds | PASS | Dense/full, MoE/full and MoE/shared tiles derive from both explicit config lists; serving schedule equivalence is tested for every layer. |
| 5 serving comparison | PASS | Pinned fused projections, latent decode, distributed norm, FP8 index cache and epsilon difference are described. |
| 6 authored numbers | PASS | Config/code/derived facts supply reader numbers; literal gate passes. |
| 7a clean reference/parity | PASS | CPU upstream fixture and continuous/intermediate parity pass within the stated scope. |
| 7b real shape count | PASS | Real shape mode equals parameterTotal exactly. |
| 7 levels of detail | PASS | Model, layer map, residual block, mechanism blocks, ops and tensor representations are present. |
| 8 detailed graphs | PASS | Projection, latent norm, rotary, cache, expansion, masks, softmax, gate/up/down, dispatch, weighting and recombination are exposed. |
| 9 inspectors | PASS | Each data-path op supplies explanation, formula, shapes and pinned references; parameterless operations have empty parameter lists. |
| 10 code-derived observations | PASS | Epsilon mismatch, rotary output ordering, correction-only choices and selection-only sharing are explained. |
| 11 layer play | PASS | Layer selection changes block kind, expert width and selection-owner sentence. |
| 12 mechanism widgets | PASS | Rotary dials, computed sparse attention heatmap, real-rule toy router and cache arithmetic are interactive. |
| 13 control/URL behavior | PASS | Browser checks exercise changing controls and reload hash persistence. |
| 14 screenshots | PASS | Every final PNG opened; three modes reviewed, internal mobile scrollers intentional. |
| 15 lane colours | PASS | Semantic legend matches graphs; selected-op highlight is distinct. |
| 16 ledes/captions | PASS | Each section states its point and figures explain the visible consequence. |
| 17 coherent design | PASS | Shared kit shell, typography, figure treatment and inspector layout throughout. |
| 18 gauges | PASS | Every op has width; sequence scores use the stated retained-key example and complete shapes remain inspectable. |
| 19 Takeaways/Check | PASS | All ten sections end with Takeaways; mechanism sections include reasoning questions. |
| 20 mechanism benefits | PASS | Cache compression, 78-to-21 indexer execution reduction and 8-of-256 expert selection are derived and explained. |
| 21 glossary | PASS | Technical mechanism terms are defined in the source section and local explanations. |
| 22 honesty | PASS | Exclusions, numerical boundaries, unrun parent commands and gate status recorded explicitly. |

## Final verification

Generated check receipt: `runs/check/glm-5.3/report.json`.

| Gate | Status | Generated details |
|---|---|---|
| pins | PASS |  |
| typecheck | PASS |  |
| literals | PASS |  |
| format | PASS |  |
| tests | PASS | Tests  14 passed (14); includes the clean reference's parity test |
| facts | PASS |  |
| coverage | PASS |  |
| parameters | PASS | parameterTotal 743377000704 differs from safetensors.total 753329940480 by -9952939776; parameterNote: Hub total 753329940480 = decoder trainable 743377000704 + excluded MTP trainable 9952920320 + router correction buffers 19456. Every tensor-index name is reconciled by constructor shape in reference/storage-audit.py; FP8 scale metadata is additional storage excluded from the Hub parameter total and reconciled separately against index metadata.total_size. |
| build | PASS | dist/glm-5.3.html 520816 bytes sha256 9983686e352df4a5405140a7b84dcd90d450d64a4dd7d32f065f28095cc8a43a |
| layout | PASS | desktop-light, desktop-dark, mobile-light: 2336 text elements checked |
| play | PASS | 160 controls (90 button, 64 op, 6 range); 12 of 12 sampled changed the page; reload kept #position=524288&toy_layer=3&head=1&query=6 |

**11 PASS, 0 FAIL, 0 SKIPPED; 14 tests pass.** Browser: Google Chrome 145.0.7632.75 at `/usr/bin/google-chrome`. Layout probes cover desktop light/dark and mobile light; interaction sampling checks 12 of 160 controls, not an exhaustive click of every control.

Final artifact: **520816 bytes**, SHA-256 `9983686e352df4a5405140a7b84dcd90d450d64a4dd7d32f065f28095cc8a43a`. Final screenshot set: **33 PNGs**, hero plus all ten sections in each of the three modes, captured from that HTML with `npm run shoot -- glm-5.3`. `screenshot-review.json` records the artifact binding and every image digest. Desktop viewport 1440×1000 at scale 1; mobile 390×844 at scale 2. All final PNGs were opened after capture.
