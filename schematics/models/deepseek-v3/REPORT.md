# DeepSeek-V3 schematic report

Built for `deepseek-ai/DeepSeek-V3`, revision `e815299b0bcbac849fa540c768ef21845365c9eb`.

## Delivered

`dist/deepseek-v3.html` is the standalone page. Ten sections cover the whole model, tensor representation, all main layers, residual block, detailed attention, YaRN, dense and routed/shared FFNs, parameters, serving differences and sources/glossary. Eleven operation graphs expose 61 selectable operations with formulas, input/output shapes, parameter shapes where applicable, output-width gauges and pinned source links. The layer selector distinguishes absolute zero-based dense indices 0–2 from MoE indices 3–60. Routing, causal access, YaRN phase and cache representation/size controls execute derived rules and retain state in the URL.

Eleven sources are pinned in `sources.json` and fingerprinted by the fetch tool. The publisher checkpoint supplies configuration, reference implementation, inference implementation, weight index and weight documentation. Independent comparisons use vLLM `155488d853a0bc42df227dbfc74005b3fd488e94` and torchtitan `e3929368edd2f2457456913890ba2b4c0f27da31`. Reading notes and decisions are in `NOTES.md`; no upstream source is pasted into the deliverable.

The main forward count is `671026419200`. The hub stored scalar count `684531386000` is reconciled exactly by adding `11610068224` unique MTP parameters, `1853358080` separately named MTP embedding/head entries and `41540496` FP8 block-scale scalars. Scale sizes use ceiling block arithmetic, including the nonaligned KV A matrix. The numeric `parameterNote` explains why the main forward count differs from the stored count.

## Decisions, exclusions and limits

- The operation walkthrough follows the reference's eager evaluation path. FlashAttention's fusion, value padding before cache update and resulting cache width are explained separately; its fused kernel internals are not portrayed as individual eager operations.
- Compressed decode is an algebraic comparison grounded in publisher absorption code and vLLM's MLA wrapper. Eager cache stores `40960`, reference FlashAttention stores `49152`, and compressed MLA stores `576` scalars per token per layer. Cache bytes assume illustrative BF16 precision and exclude allocation, padding, replication, quantization metadata and workspaces. These are not GPU measurements or throughput estimates.
- The reference's grouped biased router is implemented: group scores sum the strongest corrected scores; excluded groups use negative infinity; mixture weights gather original sigmoid scores before normalization and scaling. Shared experts have no router logits. Publisher inference instead zero-masks excluded groups, which can disagree for negative corrected scores; this difference is disclosed rather than silently unified.
- Synthetic scores and biases illustrate routing; no learned weights or biases were downloaded. Stable lowest-index tie ordering in the teaching fixture is not a claim about PyTorch top-k tie ordering.
- YaRN uses the checkpoint's blended frequencies and independently modified attention scale. Custom model-local dials replace the kit's plain-RoPE dials because the latter would execute a different rule. Other layout and interaction primitives use the kit.
- Auxiliary MTP storage is reconciled, but MTP training/speculative execution is outside the main circuit. Auxiliary and scale shapes are inferred from pinned code, documentation and index names; no tensor headers or weight values were downloaded. Numerical checkpoint parity, trained quality, GPU execution and bitwise equivalence remain unverified.
- Linear-scaling and dynamic-NTK rotary classes are inactive for this YaRN checkpoint. Pretrained loading/initialization infrastructure and the sequence-classification head are excluded with explicit coverage reasons. Direct Q projection is inactive because Q rank is configured.
- Tokenizer internals, sampling, losses, distributed collectives, optimizer state and quantization kernels are outside the inference schematic. Serving/training source comparisons do not claim complete execution coverage of those systems.
- Mobile keeps diagrams and wide evidence tables horizontally scrollable at readable size. The capture shows their initial scroll positions, not missing graph content.
- Only model-local files were authored. No changes to kit, tools, skill, README, git configuration or external worktree files; no commits or package installation.

## Validation and screenshot review

`npm test -- models/deepseek-v3/tests`: **2 files, 11 tests passed**. Seven mechanism tests cover dense/MoE boundaries and rejection, correction bias versus mixture weights, negative corrected-score group exclusion, agreement with the kit's zero-bias router, YaRN endpoints/amplitude/scale, all three cache forms and invalid input, and exact index/MTP/scale reconciliation. Behavioral tests first failed before the mechanism implementation was added; the supplied four facts tests also pass.

Browser: **Google Chrome 145.0.7632.75**, `/usr/bin/google-chrome`. Layout checks examine **2582 text elements** across desktop light, desktop dark and mobile light. Play checks inventory **143 controls**: 73 buttons, 61 operation nodes, 4 ranges and 5 radio choices; **12/12 sampled interactions** changed the page and the exercised URL restored on reload. This is sampled interaction coverage, not an exhaustive proof of all control combinations.

Three `npm run shoot -- deepseek-v3` rounds generate **33 PNGs per round**: hero plus ten sections in each of desktop light, desktop dark and mobile light. Every generated PNG was opened and reviewed. The first review caught a clipped tensor caption; it was shortened without reducing type. Dependency review also separated parallel compressed-cache state, repositioned query/key assembly branches and corrected packed expert token shapes. Later reviews confirm consistent lane colours, readable spacing and no observed unintended clipping or overlap; horizontal mobile overflow is contained in scrollable figures/tables.

Final artifact: **527573 bytes**, SHA-256 `2f3377a895aa5877bce672c84af71cdc85934c3c8eddfa355fb31984819de928`. Screenshots are under `runs/shots/deepseek-v3/`; they depict this artifact.

All ten local skill gates ran; none was skipped. The parent STRATUM `verify:engineering`, scientific and runtime scripts/docs are absent from this standalone subtree (see README and package scripts). Those parent gates are **NOT_RUN**, not represented as PASS; the subtree's actual check contract is used here. Scientific acceptance, an independent review signature and GPU runtime evidence are not established by these software checks.

## Self-review against the skill rubric

| # | Result | Evidence |
|---|---|---|
| 1 | PASS | Exact repository and full checkpoint revision appear in the hero and source table. |
| 2 | PASS | Pins, facts and coverage pass; the parameter gate accepts an exact numeric reconciliation. |
| 3 | PASS | Five sampled operations were rechecked against pinned reference lines: Q A projection and KV B projection in `DeepseekV3Attention.forward` L758–859, rotary application in that forward and `apply_rotary_pos_emb` L339–371, original-score gathering in `MoEGate.forward` L422–473, and down projection in `DeepseekV3MLP.forward` L388–390; order, axes and configured widths agree. |
| 4 | PASS | Every main layer is selectable and the exact constructor condition is printed. |
| 5 | PASS | Serving cites fused Q/KV A, gate/up, residual handling, compressed state and expanded fallback. |
| 6 | PASS | Config/code/derived quantities and illustrative fixture constants live in `facts.ts`; literals gate passes. |
| 7 | PASS | Model, representation, layer, block, operation, weight and serving levels are present. |
| 8 | PASS, scoped | Eager attention and FFN/MoE operations are explicit; fused alternatives and excluded training/speculation are stated. |
| 9 | PASS | All operation inspectors have explanations, formulas, shapes and pinned references. |
| 10 | PASS | Cache expansion/padding, choice bias versus mixture weights and exact stored-count reconciliation require reading code. |
| 11 | PASS | Dense versus MoE selection changes block, FFN widths and the explanatory sentence. |
| 12 | PASS | Causal access, grouped biased routing, YaRN and cache representation widgets execute the documented rules. |
| 13 | PASS, sampled | All controls have stateful behavior; automated play samples 12 interactions and verifies reload. |
| 14 | PASS | All light/dark/mobile PNGs opened in each round; observed caption clipping repaired. |
| 15 | PASS | Attention blue, FFN orange, state violet, residual skips teal and selected operation yellow match the legend. |
| 16 | PASS | Ledes state each section's point and figures carry interpretation captions. |
| 17 | PASS | Shared kit shell, typography, inspector and control primitives maintain one design. |
| 18 | PASS | Every data-path operation specifies output width; illustrative attention-history width is labelled. |
| 19 | PASS | Every section ends with Takeaways; mechanism Checks ask causal reasoning. |
| 20 | PASS | Derived expansions, active-versus-stored FFN matrices and compressed-cache reduction explain benefits. |
| 21 | PASS | Glossary defines attention, positional, routing, cache, precision and auxiliary terminology used by the page. |
| 22 | PASS | Exclusions, unverified execution/parity and unavailable parent gates are explicit above. |

## Final check table

Command: `npm run check -- deepseek-v3`

| Gate | Result |
|---|---|
| pins | PASS |
| typecheck | PASS |
| literals | PASS |
| format | PASS |
| facts | PASS |
| coverage | PASS |
| parameters | PASS |
| build | PASS |
| layout | PASS |
| play | PASS |

**10 pass · 0 fail · 0 skip**
