# DeepSeek-V3 schematic report

Checkpoint: `deepseek-ai/DeepSeek-V3` at `e815299b0bcbac849fa540c768ef21845365c9eb`. The current-skill upgrade preserves the original checkpoint facts, main parameter ledger, layer selector, detailed eager diagrams, serving comparison and teaching aids. It adds a clean executable reference, independent upstream parity, a browser “Run it” section, and topology summaries before the split attention and MoE walkthroughs.

## Delivered

The standalone artifact is `dist/deepseek-v3.html`. Eleven sections cover the whole model, data representation, layer stack, residual block, attention, YaRN, dense and routed/shared FFNs, the executable toy run, weights, serving and provenance. Thirteen operation graphs expose 74 selectable operations. The attention topology has seven boxes in two rows; the MoE topology has six boxes in three rows, with token input connected to both routed and shared transforms. Existing detailed operations were retained.

`reference/model.ts` distils the publisher's module arithmetic into functions on `tnsr/tensor.ts`. It uses the original checkpoint parameter names and marks intermediate tensors using the detailed graph's operation ids, qualified by layer. `reference/ops.ts` supplies independently tested shared-key head broadcasting and adjacent-pair/half-layout YaRN rotation. Shape mode visits every expert without allocating checkpoint-sized values and gives exactly **671026419200** main parameters, equal to `facts.ts.parameterTotal`.

`reference/make_fixture.py` imports the pinned publisher code directly from the source cache. It runs the complete upstream causal LM on CPU at toy widths: two dense and two MoE layers, two heads, eight routed experts, top-k two, four groups, two selected groups, a shared FFN, both compressed projection paths and partial YaRN. All parameters are replaced with seeded values, including correction biases and nonidentity norm gains. The **126526-byte** fixture contains **103 parameter tensors**, logits and **18 intermediate tensors**. The test requires exactly the fixture's parameter-name set, with none missing or invented. It also binds the fixture to the cached source SHA-256.

Measured fixture differences: logits maximum absolute difference **3.5762786865234375e-7**; intermediates maximum absolute difference **3.2782554626464844e-7**. Both are below the required `1e-4`. Intermediate comparisons include every layer's normalized latent, KV B expansion and attention probabilities, plus both MoE layers' group ids, expert ids and unbiased scaled mixture weights. The toy model has **8964** parameters. These are toy upstream parity results, not trained-checkpoint predictions.

The browser executes this same complete clean forward pass with random seeded weights. Run it compares the normalized cached latent and shared rotated positional key against the reconstructed full key and value of the selected head; a scale figure shows their scalar budgets. The sequence table gives every token's selected groups, experts and paired weights. Layer, head, token and seed controls retain state in the URL. The earlier expert widget now also consumes the reference's own computed stream/router outputs. A model-local `ToyRouter` uses the kit's existing classes and readable bar spacing: the kit's large-bank router overlapped labels when used for a small toy bank. No kit code or fonts were changed.

The main ledger remains reconciled with the hub's **684531386000** stored scalars. Its exact **13504966800** difference comprises **11610068224** unique MTP parameters, **1853358080** separately stored MTP embedding/head entries and **41540496** FP8 block-scale scalars. The existing numeric `parameterNote` remains intact.

## Decisions, exclusions and unverified items

- Reference scope is the unpadded full-sequence text decoder in eager eval mode to logits. It keeps the checkpoint's configured low-rank query branch, SiLU, sigmoid/noaux_tc grouped routing, renormalization, scaling, untied head and dense/MoE layer condition. No inactive architectural branch was invented.
- The generator ran using `/data03/yph/venv/bin/python`: torch **2.14.0+cpu**, transformers **5.17.0**. Transformers removed `is_torch_fx_available`; the generator supplies this tracing-only availability helper for import compatibility. No upstream attention, norm, FFN, router or decoder arithmetic is patched. Existing upstream deprecation/generation warnings do not prevent the CPU forward. Regenerate with `${SCHEMATICS_PYTHON:-/data03/yph/venv/bin/python} models/deepseek-v3/reference/make_fixture.py`.
- Upstream uses `topk(sorted=False)`, whose output ordering is immaterial to the weighted mixture. Fixture/test canonically sort expert/weight pairs and group ids. Clean tie ordering chooses lowest id; equivalence of tied PyTorch ordering is not claimed.
- Shared experts have no router logits. Correction bias changes selection only; original sigmoid scores supply weights. Negative corrected scores still exclude unselected groups with negative infinity. Publisher absorption inference instead zero-masks excluded groups, a pre-existing documented difference that was not silently unified.
- The marked latent and shared key identify what compressed serving can cache; the executable toy reconstructs eager full K/V. It does **not** implement persistent incremental cache updates or absorbed decode. Expanded-cache plumbing and fused FlashAttention execution are described on the page but are outside this clean full-sequence program's numerical parity scope.
- Eager, FlashAttention and compressed storage keep **40960**, **49152** and **576** scalars respectively per token per main layer. Logical cache payload controls exclude allocation, replication, padding, quantization metadata and workspaces. No GPU memory or speed measurement is claimed.
- MTP tensors and FP8 scales remain accounted for in storage but excluded from main forward execution. Auxiliary shapes are grounded in pinned code, documentation and index names; no tensor headers or model weights were downloaded. Tokenization, sampling, training losses, optimizer state, distributed collectives, quantization kernels and serving/training execution parity remain outside scope.
- Full trained-checkpoint numerical parity, model quality, long-context capability, GPU behavior and bitwise precision equivalence are unverified. Toy numerical parity establishes the scoped arithmetic only. Local custom ops do not claim automatic differentiation.
- Mobile figures and wide tables scroll within their panels at readable size. Captures show their initial scroll positions, so offscreen graph/table content is intentionally not visible in those images.
- All authored changes are under `models/deepseek-v3/`; generated tool outputs are ignored. No kit, tool, skill, README, upstream pin or git configuration was changed; no commits, branches or packages were installed.

## Validation and screenshot review

`npm run check -- deepseek-v3`: **11 pass, 0 fail, 0 skipped**. Its tests gate ran **16 passing tests in four files**, including upstream parity, exact parameter-name consumption, full-width count/shapes, negative corrected-group exclusion, broadcast and independently specified rotary values, plus the retained facts/mechanism tests. The initial new reference test failed before `model.ts` existed. Early page checks caught incorrect TensorScale props and disallowed layout literals; both were repaired locally.

Browser: **Google Chrome 145.0.7632.75**, `/usr/bin/google-chrome`. The layout gate checked **3185 text elements** across desktop light, desktop dark and mobile light. The play gate found **160 controls**: **76 buttons**, **74 operation nodes**, **7 ranges** and **3 radio choices**; **12/12 sampled interactions** changed the page and the exercised URL survived reload. This shared play gate is sampled coverage.

`node models/deepseek-v3/tests/browser-check.mjs` additionally passed on the current standalone artifact. It independently executes the reference outside the browser and compares all displayed latent/key/value vectors and every token's route table. It exercises all four Run it controls, checks that changing heads preserves the latent/shared key while changing reconstructed content, and verifies exact URL restoration after reload with no page errors.

Three `npm run shoot -- deepseek-v3` rounds produced **36 PNGs per round** (hero and eleven sections, in three views). Selected changed sections were inspected during the first two rounds; **all 36 final PNGs were opened and reviewed** against the skill examples. The first visual review found overlapping toy-router labels despite passing automated gates; the local wider-bar SVG fixed them. The final review confirms consistent typography, spacing and lane colours, contained mobile scrolling, and no observed unintended clipping or text/box/wire overlap. Images are under `runs/shots/deepseek-v3/`.

Final checked and captured artifact: **559195 bytes**, SHA-256 **`b261010f704296234085df4b3c4b7320d78350d09ea95678f6282c6f4eb40c82`**. Scope is the single portable DeepSeek-V3 page at its complete default state, not all combinations of controls.

The parent STRATUM docs and `verify:engineering`, science and runtime scripts are absent in this self-contained subtree; README explicitly assigns this subtree its own check contract. Parent gates are **NOT_RUN**, not reported as passed. No local gate was skipped. Scientific acceptance, reviewer signatures and GPU runtime receipts are not established by this work.

## Current review-rubric results

| Item | Result | Evidence |
|---|---|---|
| 1 | PASS | Exact repository and full checkpoint commit remain in hero and source table. |
| 2 | PASS | Pins, facts, coverage and parameters gates pass; stored-count difference has exact numeric reconciliation. |
| 3 | PASS | Rechecked five detailed ops against pinned source: q_a projection in Attention.forward L769, KV latent norm/expansion L780–784, rotary pair permutation/application L363–370, original-score gather in MoEGate.forward L461, and MLP down projection L388–390; execution order and configured axes/widths agree. |
| 4 | PASS | LayerMap covers every main layer and displays the exact dense/MoE condition. |
| 5 | PASS | Serving cites fused Q/KV A, gate/up and residual handling, compressed state, and expanded fallback. |
| 6 | PASS | Checkpoint and toy quantities used in sections come from facts.ts; numeric literal gate passes. |
| 7a | PASS | Function-per-module clean reference reproduces the pinned publisher's full toy forward, with every parameter perturbed and exact name-set consumption. |
| 7b | PASS | Real-width shape-mode count equals parameterTotal exactly; selected latent, KV, key, value and logits shapes are asserted. |
| 7 | PASS | All levels are present, including the newly added executable Run it section. |
| 8 | PASS, scoped | Detailed eager projections, norms, rotation, mask, softmax, cache effects and FFN/MoE arithmetic remain explicit; fused modes and persistent cache execution have stated scope. |
| 9 | PASS | Detailed ops and added topology inspectors carry purpose, formula, input/output shapes, parameter shapes where applicable and pinned refs. |
| 10 | PASS | Cache expansion/padding, unbiased mixture weights after biased selection and MTP/FP8 accounting require code inspection. |
| 11 | PASS | Selecting dense versus MoE changes the block, FFN widths and explanation. |
| 12 | PASS | Reference-executed routing and reconstructed KV complement existing causal-access, YaRN and serving-size widgets. |
| 13 | PASS, sampled plus focused | Shared play gate passes; focused browser checks exercise every Run it control and exact reload restoration. |
| 14 | PASS | All final light/dark/mobile PNGs opened; the observed toy label overlap was repaired without shrinking text. |
| 15 | PASS | Existing attention/FFN/state/residual lane mapping is preserved; the toy router retains the kit's score/group-selection convention. |
| 16 | PASS | Ledes state learning goals and captions explain what to notice, including toy versus checkpoint scope. |
| 17 | PASS | Shared shell/components and the local router's existing kit tokens preserve one visual design. |
| 18 | PASS | Detailed and topology data-path ops retain width gauges; scalar-size comparisons use TensorScale. |
| 19 | PASS | Every section ends in Takeaways; mechanism sections, including Run it, have reasoning-based hidden-answer Checks. |
| 20 | PASS | Derived query/KV expansion, compressed versus expanded cache widths and active/stored expert parameters explain the mechanisms' benefits. |
| 21 | PASS | Glossary covers the page's model-specific vocabulary and now defines routing, parity, execution and storage terms introduced by the upgrade. |
| 22 | PASS | Exclusions, compatibility helper, toy-only parity, unverified execution and unavailable parent gates are explicit. |

## Final check table

Command: `npm run check -- deepseek-v3`

| Gate | Result |
|---|---|
| pins | PASS |
| typecheck | PASS |
| literals | PASS |
| format | PASS |
| tests | PASS |
| facts | PASS |
| coverage | PASS |
| parameters | PASS |
| build | PASS |
| layout | PASS |
| play | PASS |

**11 pass · 0 fail · 0 skipped**
