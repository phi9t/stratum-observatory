# Qwen3-8B schematic completion report

Exact checkpoint: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`.

The portable page is `dist/qwen3-dense.html`. It covers ten sections: overview, data, the full layer map, selected decoder block, attention, dense gated MLP, executable toy forward pass, parameter accounting, serving/training differences, and sources/glossary. All 36 layers are selectable; all use full causal attention with independent weights. The page has 40 inspectable operations with formulas, shapes, output-width gauges and pinned references. Q/K head normalization, half-split RoPE, grouped-query replication, the cache boundary and both residual additions are explicit.

## Source and verification scope

Seven source records are pinned in `sources.json`: checkpoint config; Transformers reference, configuration and RoPE utility at `5f4ecf2d9f867a1255131d2461d75793c0cf1db2` (v4.51.3); vLLM Qwen3 and inherited Qwen2 implementation at `155488d853a0bc42df227dbfc74005b3fd488e94`; and NeMo AutoModel training implementation at `2a80793f285a1ac41572be5a25f96ea29cacb1f5`. Source-reading observations and implementation decisions are recorded in `NOTES.md`.

The clean TypeScript reference implements unpadded prefill from token IDs to logits. Shape execution at checkpoint dimensions yields **8190735360 parameters**, exactly matching the pinned safetensors metadata. CPU arithmetic is checked against synthetic weights exported by executing the pinned Transformers reference directly, using Transformers 4.51.3 and PyTorch 2.14 CPU. Dependencies needed for this fixture were isolated under ignored `runs/qwen3-python`; no global installation or home-directory modification was made. The fixture records source fingerprints. No trained weights were downloaded.

Nine tests pass. They check source-derived facts and parameter accounting, real-config shapes, upstream toy logits/attention parity, parameter-name agreement, KV-head repetition ordering and nonmutation, half-split rotary coordinates, and the negative condition that changing a future token cannot affect earlier logits. Logits tolerance is 0.0001; attention tolerance is 0.00001. The browser executes this same clean implementation at explicitly labelled toy dimensions with random weights.

Cache reads/writes and their shapes are drawn from source, but cached decoding is **not numerically verified**. Padding, alternative attention backends, bfloat16 kernel rounding, GPU kernels, tensor-parallel execution and packed Transformer Engine training are not executed. vLLM and NeMo comparisons are source-level observations only: these GPU/distributed execution paths are outside the CPU fixture's scope. Logical cache capacity is derived arithmetic, not measured allocation or throughput.

Tokenization, sampling, training loss and optimizer behavior are excluded because the page's boundary is token IDs through language-model logits. Classification and question-answering heads are excluded because this checkpoint ships the causal-language-model head. Framework initialization and typing declarations are documented as exclusions in `coverage.json`; alternate RoPE modes and sliding-window branches are excluded because the pinned config does not activate them. Quantization, checkpoint quality, real-weight behavior, benchmark performance and scientific acceptance are not established by this artifact. There are no skipped schematic gates. The parent Stratum gates are not this subtree's contract, as explained by its README.

## Visual and interaction review

Reviewed the skill's thirteen example images before designing. Screenshot review and fixes were repeated across desktop light, desktop dark and mobile light. Final capture contains 42 PNGs: hero plus ten sections in each view (33), and three attention detail graphs in each view (9). All were opened for visual inspection. Tall attention sections also received detail captures so wire labels and operation text could be inspected at reading scale. Corrections included simplifying crowded block labels, preserving the source's normalization/transpose order, displaying the rotary frequency vector before duplication, and shortening a cache-edge label that touched its box. Tables and wide figures use the kit's internal horizontal scrolling on mobile.

Browser: **Google Chrome 145.0.7632.75**, `/usr/bin/google-chrome`. Final layout checks inspect 1907 text elements across the three views. Play checks identify 97 controls (46 buttons, 40 operations, 5 ranges, 6 radios); all 12 sampled controls change the page, and reload preserves the recorded URL scenario.

Artifact: **504854 bytes**, SHA-256 `4747160ce493dd7c79676d5a390890ef18f99771bee94485c5a2755ed411a8fc`. Captures are generated from that portable HTML by `npm run shoot`; output resides in ignored `runs/shots/qwen3-dense/`.

## Skill self-review

| Rubric | Result and evidence |
|---|---|
| 1. Exact identity | PASS: repository and full checkpoint revision appear on the page. |
| 2. Fidelity gates | PASS: pins, facts, coverage and parameters pass without a parameter exception. |
| 3. Five sampled operations | PASS: checked Q projection and Q head RMSNorm against `Qwen3Attention.forward` lines 205–253; half-split rotary against `apply_rotary_pos_emb` lines 105–129 and `rotate_half` lines 98–102; scaled QK product against `eager_attention_forward` lines 144–167; SiLU gate times up projection against `Qwen3MLP.forward` lines 93–95, all at the pinned Transformers revision. Shapes and ordering agree. |
| 4. Layer kinds | PASS: config disables sliding windows; every absolute layer tile shows the same full-attention dense block. |
| 5. Serving comparison | PASS: cited packed QKV/gate-up projections, deferred residual addition and KV partition/replication distinguish vLLM execution. |
| 6. Number provenance | PASS: literals and facts gates pass; prose and graph quantities consume `facts.ts`. |
| 7a. Clean reference | PASS: own checkpoint-specific implementation and pinned upstream CPU parity test pass. |
| 7b. Shape ledger | PASS: real-config shape mode agrees with the exact parameter total. |
| 7. Detail levels | PASS: model, layer, block, operation, numerical run, weights and implementation comparison levels are present. |
| 8. Operation coverage | PASS: attention projection/norm/position/mask/cache/mix and gate/up/activation/product/down paths are explicit. |
| 9. Inspectors | PASS: each operation includes explanation, formula, input/output shapes and source references. |
| 10. Code observations | PASS: Q/K gain vectors are shared across heads; rotated K and raw V enter cache before KV repetition. |
| 11. Layer selection | PASS: layer tiles change selected block identity and explanatory sentence. |
| 12. Mechanism widgets | PASS: causal-grid selection, rotary-position dial, runnable attention and cache budget use the implementation's rules. |
| 13. Controls and URL | PASS: browser play probe changes all sampled controls and restores URL state after reload. |
| 14. Screenshots | PASS: light, dark and mobile reviewed through repeated capture/fix rounds; detailed graphs inspected for label collisions. |
| 15. Lane colours | PASS: shared kit semantic lanes consistently distinguish residual, attention, position and MLP flow. |
| 16. Ledes/captions | PASS: sections state their teaching point and figures explain what to observe. |
| 17. Consistent design | PASS: shared kit typography, spacing, cards, tables and diagram grammar throughout. |
| 18. Width gauges | PASS: data-path operations specify output widths, including head and rotary frequency dimensions. |
| 19. Teaching checks | PASS: each section ends with Takeaways; mechanism sections include reasoning checks. |
| 20. Mechanism benefits | PASS: derived head sharing, positional pair counts, FFN expansion and fourfold logical cache saving explain consequences. |
| 21. Glossary | PASS: source section defines introduced architecture and serving terminology. |
| 22. Honesty | PASS: excluded features, unexecuted backends and numerical scope are listed above; no skipped gate is labelled passed. |

## Final check

Command: `npm run check -- qwen3-dense`. The table below is copied from the generated check report, not inferred from the page.

| Gate | Status | Evidence |
|---|---|---|
| pins | PASS |  |
| typecheck | PASS |  |
| literals | PASS |  |
| format | PASS |  |
| tests | PASS | Tests  9 passed (9); includes the clean reference's parity test |
| facts | PASS |  |
| coverage | PASS |  |
| parameters | PASS | parameterTotal 8190735360 equals safetensors.total |
| build | PASS | dist/qwen3-dense.html 504854 bytes sha256 4747160ce493dd7c79676d5a390890ef18f99771bee94485c5a2755ed411a8fc |
| layout | PASS | desktop-light, desktop-dark, mobile-light: 1907 text elements checked |
| play | PASS | 97 controls (46 button, 40 op, 5 range, 6 radio); 12 of 12 sampled changed the page; reload kept #q=12&runLayer=1&ropePos=20480&runHead=2&runQ=4 |

**11 PASS · 0 FAIL · 0 SKIPPED.**
