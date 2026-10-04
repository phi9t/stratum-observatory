# Kimi K2 schematic report

Completed for `moonshotai/Kimi-K2-Instruct` at `fd1984e2b7a3350dbf7305fe73a4ede25c14de50`. The independent vLLM sources are pinned at `155488d853a0bc42df227dbfc74005b3fd488e94`. Sources, symbols, reading notes and coverage decisions are local to this model directory. No kit, tools, skill or README changes were made; no commit or branch was created.

## What is drawn

Ten sections descend from token ids and the full stack to detailed operations, then execute a toy decoder. All 61 zero-based layers are selectable: layer 0 has the dense FFN, layers 1–60 have routed plus shared experts. The block and FFN change with that selection. Attention separates query compression, query norm and expansion, latent/key split, latent norm and expansion, head reshapes and splits, YaRN frequencies, query/key rotations and joins, cache update/read, comparison, score scaling, causal mask, softmax, value mixing, head merge and output projection. Expert graphs separate score production, corrected selection, original-score gathering, normalization/scaling, token dispatch/restore/reduction, and the gated FFN. Shared experts remain outside router logits.

Playable figures use the clean reference: a checkpoint-width random router, actual YaRN frequency dials, causal visibility, and a complete random-weight toy decoder with computed attention, latents and routing. The cache calculator compares publisher eager expanded storage with vLLM MLA latent storage, using explicit BF16 cache arithmetic. It explains projection, activation, residual and expert fusion rather than claiming measured performance.

## Numerical evidence and accounting

`reference/make_fixture.py` executes unchanged AST-selected publisher module definitions from the pinned cache, using the existing CPU Python environment. This avoids compatibility glue for the old publisher file without rewriting its arithmetic. The fixture has four layers (two dense, two MoE), two batch rows, five tokens, and perturbed finite weights including correction biases. This deliberately exercises repeated block kinds; it is not the checkpoint's real layer boundary. TypeScript tests compare logits, every layer's residual and attention tensors, normalized latents, selected router sets and aligned mixture weights. Logit/intermediate tolerance is `1e-4`; router-weight tolerance is `1e-5`. Every parameter name must match, and the fixture binds the publisher source hash.

Shape-only execution at the real configuration counts `1026408232448` learned parameters. It registers the full expert bank without allocating real weights. The pinned Hub total is `1026408235864`, a difference of `3416`. Metadata-only HTTP range reads of the dense and first expert shard headers show saved rotary buffers of width `56`; repeating this over the `61` indexed buffers reconciles the difference. **The remaining 59 buffer shapes were not individually inspected.** Initial broader header requests timed out at the CDN; the supported conclusion is the stated inference, not a full shard audit. The runtime rebuilds nonpersistent rotary frequencies at its configured width. The pinned index and independently checked block grids imply `62495192` FP8 inverse-scale scalars outside the Hub total. Index-name coverage and the two observed header shape/scale grids are tested. No tensor payload or checkpoint weight was downloaded. `storage-evidence.json` preserves header metadata and hashes; `reference/read_storage.py` can regenerate it and the summary, preserving existing evidence on incomplete fetches.

A negative cache test initially rejected the implementation's acceptance of invalid batch/context sizes; the implementation now validates positive integers and the configured context bound. Router negative tests check that correction bias changes selection but never directly enters mixture weights. YaRN tests distinguish preserved high frequencies from stretched low frequencies.

## Exclusions and unverified items

- Tokenization, chat templates, sampling, generation helpers and training loss are outside the forward-architecture page.
- No exact public training forward was established. The publisher gate explicitly requires evaluation. The public MoonshotAI/Kimi-K2 repository tree was searched at `1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4`; no exact checkpoint training implementation was found. A generic training recipe was not substituted.
- Clean numerical parity is float32, eager, unpadded full-sequence CPU arithmetic at toy sizes. It does not verify trained checkpoint outputs, FP8 kernels, GPU parity, throughput, distributed collectives, speculative decoding or incremental-cache numerical parity.
- Serving behavior is source-reviewed, not executed. vLLM configuration translation and backend availability were not tested on hardware. The optional publisher FlashAttention path pads values before caching; the expanded-cache calculator specifically covers the eager path. FP16 overflow scaling and optional shared-expert kernel fusion are explained as backend/mode differences, not simulated.
- Unused rotary alternatives and sequence classification are excluded in `coverage.json`; the checkpoint has no sparse indexer or next-token prediction layers.
- No local check gate was skipped. Root Stratum engineering/science/runtime gates were **NOT_RUN**: this self-contained package does not expose those commands and its README distinguishes its gates from the root application. No scientific acceptance or GPU runtime PASS is claimed.

## Visual and interaction review

Three capture rounds were made. All 13 skill example images were opened before design. All 33 final PNGs were opened after the final `npm run shoot -- kimi-k2`, covering the hero and all ten sections in each of desktop light, desktop dark and mobile light. Earlier reviews led to a wider custom toy-router figure, sum/multiply symbols, shorter mobile inspector parameter labels, and separate compact router graphs. Final diagrams have orthogonal edges, retained residual paths and readable inspector text. Wide figures/tables deliberately use the kit's internal mobile scrolling; screenshots show the initial scroll position rather than the whole horizontal diagram.

Browser: Google Chrome `145.0.7632.75`, `/usr/bin/google-chrome`. Desktop viewport `1440 × 1000`, scale 1; mobile `390 × 844`, scale 2. Capture scope is default state in these three views, not an exhaustive screenshot sweep of every interaction. The check measured 2444 text elements and counted 144 controls (74 buttons, 61 ops, seven ranges, two radios); its 12 sampled interactions all changed the page, and reload preserved the tested URL state. Eleven tests passed.

Final artifact: `dist/kimi-k2.html`, 626423 bytes, SHA-256 `5495edce2aed41281f75a535f36d95d78e334d59c2c40cee36574272b2740da0`. Final screenshots are `runs/shots/kimi-k2/*.png`. Local generated gate report is `runs/check/kimi-k2/report.json`.

## Rubric self-review

| Line | Result | Evidence |
|---|---|---|
| 1 Identity | PASS | Exact repository and full revision appear in Sources. |
| 2 Gates | PASS | Pins, facts, coverage and numeric parameter reconciliation pass. |
| 3 Five ops | PASS | Five sampled source checks are recorded below. |
| 4 Layer kinds | PASS | All tiles derive from the actual dense/MoE condition. |
| 5 Serving differences | PASS | Fused projections, residual norm and latent decode have pinned refs. |
| 6 Numbers | PASS | Config, code constants and derived quantities are in facts.ts; literals gate passes. |
| 7a Reference parity | PASS within scope | Clean eager reference matches the upstream toy fixture; exclusions above. |
| 7b Real shape count | PASS | Shape execution equals the learned-parameter ledger. |
| 7 Detail levels | PASS | Whole model, representation, stack, block, sublayers, ops, run, weights, serving and provenance present. |
| 8 Fine operations | PASS | Attention and router are decomposed into separate logical operations. |
| 9 Inspectors | PASS | Graph helper supplies description, formula, shape, refs and learned parameter shapes where applicable. |
| 10 Beyond config | PASS | Corrected choice versus original weighting; rotary channel shuffle and scale; serving residual fusion. |
| 11 Layer selection | PASS | Shared URL layer selection changes block, FFN and explanatory sentence. |
| 12 Mechanism widgets | PASS | Actual router, YaRN, causal rule and computed decoder figures run locally. |
| 13 Controls | PASS within scope | Automated play samples pass with URL restoration; not a proof of every possible state combination. |
| 14 Screenshots | PASS | All final light/dark/mobile PNGs inspected; mobile scrolling is intentional. |
| 15 Colours | PASS | Ink stream, blue attention, orange FFN, violet position/router, teal bypass, yellow selection. |
| 16 Captions | PASS | Ledes state a point; captions explain comparison, gauges and execution boundaries. |
| 17 Design | PASS | Shared kit typography, panels and orthogonal graph vocabulary throughout. |
| 18 Width gauges | PASS | Each graph op has output width; sums preserve the residual width. |
| 19 Study aids | PASS | Every section has Takeaways; mechanism sections include reasoning Checks. |
| 20 Mechanism value | PASS | Derived cache ratio and routed-bank fraction quantify representation and sparse execution. |
| 21 Glossary | PASS | Sources defines architectural, numerical and serving terms introduced by the page. |
| 22 Honesty | PASS | Exclusions, inspection limitations and root-gate non-execution are explicit above. |

Five sampled ops were reopened in pinned `modeling_deepseek.py` and checked against the graph and config (source refs resolve to containing symbols): query down-projection at line 770 gives `[B,S,1536]`; latent RMSNorm/kv expansion at lines 781–789 gives `[B,64,S,256]` before content/value split; positional rotation at line 801 rotates only the positional slices with a head-shared key; original score gathering at lines 458–462 gives `[B×S,8]` after corrected top-k choice; SiLU(gate) times up at lines 388–389 preserves the chosen dense/expert inner width before down-projection. Each exists in the shown execution order. vLLM's corresponding containing symbols are linked in the inspectors, with fusion differences explained on the page.

## Final check table

Generated from the current local gate report, not from a task description.

| Gate | Status |
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

**11 pass, 0 fail, 0 skipped.**
