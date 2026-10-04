# DeepSeek-V4.1-Flash schematic report

Completed for `deepseek-ai/DeepSeek-V4.1-Flash`, checkpoint revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`. The portable page is `dist/deepseek-v4.1-flash.html`. Implementation decisions and source-forward notes are in [NOTES.md](NOTES.md); source URLs, commits and selectors are in [sources.json](sources.json) and [symbols.json](symbols.json).

## Delivered scope

The page contains thirteen sections: Overview, Data, Stack, Block, Attention, Compression, Experts, Engram, Auxiliary, Run, Weights, Serving and Sources. It includes the full layer map, selectable block variants, operation inspectors with formulas/shapes/source references, width gauges, mechanism controls, a browser-executed clean reference, a logical parameter ledger and a serving comparison. Checkpoint quantities come from pinned config/code facts or explicit derivations in `facts.ts`; reduced teaching dimensions are labelled separately.

Twelve public source files are pinned. Publisher inference defines the architecture. Independent serving is SGLang revision `bab04cd7913ff50d1bf1a677a2d10e5015dc4c3b`. The torchtitan revision `e3929368edd2f2457456913890ba2b4c0f27da31` is a family comparison: its same-sublayer pre coefficients and learned HC head do not establish exact V4.1 training support.

The reference executes text prefill through full logits, retaining all 40 backbone layers and the actual KV/index/candidate ownership boundaries. Toy runs use random weights and supplied synthetic lookup IDs. They are teaching fixtures, not checkpoint outputs.

## Verification and artifact identity

Current portable artifact: **564491 bytes**, SHA256 `6b8814368ac5b497d62de3ecf3b0598448e12a29ac8a432e070fb6ebf83295fd`.

`npm run check -- deepseek-v4.1-flash`:

| Gate | Result |
| --- | --- |
| pins | PASS |
| typecheck | PASS |
| literals | PASS |
| format | PASS |
| tests | PASS: 11 passed, 1 explicitly skipped parity test |
| facts | PASS |
| coverage | PASS |
| parameters | PASS with exact auxiliary reconciliation |
| build | PASS |
| layout | PASS: 3638 text elements across three viewports |
| play | PASS: 181 controls; all 12 sampled controls changed the page; URL restored |

Gate totals: **11 pass, 0 fail, 0 skipped**. The gate totals do not turn the skipped numerical parity test into a pass. Generated check receipt: `runs/check/deepseek-v4.1-flash/report.json`.

An additional model-local sweep, `reference/browser-sweep.mjs`, selected all **40 layers**, opened all **109 operation inspectors**, changed all **11 range controls**, and verified operation selections, layer and range values after URL reload. It passed against the artifact digest above. Chrome: **Google Chrome 145.0.7632.75**, executable `/usr/bin/google-chrome`.

`npm run shoot -- deepseek-v4.1-flash` generated **42 PNGs**: hero and thirteen sections in desktop light, desktop dark and mobile light. Every PNG was opened and visually reviewed. Desktop viewport is 1440 × 1000 at DPR 1; mobile is 390 × 844 at DPR 2. Screenshots are under `runs/shots/deepseek-v4.1-flash/`. Wide diagrams/tables intentionally scroll horizontally on mobile, preserving type size; screenshots show the initial scroll position, not every horizontal extent. There are no overlapping labels in the reviewed captures. The kit was not changed.

Tests cover source/fact/coverage bindings, independent parameter arithmetic, bias-only expert selection, routing weights from unbiased scores, mandatory newest candidate admission, Sinkhorn normalization, YaRN behavior, full reference shapes, finite outputs, window/completed-group causality, denominator-only sinks and refusal of missing lookup IDs. A failing behavioral routing test preceded its implementation.

## Parameter reconciliation

The independent text ledger and real-config shape-mode forward both request **748494684784** logical scalar weights. Draft-only parameters are **14225362530**; vision/aligner/delimiter parameters are **485268480**. Their sum is **763205315794**, exactly the hub total. Main embedding/head shared by draft stages are counted once. The difference from text scope is **14710631010**, explicitly stated on the page and in `parameterNote`.

This is logical weight accounting, not packed storage bytes, scale metadata, active compute or peak memory.

## Numerical parity and exclusions

**Upstream numerical parity is UNVERIFIED.** Running `reference/make_fixture.py` with `/data03/yph/venv/bin/python` exited 2 with `Numerical parity UNVERIFIED: TileLang, CUDA GPU`. The available PyTorch is CPU-only (`2.14.0+cpu`); TileLang and a CUDA GPU are unavailable. Publisher inference imports GPU kernels. No replacement upstream fixture, fake tolerance result or package installation was used. The dedicated upstream parity test is explicitly skipped.

The clean executable omits FP8/FP4 rounding, tokenizer normalization and rolling int64 hashing, distributed collectives, decode ring/paged cache execution, sampling, vision and dSpark execution. Supplied lookup IDs preserve the Engram arithmetic boundary without pretending JavaScript numbers implement the publisher's int64 hashes. Vision and draft paths are structurally drawn and cited, with their executable scope stated. The browser does not establish trained-weight behavior, GPU performance, packed cache sizes or serving precision parity. SGLang fusion/rounding differences are described from code, not benchmarked. No model weights were downloaded.

The page groups some arithmetic into named compound operations (for example score-plus-scale, activation products and auxiliary block summaries); it is not an elementary instruction-by-instruction trace. Every main attention projection, norm, positional boundary, mask and cache interaction is represented, but these compound expressions remain a deliberate readability scope decision. Auxiliary modules are structural explanations rather than executable mechanism widgets. Foundational formula terms such as RMSNorm/SiLU/GELU appear in inspectors; the glossary emphasizes architecture-specific vocabulary rather than being a full mathematical textbook.

The Stratum root engineering/science/runtime commands from the supplied general contract are not entry points of this standalone schematic subtree. They were not run. The applicable local skill gate is the check table above; no scientific acceptance or runtime parity PASS is claimed.

## Source spot-check

A deterministic random sample (`Random(41)`) selected five attention/FFN operations. Each was checked against the pinned publisher source, not against the browser reference:

| Operation | Source and evidence |
| --- | --- |
| Router `topk` | `model.py`, Gate.forward L809–827: selection bias applies before top-k; original scores are gathered at L821–824; selected shape is `[B*S, 6]`. |
| Router `score` | Gate.forward L816: sqrt(softplus) follows the fp32 projection; scores are `[B*S, 384]`. |
| Window `kvquant` | Attention._window_kv L700–720: whole-vector fake FP8 follows tail RoPE at L707; shared KV is `[B, S, 512]`. |
| Window `kvnorm` | Attention._window_kv L705: normalization follows KV projection and precedes tail rotation. |
| Expert `down` | Expert.forward L841–851: clipping, SiLU product and routing weight precede w2; output is `[T, 5120]`. |

## Skill review rubric

| Line | Score and evidence |
| --- | --- |
| 1 | PASS: exact repository and full pinned revision are visible in Sources. |
| 2 | PASS: pins/facts/coverage pass; parameter difference has an exact numeric reconciliation. |
| 3 | PASS: five randomly selected operations were reopened in pinned code, as recorded above. |
| 4 | PASS: all backbone layer kinds and source/consumer transitions derive from the actual config arrays. |
| 5 | PASS: Serving cites projection fusion, fused norm/RoPE/cache writes and delayed-pre fusion exclusions. |
| 6 | PASS: literal/facts gates pass; architectural quantities and teaching dimensions are centrally sourced. |
| 7a | PASS within permitted unverified scope: clean source exists; the GPU/TileLang blocker and skipped parity test are explicitly recorded. |
| 7b | PASS: real-config shape mode equals the independent text ledger. |
| 7 | PASS: checkpoint overview, data path, stack, block, operation inspection, numerical run, weights and serving levels are present. |
| 8 | FAIL, stated scope: some arithmetic expressions and auxiliary blocks remain compound nodes rather than one node per elementary operator. Main projection/norm/position/mask/cache boundaries are explicit. |
| 9 | PASS: all 109 inspectors expose meaningful descriptions, formulas, shapes and pinned references; no placeholder inspector remains. |
| 10 | PASS: delayed pre-coefficient ownership, unrotated index keys, signed index-head weights and denominator-only sinks come from code rather than config alone. |
| 11 | PASS: the full browser sweep changes all 40 layer selections; the layer-specific block and ownership sentence update. |
| 12 | FAIL, stated scope: text mechanisms have executing widgets, but quantization, hashing, vision and draft execution are excluded for the reasons above. |
| 13 | PASS: all operation/layer/range controls were exercised and their URL state restored; the standard gate also sampled navigation controls. |
| 14 | PASS within scrolling design: all 42 captures reviewed; no overlaps; wide mobile diagrams/tables retain readable size with horizontal scrolling. |
| 15 | PASS: attention blue, FFN orange, state violet and residual teal are consistently explained and applied. |
| 16 | PASS: section ledes state their teaching point; widget captions distinguish observations from checkpoint claims. |
| 17 | PASS: shared kit typography/cards/inspectors give one design; model-local chart spacing fixes preserve it. |
| 18 | PASS: graph data-path operations carry width metadata for the gauges. |
| 19 | PASS: sections end with Takeaways; mechanism checks ask about causal consequences rather than recalling config values. |
| 20 | PASS within structural auxiliary scope: main mechanism benefits use derived dimensions, active fractions, compression ratios or ownership counts. |
| 21 | FAIL, stated scope: architecture-specific glossary is supplied, but foundational mathematical operators are explained in formulas rather than exhaustively defined in the glossary. |
| 22 | PASS: this report lists exclusions, unverified parity, skipped test and non-applicable root gates. |

Only files under `models/deepseek-v4.1-flash/` were authored; tool outputs are in the allowed ignored directories. No commits, branches, pushes, global installations or changes to shared tools/kit were made.
