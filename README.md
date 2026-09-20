# STRATUM — Model Systems Observatory

**Nine exact model variants. Seven connected lenses. One offline scientific interface.**

Open `dist/index.html` in a full browser for the [landing directory](https://phi9t.github.io/stratum-pages/). [Modeling Foundations](https://phi9t.github.io/stratum-pages/foundations.html) covers DeepSeek V2/V3 and Qwen3. Dedicated pages cover [Qwen3.8](https://phi9t.github.io/stratum-pages/qwen3.8.html), [DeepSeek V4.1 Flash](https://phi9t.github.io/stratum-pages/deepseek-v4.1.html), and [Kimi K3](https://phi9t.github.io/stratum-pages/kimi-k3.html). Shared sections cover [GPU Hardware](https://phi9t.github.io/stratum-pages/hardware.html), [Networking](https://phi9t.github.io/stratum-pages/networking.html), and [Data Representation](https://phi9t.github.io/stratum-pages/data-representation.html). The data chapter is a general reference, with source acceptance and independent review still pending. Home returns to the directory. Model tabs are Architecture, Training, Serving, and Evidence; Networking has an explicit workload-model selector. All eight HTML files are portable and have no external runtime assets, weights, or telemetry; source links require connectivity. Run `npm start` to build and serve `dist/` at `http://localhost:8000`. Existing scenario links to `index.html#model=…` route to the appropriate interactive page.

## Experience

Choose a checkpoint, choose a lens, and inspect an operation. Every model keeps the same visual language while its actual motif changes. Parameter, cache, memory and traffic accounting are derived from shared model definitions, not duplicated display totals.

| Variant | Implemented architectural distinctions |
|---|---|
| DeepSeek V2 | 60 layers; MLA; first layer dense; 160 routed experts, top-6, two shared experts |
| DeepSeek V3 | 61 layers; MLA; first three layers dense; 256 routed experts, top-8, one shared expert; selection/mixture distinction |
| Qwen3 8B | Dense GQA, query/key normalization, 36 layers |
| Qwen3 30B-A3B | 48 layers; GQA; 128 experts, top-8; no shared expert |
| Qwen3 235B-A22B | 94 layers; GQA; query projection width differs from residual width; 128 experts, top-8 |
| Qwen3.8 27B | 64 layers; GatedDeltaNet/full-attention hybrid; dense FFN; vision input |
| Qwen3.8 Flash-Next | 48 layers; GatedDeltaNet/QSA; latent index micro-block selection; 512 experts/top-10 plus shared; four-stream gated residual; layer-2 n-gram memory; vision input |
| DeepSeek V4.1 Flash | Dedicated causal encoder/decoder page; shared compressed KV, sliding-window state, mHC, Engram and draft-head context |
| Kimi K3 | 93 layers; vector-decay KDA/gated NoPE MLA; latent MoE; depth attention residuals; vision input |

The source inspector attaches publisher configuration, publisher/reference implementation, independent serving implementation and document references. Exact checkpoint IDs and commit pins are in `src/models.mjs` and `evidence/reference-registry.json`. These are **reference mappings, not accepted raw-byte evidence or signed scientific approvals**.

### Model lenses and shared sections

**Architecture:** whole-model context, every one of the 544 decoder layers, stable attention/FFN regions, tensor contracts, delayed state feedback and optional input/residual/PLE close-ups.

**Training:** FSDP, replicated DDP and an illustrative agentic-RL lifecycle. Adam persistent-state accounting and dependencies are educational scenarios, not reconstructions of a publisher's production run.

**Serving:** token history versus recurrent matrices, convolution history, indexer state, weight replicas and prefill/decode ownership. Hybrid prefix state must restore at a compatible boundary. Precision controls are arithmetic assumptions, not validated quantization recipes.

**Shared GPU Hardware:** logical H100/H200/B200 eight-GPU systems and GB300 NVL72; per-GPU versus rack identity; NVLink versus scale-out interfaces; dense-FP8 roofline ceilings. The GB300 drawing groups GPUs visually, not into claimed physical trays.

**Shared Networking:** InfiniBand or RoCE; expert dispatch, ring all-reduce or cache handoff; NCCL, NVSHMEM or NIXL responsibilities; registration, progress and completion. Choosing a library describes a scenario and obligations, not certification that every combination is supported on a particular stack.

**Evidence:** publisher → reference implementation → independent framework → consumers → review/runtime boundary. Engineering, publication and runtime statuses are separate.

### Controls

Click an operator to inspect its equation, dimensions, sources and related lenses. Trace dependencies, select a layer in the grid, change context/batch/precision/placement/network assumptions, switch graphite/paper themes, zoom, or focus the canvas. Export the current SVG and complete scenario JSON.

Keyboard: `1`–`7` choose lens; `/` or `Ctrl/Cmd+K` search; `F` focuses the canvas; `0` resets zoom; `Escape` closes focus/trace/dialogs. Operator and layer selection are keyboard accessible. The URL fragment persists the scenario. Mobile preserves a legible schematic width in a scrollable viewport instead of shrinking or wrapping labels.

## Scientific scope

A source link, formula test or clean diagram is not a scientific certificate. **Publication remains BLOCKED and runtime experiments remain NOT_RUN.** `VERIFICATION.md` records the fresh engineering result; `verification/reports/publication.json` records missing evidence and reviews.

Exact decoder ledgers for DeepSeek and Qwen3 count the declared model structure, untied vocabulary and norms, not checkpoint storage bytes. MTP and vision exclusions are explicit. Kimi and Qwen3.8 use publisher-rounded parameter scope where a complete tensor manifest has not been reconciled. Flash-Next's 125B backbone and 51B n-gram memory are counted separately from 4B MTP; uniform-weight precision is deliberately not a claim about the released mixed-precision checkpoint.

Cache results are logical subtotals. They exclude page rounding, replication, allocator reserve, retained prefix snapshots, output-token reserve and runtime workspace. Flash-Next PLE convolution/history, incomplete index-block state and other auxiliary caches remain explicitly excluded rather than silently assigned zero. Sparse reads do not imply that full token KV storage disappears. Training bars are persistent averages, not peak-memory fit guarantees. Transport is a payload/rate bound, not measured latency, TTFT or throughput. GB/TB are decimal; GiB/TiB are binary. NVLink is labeled bidirectional aggregate per GPU; NIC rates are labeled one-way Gb/s.

## Develop and verify

Node.js 20+ and Python 3 are required for tests/build. The verifier uses Python's standard library and OpenSSL for signed review tests. Browser tests use Python Playwright and Chromium; set `CHROMIUM_PATH` when needed. The CI workflow pins Playwright 1.55.0. There are no runtime npm dependencies.

```sh
npm test                    # Model math, negative cases, semantic graph contracts
npm run build               # Deterministic portable HTML
npm run test:verifiers      # Evidence, signatures, contracts and negative tests
npm run verify:schematics   # What the diagram means, over all 544 decoder layers
npm run verify:aesthetics   # Measured legibility, type/spacing scales, composition drift
npm run verify:presentations # Exports, accessibility, print, portability, preview coverage
npm run verify:accounting   # Independent book-method arithmetic; source acceptance stays separate
npm run verify:engineering  # Verifier/Node/accounting checks, two identical builds, browser tests
npm run verify:science      # Expected exit 2 until evidence/reviews are completed
npm run verify:runtime      # Expected exit 2 until actual experiments are attested
npm run capture             # Fresh PNG and SVG previews of the current build
npm run verify              # engineering, schematics, aesthetics, presentations AND publication; fail-closed
```

`0` means PASS, `1` means a contradiction/malformed input, `2` means BLOCKED or required work NOT_RUN. GitHub Pages deploys on the four engineering-class gates (engineering, schematics, aesthetics, presentations) and a `package` job; publication still runs on every push and still reports BLOCKED, so the page states its own scientific status rather than being withheld. This archive does not configure a remote repository or bypass a blocked gate.

## Source layout

```text
src/models.mjs          Checkpoint identities, fields, hardware profiles, source links
src/core.mjs            Scenario validation and pure accounting functions
src/scene.mjs           Visual IR and structural geometry invariants
src/diagram-model.mjs   Layer-specific attention, dense/MoE and residual motifs
src/diagram-systems.mjs Training, serving, GPU and network templates
src/diagram-meta.mjs    Compare, evidence, vision/residual/PLE focus templates
src/diagrams.mjs        Scene dispatcher
src/app.mjs             Browser rendering, inspector, trace, state, audit and exports
src/styles.css         Graphite/paper tokens, responsive layout and typography
build.mjs              Portable bundler; does not fetch external assets
verification/          Fail-closed provenance, contract, review and runtime checks
tools/export-verification.mjs  Actual core/function/scene inventory adapter
docs/design/           First-principles design and implementation record
docs/verifiers/        Domain requirements and coding-agent evidence protocol
archive/kimi-baseline/ Historical source for reference only, not active verification
```

This is a deterministic **template-based architecture schematic system**, not an arbitrary-model importer, unrestricted graph auto-layout engine, model execution runtime, cluster simulator or benchmark suite. The original visual inspiration is Edward Z. Yang's DeepSeek diagram. This implementation uses newly authored layouts and application code; no upstream source-site assets or font files are bundled.


The analysis panels on every portable page include expandable **Accounting assumptions**.
They explain the selected precision, ownership and units using current ledger values.
The [book audit design](docs/design/BOOK-ACCOUNTING.md) and [source record](docs/design/BOOK-SOURCES.md)
distinguish methods from checkpoint evidence. `verify:accounting` emits JSON and Markdown under
`verification/reports/` and is required by engineering. Its arithmetic agreement cannot replace
publication review or runtime measurements; specialized architectures carry explicit exclusions.

## Publishing a clean source snapshot

Run `npm run verify:privacy` and `npm run package:public`. The latter writes
`.publication/stratum-public.tar.gz` from current tracked and non-ignored files, excluding
Git history and local caches. Extract it into a new directory and initialize a new repository
with your chosen public commit identity. The pre-scrub history contains personal author
metadata and private hardware captures. Publish only audited, rewritten branches; never
restore or push the private recovery bundle or old local worktree refs. The public runtime
receipt manifest is empty; withdrawn hardware evidence is not replaced with fabricated values.
See [publication privacy](docs/PUBLICATION.md) for scope and limitations.
