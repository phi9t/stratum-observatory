# STRATUM coding-agent contract

Read `README.md`, `docs/design/OBSERVATORY.md`, `docs/AGENT_VERIFIER.md`, and the relevant `docs/verifiers/` contract before changing code.

## Boundaries
- Data/source identities: `src/models.mjs`. Math: `src/core.mjs`. No DOM in either.
- Visual IR: `src/scene.mjs`. Model motifs: `src/diagram-model.mjs`. Systems: `src/diagram-systems.mjs`. Comparison/evidence/focal views: `src/diagram-meta.mjs`.
- UI/state/export: `src/app.mjs`; shared style tokens: `src/styles.css`.
- `src/diagrams.mjs` dispatches templates. `build.mjs` is dependency-free and deterministic.
- Written exposition: `src/innovations.mjs`. Every record's `effect` is a **function** of the model, never a string, so no architectural number can be retyped into prose. `replaces`/`mechanism` are the only authored sentences and carry **no quantity at all** — not a digit, and not a cardinal spelled as a word; "the bottom half of the stack" asserts `encoderLayers === layers / 2` where nothing can check it, and a test rejects it. A quantity belongs in `effect`. Every record cites `SOURCES` ids that must resolve.
- Current scientific target is `verification/manifests/observatory.json`; `kimi-k3.json` is historical, not the current consumer inventory.

## Non-negotiable
Preserve exact variant identity, shape contracts, precision/rate units, ownership and time/depth axes. No branch invented to fit a template. Shared experts do not consume router logits. Do not remove meaningful edges to pass geometry tests. Expand lanes/boxes, not font degradation. Do not turn missing numeric evidence into zero.

Every changed consumer needs its exact binding and upstream evidence chain. A new fingerprint is not an approval. Do not overwrite accepted source bytes, reviewed subjects, runtime receipts or trust roots to silence drift. New variants must appear in the actual exporter and browser layer matrix.

`VERIFICATION.md` is the public record and must agree with the reports in `verification/reports/`; `verification/tests/test_reported_numbers.py` fails when it drifts. Every figure written there must come from a report actually generated on the current bytes — never copied from a task description, a commit message or a previous run.

Write a failing behavioral/negative test, implement locally, run `npm run verify:engineering`. Run scientific/runtime gates separately and report BLOCKED/NOT_RUN faithfully. `npm run verify` remains fail-closed on all five of its gates. **Pages deploys on the four software gates** (engineering, schematics, aesthetics, presentations) and no longer requires publication -- an amended contract, recorded in the workflow and in `VERIFICATION.md`, because publication is BLOCKED until an independent reviewer signs and the page could otherwise never ship. **Publication completeness is offline-only** -- `npm run verify:science` on a trusted machine with the source cache, exit 2 BLOCKED, report committed to `verification/reports/publication.json`. It cannot run in CI (the cache is git-ignored because the licences forbid vendoring; no reviewer identity exists) and a job that can never pass is noise, so there is no `publication` job. `test_reported_numbers.py`, inside the green engineering gate, pins the record to that report and the report to the current lock and consumer digests, so offline cannot become stale. CI runs `publication-contradictions` on every push: the full audit on CI's bytes, failing on exit 1, any FAIL finding, or a missing/empty report, accepting exit 2. It never gates the deploy. No `continue-on-error`, fake raw hashes, invented reviewer keys, copied signatures or benchmark claims from payload/rate calculations.

Previews must come from the current portable HTML via `npm run capture`. Include actual counts, browser, artifact identity and scope in completion reports. No weights, credentials, telemetry, bundled fonts or external runtime scripts.

## Commands

```sh
npm test                              # Node model math + scene contract tests (tests/*.test.mjs)
node --test tests/core.test.mjs       # One test file
node --test --test-name-pattern 'QSA' tests/core.test.mjs   # One test by name
npm run build                         # index.html + src/ -> dist/index.html (deterministic)
npm start                             # Build and serve all portable pages at localhost:8000
npm run test:verifiers                # uv run python -m unittest discover -s verification/tests -v
uv run python -m unittest verification.tests.test_evidence -v   # One verifier module
uv run python tests/browser.test.py   # Chromium suite; CHROMIUM_PATH optional, Playwright resolves it
npm run capture                       # Regenerate previews/ PNG+SVG from the current build
npm run verify:schematics             # Diagram semantics vs core.mjs over all 544 layers; no browser
npm run verify:presentations          # Exports, accessibility, print, portability, preview freshness
npm run verify:aesthetics             # Measured legibility, type/spacing scales, composition, drift
npm run verify:aesthetics -- --record  # Write a candidate composition baseline for review
npm run verify:accounting             # Independent book-method arithmetic, separate source acceptance
npm run verify:engineering            # Verifier tests + node tests + two builds + browser; writes receipt
npm run verify:science                # Publication gate; exits 2 while BLOCKED
npm run verify:runtime                # Runtime gate; exits 2 while experiments are NOT_RUN
npm run verify                        # engineering, schematics, aesthetics, presentations AND publication, fail-closed
```

Exit codes: `0` PASS, `1` contradiction/malformed input, `2` BLOCKED or NOT_RUN.

Python is uv-managed: `uv sync --frozen` provisions CPython 3.13.12 and Playwright 1.55.0 from
`uv.lock`, then `uv run --frozen playwright install chromium`. Never `pip install` into an ambient
interpreter. Node 20+ for the application; the app itself has no runtime dependencies.

Review signatures need an OpenSSL that implements Ed25519. macOS resolves `openssl` to LibreSSL,
which does not; `evidence.ed25519_openssl()` finds a capable binary or the gate reports
`REVIEW_TOOLCHAIN` BLOCKED rather than failing an honest reviewer.

## Architecture

**Application** — zero-dependency ESM, one direction of data flow:

`URL fragment + localStorage` → `sanitizeState` (`core.mjs`, clamps/whitelists every field) →
pure ledgers (`parameterLedger`, `cacheLedger`, `servingLedger`, `trainingLedger`, `networkLedger`,
`roofline`) and `buildScene` (`diagrams.mjs`) → scene IR (`scene.mjs`) → SVG + inspector (`app.mjs`).

- `models.mjs` is the only place checkpoint/hardware facts live; every displayed total is derived from it, never re-typed in a template.
- `diagrams.mjs` dispatches on `state.view`, plus `state.stage` focal views (`inputs` when `m.vision`, `residual` when `m.residual!=='standard'`, `ple` for `q38flash`, `ced` when `m.encoderLayers`). A new stage needs four edits or it silently never renders: the `choices.stage` whitelist in `sanitizeState`, the dispatch, the segment list in `app.mjs`, and the model-switch guard that resets a stale stage.
- Scene IR is plain data: `nodes/edges/groups/labels`. `structuralReport(scene)` is the geometry contract — unique ids, finite positive boxes, everything inside `scene.width/height`, edge endpoints resolvable, orthogonal (no diagonal) segments. Tests assert it returns `[]` for every layer of every model and every lens.
- `app.mjs` exposes `window.STRATUM` (`getState/setState/getScene/selectNode/geometryReport/exportSVG/exportConfig/version`). That global is the seam the Chromium suite drives; keep it stable.
- Reading-size chrome prose must be *measured*, not merely subject to a policy: `aesthetics.py`'s `CHROME_TEXT_SELECTORS` is the measured set, `CHROME_PROBE_LIVE` fails if one of its classes is renamed out from under it, and `CHROME_COVERAGE_OK` fails if a reading-size chrome rule is in neither that set nor the recorded `UNMEASURED_CHROME` debt list.
- `presentations.py` opens the artifact at the default state only; `focal_states()` derives the extra states it must also visit from the live model table. A stage that carries prose or a table of its own reaches a reader unmeasured unless it appears there.
- `innovations.mjs` sits between `diagrams.mjs` and `app.mjs` in the module order: it imports ledgers from `core.mjs`, and `app.mjs` imports `innovationRows` from it. `COVERAGE_UNCITED` and `COVERAGE_DERIVED` in `verification/ced.py` read the rendered rows out of `samples.innovations`; `COVERAGE_DERIVED` calls the same `derived_main_kv_bytes_per_token` the cache ledger is held to, so the exposition is checked against the computation rather than against itself.
- `build.mjs` inlines `src/styles.css` and concatenates the modules into an IIFE module registry. It supports only single-line `import {…} from './x.mjs';` and `export const|let|function|class` / `export {…};` — anything else (default exports, namespace imports, multi-line import lists, external specifiers) throws `Unsupported module syntax`. Module order is the hardcoded `files` array.

**Verification** — six independent tracks that never infer one from another:

- `verification/engineering.py` runs verifier tests → node tests → build → repeat build (byte-identical check) → `tests/browser.test.py`, then writes `verification/reports/engineering.json` binding four digests: `consumer_sha256` (live inventory from `node tools/export-verification.mjs`), `verifier_sha256` (all `verification/*.py`, `tools/*`, workflows, schemas, trust, `package.json`), `lock_sha256` (target + source manifests), `artifact_sha256` (`dist/index.html`).
- `verification/runner.py` re-derives all four and rejects a receipt that is stale, >24h old, or bound to different bytes. Practical consequence: **any** edit under `src/`, `tools/`, `verification/`, workflows or manifests invalidates the receipt — re-run `verify:engineering` before quoting a science/runtime result.
- `tools/export-verification.mjs` derives the inventory from the running code (core fields, function source hashes, file hashes, and a sampled scene-contract signature per node `info` and per graph). Renaming a node id or editing inspector text changes those signatures and therefore the consumer coverage the audit expects.
- `verification/manifests/observatory.json` is the active target (3520 claims, 32 contracts, 32 runtime obligations); `sources.json` is the source catalog; `reviews.json` and `trust/reviewers.json` are intentionally empty of fabricated identities. `runtime.json` is empty in the public distribution: private GPU probes and their receipts were withdrawn for privacy. Never restore them from local backups or Git history into public output. Binding tests use isolated synthetic fixtures; absent real receipts remain NOT_RUN. What retires a receipt depends on what it attests: **behavioural** obligations bind `lock_sha256` + `consumer_sha256`, so any manifest or `src/` edit turns them into `RUNTIME_STALE` FAIL; **hardware-attribute** obligations (`gpu`, `scale_up`, `scale_out`) do not, because no edit of ours changes what a machine is. They bind `probe_schema_sha256` (the hardware data model — `RUNTIME_SCHEMA` BLOCKED), `contract_sha256` (the contract record with bindings applied — `RUNTIME_CONTRACT_DRIFT` FAIL) and `hardware_sha256`, which the gate **re-derives** from the `stratum-hardware/v1` capture the receipt names in `hardware_artifact` (a digest that does not reproduce is `RUNTIME_HARDWARE_DRIFT` FAIL; an absent or unpinned record is `RUNTIME_HARDWARE` BLOCKED). They still expire, because nothing here can detect a re-capped GPU or a swapped NIC between two probes. The repair is always to **re-probe the hosts and rebuild the receipts** with `tools/build-runtime-receipts.py` — never to edit a recorded digest in place, even when the re-probe returns byte-identical values.
- Evidence anchoring: an evidence entry naming a source but no selector reports `CLAIM_ANCHOR` BLOCKED. That is the correct fail-closed report, not a defect to be cleared by inventing a quote. `docs/verifiers/ANCHOR_RESIDUE.md` records a decision for every citation that could not be anchored — consult it before touching one. Derived claims carry provenance through their `dependencies`; a citation whose source a dependency already cites is a copy and may be removed, one no dependency supplies may not. PDF sources cannot be pinned: `sources.py` and `evidence.py` decode strictly as UTF-8 before any other check.
- `verification/aesthetics.py` is a fourth gate, peer to `engineering.py`, reached by
  `npm run verify:aesthetics`. It measures WCAG AA contrast against a resolved backdrop stack, a
  rendered font-size floor, label clearance (separation, not mere non-intersection), edge crossings
  and label truncation, and compares a normalized composition signature against
  `verification/baselines/aesthetics/composition.json`. The signature names colours by token and
  rounds coordinates, so retuning a palette does not churn it while a label changing semantic role
  does. `policy.json` is inside `toolchain_digest`; recorded baselines deliberately are not.
  It also enforces the two type scales and the spacing scale statically from the stylesheet:
  a **chrome scale** (10/11/12/14/16/20/24/32) that is a design choice, and a **canvas scale**
  that is a frozen inventory — those sizes are locked to node boxes authored in
  `src/diagram-*.mjs`, so changing one means re-authoring geometry and re-verifying 544 layers.
  AA (4.5:1) is the floor everywhere; reading-size **chrome** prose is held to AAA (7:1), while
  canvas labels stay at AA because they sit on the semantic tint fills that carry the colour
  grammar. It is **engineering-class** evidence — measured on our own bytes — and per `docs/verifiers/DESIGN.md`
  trust boundary 1 it must never contribute to a scientific PASS. It is not a member of
  `contracts.DOMAINS`, because adding a domain would force both a contract and a runtime obligation.
- `verification/schematics.py` (`npm run verify:schematics`) checks what the diagram *means*.
  `tools/export-scenes.mjs` dumps the scene IR plus, per scene, what `core.mjs` independently says
  is true; Python judges the two against each other, so agreement is a cross-check rather than
  self-consistency. No browser, which is what makes the full 755-scene / 544-layer matrix
  affordable. It enforces the IR's closed value sets, the groups and labels `structuralReport`
  never inspects, the `kind==='delay'` / `delay` biconditional, lane collinearity (invisible to
  crossing counts), scene-vs-model routing and layer-tile agreement, and the row alignment
  `OBSERVATORY.md` asserts. Connectivity is a **metric**, not a rule: every lens carries annotation
  cards that sit beside the flow deliberately, and the IR has no annotation flag.
- `verification/presentations.py` (`npm run verify:presentations`) checks what reaches a reader.
  It is the only suite that opens the artifact from a real `file://` origin. It validates the
  exported SVG and the exported scenario against `schemas/scenario.schema.json`, checks accessible
  names, heading order and visible focus, emulates print and reduced motion, asserts nothing is
  *loaded* from an external origin (clickable source citations are the provenance feature and are
  fine), and consumes `previews/manifest.json` so previews cannot silently go stale **or
  silently omit a checkpoint**: `PREVIEW_FRESH` compares the recorded artifact digest, and
  `PREVIEW_COVERAGE` requires every model in the live table — and every encoder/decoder page — to
  be depicted, because a manifest showing eight of nine checkpoints is exactly as fresh as one
  showing all nine. It also measures the `visualization` contract record.
- `.github/workflows/pages.yml` deploys on `engineering`, `schematics`, `aesthetics`, `presentations` and `package`. Each must genuinely pass; none is `continue-on-error`. There is no `publication` job: completeness is offline-only (see above), and `publication-contradictions` runs the audit on CI's bytes and gates on contradictions alone. `package` exists so that building the deployable artifact is never entangled with the scientific audit. A test asserts no `publication` job exists and that `publication-contradictions` never becomes a deploy gate.

Every model/view/hardware/stage axis in every sweep, test and gate is **derived from the application**, never transcribed. A literal encoding "there are nine models" or "there are seven views" is only acceptable where it fails loudly on divergence (`test_observatory.py`, `tests/core.test.mjs:5`, `tests/diagrams.test.mjs:56` are verified loud). A comment saying "keep in sync" is not. `previews/` is published output: every file in it must be produced by `npm run capture` and named by a capture record -- `PREVIEW_INVENTORY` enforces it, `PREVIEW_INTEGRITY` verifies each recorded `svg_sha256`, and scratch scripts write under `tests/`.

`docs/verifiers/` holds the per-domain requirements (MODEL/GPU/NETWORK/EVIDENCE/…) and `AGENT_RUNBOOK.md` the evidence protocol; `docs/design/` records the design rationale.


## Dedicated entry points

`src/pages.mjs` registers dedicated entry points; `build.mjs` emits them and the atlas from
one shell. Keep every registered page inside engineering artifact identity, repeat-build checks,
browser/presentation coverage, preview identity, and Pages packaging. Qwen3.8 permits its 27B and
Flash-Next variants; Kimi K3 and DeepSeek V4.1 fix checkpoint identity. V4.1 defaults to CED.
Atlas cards link to each dedicated model page; every page links back. Shared hardware/network
sections have fixed views and no fixed checkpoint; Networking exposes its workload model.
Global section links preserve scenario state. Model tabs exclude Compare and shared sections. `npm start` serves `dist/`.


Book accounting: `verification/manifests/accounting-methods.json` pins methods/examples and is
bound into the toolchain and publication consumer inventory. `tools/export-accounting.mjs`
exports actual live ledger results; `verification/accounting.py` independently constructs
expected values. Do not derive its oracle from the JavaScript ledger, substitute a generic
Transformer formula for specialized architecture coverage, or use `methodology_reference`
as publisher/framework/vendor authority. Reader notes come from `core.accountingAssumptions`;
they describe the current scenario, not a scientific PASS badge. Engineering requires the
accounting command. Source acceptance remains in publication.

PyTorch examples: `examples/deepseek_v41/*.py` is the importable source of truth.
Constructors share `V41Config`; numerical CPU fixtures replace sizes, not architecture
modes. Edit the Python files, then build to regenerate `src/pytorch-source.mjs`; never
hand-edit that generated copy. `npm run validate:pytorch` runs the CPU program and
`npm run test:pytorch` checks independent math, precision, gradients and source delivery.

CED chapter selectors use `core.layerNumbers`: Encoder owns absolute layers 1..encoderLayers; Decoder owns encoderLayers+1..layers. Legacy model-view block links resolve to the owning chapter without renumbering. Both chapters use the shared detailed block renderer. Keep the full-backbone verification sweep, even though each chapter shows only its half.

Worked traces: numerical inputs, intermediates and outputs come from the CPU program
`examples/deepseek_v41/worked_examples.py`, saved with source fingerprints in
`worked-results.json`. Regenerate with `npm run generate:worked`; never hand-edit
`src/worked-results.mjs`. Build rejects stale recipe hashes. CPU artifact tests compare
current execution with saved values; independent scalar tests check the math. Browser
tests compare displayed values and replay nodes with that same artifact. Teaching
fixtures do not establish checkpoint parity, scientific acceptance or GPU performance.

Numerical notation: never use thousands separators in reader-facing numbers.
Write `5120` and `[B, S, 4, 5120]`; commas in shapes separate axes only.
Prefer code typography or mathematical typesetting for numbers and formulas.
Keep surrounding prose readable; do not reduce type sizes to fit notation.
V4.1 encoder/decoder layer selectors group absolute layer tiles by the checkpoint's
SWA and Full/Reindex-with-Reuse bundles, derived through `layerBands`, rather than
wrapping the chapter's layers into an arbitrary rectangular grid.
