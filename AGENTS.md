# STRATUM coding-agent contract

Read `README.md`, `docs/design/OBSERVATORY.md`, `docs/AGENT_VERIFIER.md`, and the relevant `docs/verifiers/` contract before changing code.

## Boundaries
- Data/source identities: `src/models.mjs`. Math: `src/core.mjs`. No DOM in either.
- Visual IR: `src/scene.mjs`. Model motifs: `src/diagram-model.mjs`. Systems: `src/diagram-systems.mjs`. Comparison/evidence/focal views: `src/diagram-meta.mjs`.
- UI/state/export: `src/app.mjs`; shared style tokens: `src/styles.css`.
- `src/diagrams.mjs` dispatches templates. `build.mjs` is dependency-free and deterministic.
- Current scientific target is `verification/manifests/observatory.json`; `kimi-k3.json` is historical, not the current consumer inventory.

## Non-negotiable
Preserve exact variant identity, shape contracts, precision/rate units, ownership and time/depth axes. No branch invented to fit a template. Shared experts do not consume router logits. Do not remove meaningful edges to pass geometry tests. Expand lanes/boxes, not font degradation. Do not turn missing numeric evidence into zero.

Every changed consumer needs its exact binding and upstream evidence chain. A new fingerprint is not an approval. Do not overwrite accepted source bytes, reviewed subjects or trust roots to silence drift. New variants must appear in the actual exporter and browser layer matrix.

Write a failing behavioral/negative test, implement locally, run `npm run verify:engineering`. Run scientific/runtime gates separately and report BLOCKED/NOT_RUN faithfully. `npm run verify` and Pages remain fail-closed. No `continue-on-error`, fake raw hashes, invented reviewer keys, copied signatures or benchmark claims from payload/rate calculations.

Previews must come from the current portable HTML via `npm run capture`. Include actual counts, browser, artifact identity and scope in completion reports. No weights, credentials, telemetry, bundled fonts or external runtime scripts.

## Commands

```sh
npm test                              # Node model math + scene contract tests (tests/*.test.mjs)
node --test tests/core.test.mjs       # One test file
node --test --test-name-pattern 'QSA' tests/core.test.mjs   # One test by name
npm run build                         # index.html + src/ -> dist/index.html (deterministic)
npm start                             # python3 -m http.server 8000; unbundled dev at localhost:8000
npm run test:verifiers                # uv run python -m unittest discover -s verification/tests -v
uv run python -m unittest verification.tests.test_evidence -v   # One verifier module
uv run python tests/browser.test.py   # Chromium suite; CHROMIUM_PATH optional, Playwright resolves it
npm run capture                       # Regenerate previews/ PNG+SVG from the current build
npm run verify:aesthetics             # Measured legibility, type/spacing scales, composition, drift
npm run verify:aesthetics -- --record  # Write a candidate composition baseline for review
npm run verify:engineering            # Verifier tests + node tests + two builds + browser; writes receipt
npm run verify:science                # Publication gate; exits 2 while BLOCKED
npm run verify:runtime                # Runtime gate; exits 2 while experiments are NOT_RUN
npm run verify                        # engineering AND science, fail-closed
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
- `diagrams.mjs` dispatches on `state.view`, plus `state.stage` focal views (`inputs` when `m.vision`, `residual` when `m.residual!=='standard'`, `ple` for `q38flash`).
- Scene IR is plain data: `nodes/edges/groups/labels`. `structuralReport(scene)` is the geometry contract — unique ids, finite positive boxes, everything inside `scene.width/height`, edge endpoints resolvable, orthogonal (no diagonal) segments. Tests assert it returns `[]` for every layer of every model and every lens.
- `app.mjs` exposes `window.STRATUM` (`getState/setState/getScene/selectNode/geometryReport/exportSVG/exportConfig/version`). That global is the seam the Chromium suite drives; keep it stable.
- `build.mjs` inlines `src/styles.css` and concatenates the modules into an IIFE module registry. It supports only single-line `import {…} from './x.mjs';` and `export const|let|function|class` / `export {…};` — anything else (default exports, namespace imports, multi-line import lists, external specifiers) throws `Unsupported module syntax`. Module order is the hardcoded `files` array.

**Verification** — four independent tracks that never infer one from another:

- `verification/engineering.py` runs verifier tests → node tests → build → repeat build (byte-identical check) → `tests/browser.test.py`, then writes `verification/reports/engineering.json` binding four digests: `consumer_sha256` (live inventory from `node tools/export-verification.mjs`), `verifier_sha256` (all `verification/*.py`, `tools/*`, workflows, schemas, trust, `package.json`), `lock_sha256` (target + source manifests), `artifact_sha256` (`dist/index.html`).
- `verification/runner.py` re-derives all four and rejects a receipt that is stale, >24h old, or bound to different bytes. Practical consequence: **any** edit under `src/`, `tools/`, `verification/`, workflows or manifests invalidates the receipt — re-run `verify:engineering` before quoting a science/runtime result.
- `tools/export-verification.mjs` derives the inventory from the running code (core fields, function source hashes, file hashes, and a sampled scene-contract signature per node `info` and per graph). Renaming a node id or editing inspector text changes those signatures and therefore the consumer coverage the audit expects.
- `verification/manifests/observatory.json` is the active target (3025 claims, 31 contracts, 31 runtime obligations); `sources.json` is the source catalog; `reviews.json`, `runtime.json` and `trust/reviewers.json` are intentionally empty of fabricated identities.
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
  `src/diagram-*.mjs`, so changing one means re-authoring geometry and re-verifying 504 layers.
  AA (4.5:1) is the floor everywhere; reading-size **chrome** prose is held to AAA (7:1), while
  canvas labels stay at AA because they sit on the semantic tint fills that carry the colour
  grammar. It is **engineering-class** evidence — measured on our own bytes — and per `docs/verifiers/DESIGN.md`
  trust boundary 1 it must never contribute to a scientific PASS. It is not a member of
  `contracts.DOMAINS`, because adding a domain would force both a contract and a runtime obligation.
- `.github/workflows/pages.yml` gates Pages on engineering **then** publication; publication is expected to fail until real evidence exists, so Pages does not deploy. The aesthetics job runs alongside and is currently red on real defects.

`docs/verifiers/` holds the per-domain requirements (MODEL/GPU/NETWORK/EVIDENCE/…) and `AGENT_RUNBOOK.md` the evidence protocol; `docs/design/` records the design rationale.
