# STRATUM Observatory 3.0 — delivered-artifact verification

Generated from the fresh engineering receipt and the three adjacent gate reports in
`verification/reports/`.

## Separate results

| Gate | Result | Exit | What actually ran |
|---|---|---:|---|
| Engineering | PASS | 0 | 181 verifier tests, 40 model/graph tests, two byte-identical builds, actual Chromium suite |
| Aesthetics | PASS | 0 | 112 scenarios, 102 distinct measurements; type, spacing, contrast and composition |
| Scientific publication | BLOCKED | 2 | Eight-model consumer inventory and strict evidence/review audit |
| Runtime certification | BLOCKED | 2 | 31 obligations NOT_RUN; no model/GPU/fabric experiment was executed or attested |

## Toolchain

CPython 3.13.12 and Playwright 1.55.0, both provisioned by uv from `uv.lock`.
OpenSSL 3.6.2 verifies review signatures. `uv sync --frozen` then
`uv run --frozen playwright install chromium` reproduces it. The interpreter, the lock, the
Python pin, `tests/`, `.github/CODEOWNERS` and the aesthetics policy are all inside
`toolchain_digest`, so a receipt cannot outlive any of them.

## Browser evidence

**1,336 rendered states**, including all **504 decoder layers** across eight checkpoints in both
label modes, **56 responsive viewport/lens/theme combinations** and **32 device-scale cases at
1.25 and 2**. **0 geometry defects, 0 runtime JavaScript errors, 0 external runtime requests.**
Model/layer/scenario controls, search, keyboard navigation, tracing, focus, zoom, URL restore,
SVG/XML rendering and JSON exports passed.

This result now reproduces under **macOS system fonts** as well as the Linux CI fallback, which
is meaningful because no fonts are bundled and every geometry check is built on measured
`getBBox()` bounds. The layout does not depend on a single font stack.

It is not Safari/Firefox certification, all-system-font coverage, an exhaustive state-space proof
or GPU execution.

## Aesthetics evidence

`npm run verify:aesthetics` passes: **207 checks, 0 failures** across 112 scenarios. The first run
of this gate failed on 21 defects that shipped in the delivered artifact; all are now fixed at the
source rather than by moving a threshold.

- **17 WCAG AA contrast failures**, worst `--subtle` on `--bg` at **3.28:1** against a 4.5:1
  requirement in the light "paper" theme, some of it inside the exported SVG. Five tokens were
  retuned; the measured ratios are now 4.63–6.96:1 and are pinned in a test, so the algorithm stays
  calibrated rather than merely passing.
- **4 rendered sizes below the floor** (`.source-role` 8px, `.status-pill` 8.5px). The chrome type
  scale removed the whole class: the smallest chrome text is now 10px.
- **24 ad-hoc font sizes became 8**, and 31 ad-hoc spacing values became one 2/4px rhythm. Both are
  now enforced statically, so they cannot silently return.

AA (4.5:1) is the floor everywhere. Reading-size **chrome** prose is held to AAA (7:1). Canvas
labels stay at AA deliberately: they sit on the semantic tint fills that carry the colour grammar
of `docs/design/OBSERVATORY.md`, and forcing 7:1 there would wash that grammar out of the diagram.

Composition is byte-identical between dark and light across all 56 scenarios, and the recorded
baseline did not move through the entire redesign — confirming the change was purely presentational
and touched no scene geometry.

Ratios are computed from specified colours against a resolved backdrop stack. That is an upper
bound on legibility, not a guarantee of it: anti-aliasing and stem weight are not modelled, and an
element whose backdrop cannot be resolved is reported BLOCKED rather than passed.

## Scientific inventory

| Item | Count |
|---|---:|
| Input leaves | 455 |
| Function/file identities | 27 |
| Distinct sampled scene contracts | 2520 |
| Claims including domain obligations | 3025 |
| Candidate source entries | 56 |
| Accepted raw-byte snapshots | 0 |
| Scientifically certified claims under the signed policy | 0 |
| Domain contracts / runtime obligations | 31 / 31 |

The publication audit has **15,740 BLOCKED findings and no FAIL contradictions**. Many are
cascading source dependencies and absent independent reviews; this is **not** a claim that 15,740
facts are false. No review identity, accepted source hash or execution receipt was invented.

**Verified scientific claims: 0.** Two claims previously passed — the stylesheet and the HTML
entry point — because `presentation` claims skipped review entirely. Every claim is now reviewed,
so those green cells are gone. Calling a user-visible value presentational is itself a judgement,
and it is the judgement most worth checking.

**Qwen presets have no publisher implementation.** `huggingface/transformers` was previously
registered as `publisher_code`, so `transformers` plus `sglang` satisfied "publisher
implementation AND independent framework implementation" using two third-party reimplementations
and no publisher at all. transformers is now `framework_code`, and dual-code independence compares
organizations rather than repository names. Qwen model claims are consequently and correctly
BLOCKED on `MODEL_DUAL_CODE` and `CLAIM_ROLE`.

## Known and unclosed

Recorded so they are not mistaken for oversights:

- The engineering receipt is **self-produced and unsigned**. Every field it asserts is
  hand-computable. Closing this needs CI attestation and branch protection, not local code.
  `.github/CODEOWNERS` now exists but has no force without protected branches.
- **23 of 31 contracts carry empty `bindings`**, so their constants are hand-transcribed into the
  manifest rather than bound to actual application values. Only the 8 model contracts bind.
- No upstream bytes have been acquired, so every source remains `SOURCE_UNFETCHED`.

## Artifact identity

Portable HTML SHA-256: `ced16558478295db61e131e4d538be83177d06f2caadd1db6d995638a8c1a573`

Scientific lock SHA-256: `898d279515ae8ba392a555d9e93690654989f941d3e8d745646e4ee5972ff0dd`

Verifier identity SHA-256: `6ca6fbc42fd053815bb20967e5cdb22fe295380bba326602b054c1efc846a540`

Complete command logs and receipts: `verification/reports/`. Full browser matrix:
`tests/browser-cases.json`. Re-run after changing application, verifier or evidence; old receipts
are not transferable.

## Reproduce

```sh
uv sync --frozen
uv run --frozen playwright install chromium
npm run verify:aesthetics    # exit 0
npm run verify:engineering   # exit 0
npm run verify:science       # exit 2 while evidence and reviews are outstanding
npm run verify:runtime       # exit 2 while experiments are unrun
```

`npm run verify` requires engineering, aesthetics AND publication. The Pages workflow remains fail-closed and
does not deploy. Success is not all-green; success is each gate telling the truth about what it
actually checked.
