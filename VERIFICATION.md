# STRATUM Observatory 3.0 — delivered-artifact verification

Generated from the fresh engineering receipt and the three adjacent gate reports in
`verification/reports/`.

## Separate results

| Gate | Result | Exit | What actually ran |
|---|---|---:|---|
| Engineering | PASS | 0 | 171 verifier tests, 40 model/graph tests, two byte-identical builds, actual Chromium suite |
| Aesthetics | FAIL | 1 | 112 scenarios, 106 distinct measurements; 21 real legibility defects |
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

The first run of `npm run verify:aesthetics` fails on **21 defects that ship today**:

- **17 WCAG AA contrast failures.** Worst is `--subtle` on `--bg` at **3.28:1** against a 4.5:1
  requirement, in the light "paper" theme. Also `--subtle` on `--surface` 3.49:1 (light) and
  4.38:1 (dark), `--amber` on `--amber-bg` 4.15:1, `--purple` on `--purple-bg` 4.30:1. Some of
  this text is inside the **exported SVG**, so it travels with the download.
- **4 rendered font sizes below the 9px chrome floor** (`.source-role` at 8px, `.status-pill`
  at 8.5px).

Composition is byte-identical between dark and light across all 56 scenarios, so position never
depends on theme; that invariant is now recorded in
`verification/baselines/aesthetics/composition.json`.

Ratios are computed from specified colours against a resolved backdrop stack. That is an upper
bound on legibility, not a guarantee of it: anti-aliasing and stem weight are not modelled, and
an element whose backdrop cannot be resolved is reported BLOCKED rather than passed. The remedy
is to retune the tokens, never to lower `verification/baselines/aesthetics/policy.json`.

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

Portable HTML SHA-256: `2df8de74b7d21cae84e4849a65c860c3d69463a0f3b530ff9a66af0b2655b51f`

Scientific lock SHA-256: `1943e4a1d8aa8ec64d72778fb0a4b1e348aa5ad22e363865ab783d64e5baa4ef`

Verifier identity SHA-256: `e14be8b5ca9fb002885cfddd5d8da8f55f346dc0b3077a7e54d757f6d4363d44`

Complete command logs and receipts: `verification/reports/`. Full browser matrix:
`tests/browser-cases.json`. Re-run after changing application, verifier or evidence; old receipts
are not transferable.

## Reproduce

```sh
uv sync --frozen
uv run --frozen playwright install chromium
npm run verify:aesthetics    # exit 1 until the token contrast defects are fixed
npm run verify:engineering   # exit 0
npm run verify:science       # exit 2 while evidence and reviews are outstanding
npm run verify:runtime       # exit 2 while experiments are unrun
```

`npm run verify` requires engineering AND publication. The Pages workflow remains fail-closed and
does not deploy. Success is not all-green; success is each gate telling the truth about what it
actually checked.
