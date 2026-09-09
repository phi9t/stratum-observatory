# STRATUM Observatory 3.0 — delivered-artifact verification

Generated from the fresh engineering receipt completed at **2026-09-06T20:37:20.906862+00:00**.

## Separate results

| Gate | Result | What actually ran |
|---|---|---|
| Engineering | PASS | 117 verifier tests, 40 model/graph tests, two byte-identical builds, actual Chromium suite |
| Scientific publication | BLOCKED (exit 2) | Current eight-model consumer inventory and strict evidence/review audit |
| Runtime certification | BLOCKED (exit 2); 31 NOT_RUN | No model/GPU/fabric experiment was executed or attested |

## Browser evidence

**1,336 rendered states**, including all **504 decoder layers** across eight checkpoints in both label modes. This includes **56 responsive viewport/lens/theme combinations** and **32 additional device-scale cases at 1.25 and 2**. There were **0 detected geometry defects, 0 runtime JavaScript errors and 0 external runtime requests**. Actual model/layer/scenario controls, search, keyboard navigation, tracing, focus, zoom, URL restore, SVG/XML rendering and JSON exports passed.

The suite injects the exact portable document into isolated Chromium pages. It tests the delivered HTML, not only imported source modules. It is not Safari/Firefox certification, all-system-font coverage, physical filesystem policy certification, an exhaustive state-space proof or GPU execution. Fractional-DPR capture revealed a 30px two-line Kimi pill with insufficient clearance: a failing regression was added, the box grew to 34px and its edge was rerouted. No font shrink or linter tolerance change hid that defect.

Twenty-three full-view PNG/SVG capture pairs plus a narrow-screen image were captured from the same artifact. Representative dark, paper, Kimi, Flash-Next, GPU, networking and mobile layouts were visually inspected.

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

The publication audit has **13,255 BLOCKED findings**, with no FAIL contradictions in the migrated target. Many findings are cascading source dependencies and absent independent reviews; this is **not** a claim that 13,255 facts are false. Static references and inspected commit identities are preserved, but raw bytes, precise selector digests, reviewed scenario records and signed interpretations are incomplete. No review identity, accepted source hash or execution receipt was invented. The UI/export states remain reference-mapped, review-pending and runtime-unrun.

## Artifact identity

Portable HTML SHA-256: `2df8de74b7d21cae84e4849a65c860c3d69463a0f3b530ff9a66af0b2655b51f`

Scientific lock SHA-256: `aea0f169461434357667aa080840dffc36858f3a88bbcdb4dec3fec157bc6f0d`

Verifier identity SHA-256: `4728343c2115113bdabb5041aef42cd05eae781b8b04227da91f1f34bfa68156`

Complete command logs and receipts: `verification/reports/`. Full browser matrix: `tests/browser-cases.json`. Capture states and SVG hashes: `previews/manifest.json`. Re-run after changing application, verifier or evidence; old receipts are not transferable.

## Reproduce

```sh
npm run verify:engineering
npm run verify:science
npm run verify:runtime
npm run capture
```

The last two verification profiles intentionally exit 2 while evidence or experiments are outstanding. `npm run verify` requires engineering AND publication. The Pages workflow remains fail-closed. No remote website or repository was deployed by this delivery.
