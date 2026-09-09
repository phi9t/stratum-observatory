# STRATUM Verification Implementation Plan

**Goal:** Add reusable fail-closed scientific verifiers without changing model calculations or schematic geometry.
**Architecture:** Local inventory -> versioned manifests -> pinned evidence/semantic contracts -> signed review/execution evidence -> report -> CI gate.
**Tech Stack:** Node 20+, Python 3.11+ standard library, OpenSSL for Ed25519 review verification, existing Chromium checks.
**Spec:** `docs/verifiers/DESIGN.md`

## Global constraints
Zero new application runtime dependencies. No upstream model code execution during source inspection. Keep existing numerical outputs unchanged. Never replace missing proof with an assumed pass. Never silently rewrite source locks or review signatures.

## Task 1 — evidence engine
- [x] Write `verification/tests/test_evidence.py`: missing/duplicate claims, changed value/body, mutable revision, path escape, digest mismatch, JSON extraction, code symbol/span, dual-code roles, dependency cycles, expired and invalid signatures.
- [x] Run `python3 -m unittest verification.tests.test_evidence` and record missing-implementation failure.
- [x] Implement `verification/evidence.py` and `verification/model.py`, returning structured `Finding` records.
- [x] Run evidence tests; commit.

## Task 2 — domain contracts
- [x] Write negative and positive fixtures in `verification/tests/test_domains.py` for every domain in DESIGN.
- [x] Run `python3 -m unittest verification.tests.test_domains` before implementation.
- [x] Implement pure domain validators under `verification/domains/`; use typed units/scopes and no network calls.
- [x] Run all verifier tests; commit.

## Task 3 — integration and provenance
- [x] Test actual core inventory, empty inventory refusal, target coverage, source acquisition bounds, unknown flags and release exit codes.
- [x] Add `tools/export-verification.mjs`, source catalog, target manifest, runner and explicit acquisition command. Core values are read from actual exports, not transcribed ledger values.
- [x] Run local audit; retain missing-evidence blockers in generated JSON/Markdown.

## Task 4 — CI and agent interface
- [x] Add domain specs, schema, root agent contract, negative/mutation tests, CI engineering/publication jobs and deploy dependency.
- [x] Run `npm run verify:engineering`; run strict audit and confirm blockers return nonzero.
- [x] Produce artifact-bound verification report and source bundle. Report exactly which gates ran and which remain blocked; do not claim runtime certification.

## Delivery interpretation
Implementation tasks are complete. Scientific source hydration, independent approvals and actual hardware/framework/fabric experiments are intentionally not checked off by this implementation plan; they remain machine-visible obligations in the publication/runtime reports. Final command outcomes are recorded in `VERIFIER_STATUS.md`.
