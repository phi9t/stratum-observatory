# Delivered-artifact verification (v2.1)

The current authoritative result is **[VERIFIER_STATUS.md](VERIFIER_STATUS.md)**. It supersedes the earlier single-gate verification summary.

The software/geometry gate passes. Scientific publication is BLOCKED under the new evidence policy; runtime certification has not been performed. A previous green software test result must not be represented as source or GPU/network validation.

Machine-readable results:
- `verification/reports/engineering.json`
- `verification/reports/publication.json`
- `verification/reports/runtime.json`

Reproduce with `npm run verify:engineering`, `npm run verify:science` and `npm run verify:runtime`. Domain contracts, trust boundaries and the coding-agent protocol are specified in `docs/verifiers/`.
