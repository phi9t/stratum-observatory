# Current verifier status

Six gates, reported separately. See `VERIFICATION.md` and the machine-readable
`verification/reports/` files.

| Gate | Result | Exit |
|---|---|---:|
| Engineering | PASS | 0 |
| Schematics | PASS | 0 |
| Aesthetics | PASS | 0 |
| Presentations | PASS | 0 |
| Scientific publication | BLOCKED | 2 |
| Runtime certification | BLOCKED, 31 obligations NOT_RUN | 2 |

`npm run verify` requires engineering, schematics, aesthetics, presentations and publication, so it
exits 2 by design. GitHub Pages deployment requires all of them.

BLOCKED means not established. It is not a claim that anything is false, and it is not a
passing test. The historical Kimi-only receipt does not apply.
