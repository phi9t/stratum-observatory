# Current verifier status

Four gates, reported separately. See `VERIFICATION.md` and the machine-readable
`verification/reports/` files.

| Gate | Result | Exit |
|---|---|---:|
| Engineering | PASS | 0 |
| Aesthetics | PASS | 0 |
| Scientific publication | BLOCKED | 2 |
| Runtime certification | BLOCKED, 31 obligations NOT_RUN | 2 |

`npm run verify` requires engineering, aesthetics and publication, so it exits 2 by design.
GitHub Pages deployment additionally requires all three; publication is expected to block
until upstream evidence and independent reviews exist.

BLOCKED means not established. It is not a claim that anything is false, and it is not a
passing test. The historical Kimi-only receipt does not apply.
