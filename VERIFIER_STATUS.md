# Current verifier status

Four gates, reported separately. See `VERIFICATION.md` and the machine-readable
`verification/reports/` files.

| Gate | Result | Exit |
|---|---|---:|
| Engineering | PASS | 0 |
| Aesthetics | FAIL — 21 measured legibility defects | 1 |
| Scientific publication | BLOCKED | 2 |
| Runtime certification | BLOCKED, 31 obligations NOT_RUN | 2 |

The aesthetics gate is red on real, currently-shipping defects: 17 WCAG AA contrast
failures (worst 3.28:1 against a 4.5:1 requirement, in the light "paper" theme, including
text inside the exported SVG) and 4 rendered font sizes below the 9px chrome floor.
The fix is to retune the tokens, never to lower `verification/baselines/aesthetics/policy.json`.

BLOCKED means not established. It is not a claim that anything is false, and it is not a
passing test. The historical Kimi-only receipt does not apply.
