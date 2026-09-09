# Coding-agent runbook

**Current application:** STRATUM Observatory 3.0.0. Active target: `verification/manifests/observatory.json`; actual consumer exporter: `node tools/export-verification.mjs`. Eight checkpoints, seven lenses; old Kimi receipts are not transferable.

**Objective:** add evidence or functionality without making unresolved science look verified.

1. Read the target's audit report and relevant verifier spec. Select exact consumer/claim IDs; do not replace the task with a broad paper summary.
2. Resolve publisher configuration, publisher implementation and independent framework implementation at immutable commits. Follow active call paths and kernel dispatch, not comments or names alone. Keep differences visible.
3. Acquire bytes into a fresh candidate. Resolve JSON/AST/span anchors, units, shapes, aliases, guards and dependencies. Never regenerate accepted values/checksums merely to erase drift.
4. Write a failing arithmetic/contract/negative test. Implement the smallest change. A scientific label is code: update its scene claim as well as the calculation.
5. Obtain independent signatures for exact claim/contract changes. No private reviewer key belongs in the project, CI artifacts or an agent prompt. Publication reviews and real execution receipts are different deliverables.
6. Run engineering, publication and—when asserting execution—runtime gates. Report all three statuses separately. `BLOCKED` and `NOT_RUN` are not green.

```sh
npm run test:verifiers
npm run verify:engineering
npm run verify:science
npm run verify:runtime
```

**Do not:** disable a gate, add `continue-on-error`, treat config as the second code implementation, promote fixtures, relax tolerance after inspecting outputs, call an analytic bound measured latency, multiply overlapping parallel groups, infer full-rack capacity from a per-GPU value, or reuse a receipt after changing source/lock/verifier code.

**Completion response:** name changed consumers and source revisions; report exact test/gate results; link the JSON/Markdown reports; distinguish verified, blocked, unrun and unsupported cases; list the next evidence artifact needed for each blocker. Never say “all scientifically accurate” because the screenshot suite passed.
