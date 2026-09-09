# Observatory verifier

## Commands

```sh
npm run verify:engineering
npm run verify:science
npm run verify:runtime
npm run capture
```

The engineering gate runs verifier self-tests, model/graph tests, two builds and Chromium tests against the resulting HTML. It binds its receipt to source, verifier, manifest and artifact hashes. The scientific gate separately requires accepted source snapshots, exact anchors, complete consumer coverage and independent signed reviews. Runtime requires actual artifact-backed experiments on the declared stacks.

## Required regression coverage
- DeepSeek dense exceptions, total-vs-activated expert accounting, no MTP in base decoder ledger.
- Qwen3 GQA/QK norms; 235B query width != residual width; no shared experts.
- Qwen3.8-27B versus Flash-Next: full/QSA layer schedules; SiLU versus sigmoid recurrent gate; four streams and layer-2 PLE only where configured.
- Kimi final layer 93 is MLA; 69 KDA/24 MLA; NoPE does not drop shared key64; weighted latent aggregate precedes normalization/lift; depth residuals are not token recurrence.
- Cache history grows with sequence; recurrent matrices do not; batch scales both. QSA does not erase full KV. PLE auxiliary exclusions remain visible.
- Per-GPU/node/rack scopes, dense FP8 roofline, GB versus GiB, Gb/s versus GB/s, single-rank zero inter-rank traffic.
- Orthogonal routes, finite contained bounds, browser-rendered text/edge/node nonintersections; all 504 decoder layers in both label modes, seven lenses, supported focal views, themes and mobile.
- Actual UI interactions, validated URL restore, no external runtime requests, self-contained SVG and JSON exports.
- Negative provenance, source authority/anchor, signature, changed consumer, dependent-claim and unrun-experiment tests.

A passing test against a template is not proof of every model semantic, font, browser, allocation or GPU implementation. Inspect freshly captured representatives. Source/implementation changes require re-review, not a green label based on old receipts.
