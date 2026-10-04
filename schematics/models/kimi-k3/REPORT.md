# Kimi K3 schematic completion report

Built the exact `moonshotai/Kimi-K3` checkpoint as a portable, playable architecture deep dive. Authored files are confined to `models/kimi-k3/`; generated tools output lives in the permitted ignored directories. No weights were downloaded, no shared tooling was edited, and no git commits or configuration changes were made.

## Artifact and evidence

- Publisher revision: `f831ab66814297da540d832a5235f8e904f29d06`.
- Independent SGLang revision: `fea3c088a1412ba794b1f7998521360ad280c714`.
- FLA numerical-reference revision: `3e52d5abb6ad85312846f39c3e946c2e9881a823`.
- Thirteen pinned source entries in `sources.json`; full reading notes and implementation decisions in [NOTES.md](NOTES.md).
- Portable artifact: `dist/kimi-k3.html`, 567762 bytes, SHA-256 `74a3d36801a9087610389a920eb3bbbccc9a795a33d352f1b2526aad239368cc`.
- Browser: `/usr/bin/google-chrome`, Chrome 145.0.7632.75. Final layout sweep checked 3524 text elements across desktop-light, desktop-dark and mobile-light. Desktop viewport 1440 × 1000 at scale 1; mobile 390 × 844 at scale 2.
- Four `npm run shoot -- kimi-k3` review rounds, each producing 36 PNGs (144 captures across rounds, 36 final files). All images were opened. Final files are in `runs/shots/kimi-k3/`; fixes included overview labels, branch topology, recurrent write-to-cache edge, sigmoid output-gate labeling, signed-state presentation and narrow-screen reference wrapping. Mobile wide diagrams retain readable type and horizontal scrolling.
- Play gate found 234 controls: 116 buttons, 113 operations and 5 ranges. All 12 sampled controls changed the page; tested URL state survived reload.

## What is drawn

Eleven sections cover model overview, tensor/media interfaces, every layer and its selected block, blockwise attention residuals, expanded/absorbed full attention, KDA recurrence, latent routed/shared experts, a running scalar toy decoder, parameter accounting, serving memory and provenance. Full attention's historical rope-named branch is explicitly not rotated in the text reference. Router correction bias affects selection only; shared experts bypass router logits. The last adjacent full-attention layers are retained.

The logical text ledger is **2779484476000** parameters. The Hub total is **2779931837184**. Its excess **447361184** reconciles exactly as vision tower/projector **447358976** plus **2208** serialized A_log surplus entries: `(128 - 96) × 69`. Shape-mode execution independently enumerates the text reference's parameter names and sizes. Neither figure is a packed-storage byte count.

## Numerical verification and limits

All **16 tests pass**. Component fixtures execute pinned publisher router/MoE, eager MLA and depth-mixture definitions isolated by AST, plus the pinned FLA naive gate/recurrence. Inputs and perturbed toy weights are deterministic; saved fixtures carry source fingerprints. Tests check outputs, routed identities, parameter names, real-config shape counts, final-layer identity, bias/weight separation, raw-value depth mixing, causal write-before-read recurrence, sigmoid output gating and parameter reconciliation.

**Whole-decoder upstream numerical parity: UNVERIFIED / NOT_RUN.** This environment has CPU PyTorch but lacks FLA and FlashAttention required by the publisher's fused complete decoder. Eager component checks do not prove its complete call graph, fused-kernel parity, logits or checkpoint parity. The generic check-tool phrase “clean reference's parity test” describes detected test files; the actual proven scope is the component checks above. The page states this limit. The scalar browser decoder uses seeded toy weights, not released checkpoint weights.

Excluded with reasons:

- GPU performance, quantized/packed kernel behavior and released-weight outputs: no weights or GPU runtime were requested or downloaded.
- Complete multimodal execution and vision block internals: this page's clean executable reference is the text decoder. The pinned vision interface, patching, width changes and parameter reconciliation are shown, without claiming vision numerical parity.
- Incremental decode and production cache scheduling in the scalar reference: it implements causal text prefill; serving cache comparisons are source-backed layouts and illustrative arithmetic, not executable backend coverage.
- A separate training implementation: none was established among the pinned public checkpoint sources; no training equivalence is asserted.
- Stratum scientific/runtime/publication gates: NOT_RUN, because this self-contained subtree has no such package commands/contracts. Its local README and skill define the 11 checks below. No scientific acceptance, benchmark or runtime receipt is inferred from them.

No local check gate was skipped or modified.

## Five randomly selected source spot-checks

Selected reproducibly from operation names with random seed 73. All refer to pinned publisher `modeling_kimi_linear.py` at the revision above:

| Operation | Source and inspected shape/order |
|---|---|
| `o_norm` | Constructor lines 539–540 and forward 650–660: per-head RMS normalization and sigmoid full-rank gate before merge/output projection; gain `[128]`. |
| `q_a_layernorm` | Constructor line 368, forward line 419, RMSNorm lines 227–236: normalize `[B,S,1536]` after q_a and before q_b; reference default epsilon is `1e-6`. |
| `expert_w2` | Constructor line 250, forward line 269: SiTU-gated intermediate precedes down projection; routed weight `[3584,3072]`. |
| `b_proj` | Constructor line 529 and forward line 603: normalized stream projects to `[B,S,96]`, then sigmoid update coefficient enters recurrence. |
| `router_sigmoid` | Forward lines 703–727: float32 logits become sigmoid scores before selection-only correction bias; flattened scores `[B*S,896]`. |

## Rubric self-review

| Item | Result | Evidence |
|---|---|---|
| 1 | PASS | Exact moonshotai/Kimi-K3 identity and publisher revision shown on page. |
| 2 | PASS | Pins, facts, coverage and parameters gates pass with explicit numeric reconciliation. |
| 3 | PASS | Five source spot-checks recorded below. |
| 4 | PASS | All 93 layers are selectable; explicit KDA membership, dense prefix, MoE and depth boundaries drive the block. |
| 5 | PASS | Serving comparison cites absorbed MLA cache, A_log narrowing and query-normalization epsilon difference. |
| 6 | PASS | Section numbers come from facts.ts; literals gate passes. |
| 7a | SCOPE | Clean text reference exists; component parity passes, whole-decoder parity remains unverified for the runtime reasons below. |
| 7b | PASS | Real-config shape execution equals 2779484476000 logical text parameters. |
| 7 | PASS | Overview, data, layer/block, op, execution, serving, weights and provenance levels are present. |
| 8 | PASS | Separate projection/norm/mask/cache ops and routed/shared FFN paths; multimodal internals are explicitly excluded. |
| 9 | PASS | Operation inspectors provide descriptions, formulas, tensor shapes and pinned references. |
| 10 | PASS | Code reveals no rotary application in text MLA, selection-only router bias, and sigmoid KDA output gate. |
| 11 | PASS | Layer selection changes attention/FFN/boundary topology and explanatory sentence. |
| 12 | PASS | Causal attention, routing, depth mixture and recurrent-memory widgets compute source rules at declared toy size. |
| 13 | PASS | All 12 sampled controls change the page and reload preserves tested URL state; unsampled op controls were inspected during review. |
| 14 | PASS | All 36 final light/dark/mobile PNGs opened; four screenshot/review rounds corrected topology and labels. Wide graphs scroll on mobile. |
| 15 | PASS | Attention blue, FFN orange, recurrent memory violet and depth teal follow the legend. |
| 16 | PASS | Section ledes and figure captions explain the intended reading. |
| 17 | PASS | Shared kit typography, cards, inspectors and spacing unify the page. |
| 18 | PASS | Data-path operations have channel-width gauges; state matrices retain explicit inspector axes. |
| 19 | PASS | Each section ends in Takeaways; mechanism sections include reasoning Checks. |
| 20 | PASS | Derived widths, cache ratios and state counts explain mechanism tradeoffs. |
| 21 | PASS | Glossary covers architecture vocabulary; formulas and inspectors define local notation. |
| 22 | PASS | Scope limits and verification omissions are listed below. |

## Final local check

`npm run check -- kimi-k3`

| Gate | Result |
|---|---|
| pins | PASS |
| typecheck | PASS |
| literals | PASS |
| format | PASS |
| tests | PASS |
| facts | PASS |
| coverage | PASS |
| parameters | PASS |
| build | PASS |
| layout | PASS |
| play | PASS |

**11 pass, 0 fail, 0 skipped.** Receipt: `runs/check/kimi-k3/report.json`.
