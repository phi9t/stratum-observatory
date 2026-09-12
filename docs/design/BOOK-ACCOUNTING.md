# Book accounting cross-checks

The approved increment adds independent arithmetic checks and reader-visible assumptions to the existing ledgers. Model-specific FLOP accounting remains outside this change.

## Evidence boundary

Book snapshots supply methods and reference examples. Their `methodology_reference` role cannot meet publisher-code, independent-framework, vendor, or communication-library requirements. Exact byte snapshots and anchors follow the existing candidate-source and publication workflow. Raw copyrighted source files remain in the ignored cache.

Arithmetic runs offline against the live exported model and scenario inventory. The Python oracle does not call the JavaScript ledger to construct expected results. Agreement establishes the stated arithmetic only, and unsupported architecture scopes remain explicit. Specialized cache ownership checks do not certify an entire architecture through a generic Transformer formula.

## Reader decisions

Both portable pages expose a native expandable Accounting assumptions section beneath each existing analysis panel. Notes derive quantities from the selected model and current ledgers. They show applicability and exclusions without suggesting that arbitrary reader-selected scenarios have a published audit receipt. Existing URL state, controls and exports remain unchanged.

The training baseline remains BF16 weights and gradients plus FP32 master weights and Adam moments. The additional FP32 accumulation-buffer alternative is displayed separately. Shared CSA2 cache is counted once per retained owner; repeated reads during execution are a separate bandwidth concern.

## Implementation and validation record

- Source policy: restricted methodology role; dated snapshots, edition attribution and exact anchors in `BOOK-SOURCES.md` and the method registry.
- Arithmetic: `tools/export-accounting.mjs` exports the live inventory; `verification/accounting.py` emits JSON and Markdown via `npm run verify:accounting`.
- Integration: arithmetic is a required engineering step; the method registry participates in the verifier digest and publication consumer inventory.
- Reader notes: pure `accountingAssumptions` in `core.mjs`; rendering in `app.mjs`; native details keyboard and print behavior.
- Initial negative reader tests failed for the missing function before implementation. Focused checks cover both baseline preservation and specialized applicability.

Final gate results are recorded in `VERIFICATION.md`, not copied into this design record.

An independent code review found one missing exact source anchor for the hosted playbook's
additional FP32 gradient copy. The line 477 anchor was verified against cached bytes and added
alongside the displayed equations; dependent publication bindings were regenerated. Follow-up
review found no remaining implementation blocker. Reader verification exercises real Chromium
PDF printing and file-origin offline operation on both portable pages. Expanded assumptions
are part of the aesthetics text measurement, not an unmeasured disclosure.
