# STRATUM Observatory 3.0 — delivered-artifact verification

Generated from the fresh engineering receipt and the adjacent gate reports in
`verification/reports/`. Every figure below is read out of one of those files;
`verification/tests/test_reported_numbers.py` fails if this document and the reports drift apart
again.

## Separate results

| Gate | Result | Exit | What actually ran |
|---|---|---:|---|
| Engineering | PASS | 0 | 617 verifier tests (10 skipped), 103 model/graph tests, two byte-identical builds of all eight portable entry points and 1,451 Chromium states over all nine checkpoints; the receipt binds the consumer inventory, the verifier identity, the scientific lock and the artifact |
| Accounting arithmetic | PASS | 0 | 399 live scenarios; 309 agreement, 82 explained differences, 186 unsupported findings, zero contradictions; required by engineering |
| Schematics | PASS | 0 | 757 scenes including all 544 decoder layers; 3,129 checks against core.mjs, of which 238 are recorded not-applicable passes |
| Presentations | PASS | 0 | 82 checks over 21 delivered states — exports, accessibility, table semantics, print, portability, preview freshness and preview coverage from a real file:// origin |
| Aesthetics | PASS | 0 | 182 scenarios, 427 checks; type, spacing, contrast, clearance, probe coverage and composition |
| Scientific publication | BLOCKED | 2 | Nine-model consumer inventory and strict evidence/review audit; **no FAIL** |
| Runtime certification | BLOCKED | 2 | Run on these bytes: 32 of 32 obligations NOT_RUN, **no FAIL and no `RUNTIME_STALE`**; all private hardware receipts were withdrawn for publication |

## Toolchain

CPython 3.13.12 and Playwright 1.55.0, both provisioned by uv from `uv.lock`.
OpenSSL 3.6.0 verifies review signatures. `uv sync --frozen` then
`uv run --frozen playwright install chromium` reproduces it. The interpreter, the lock, the
Python pin, `tests/`, `.github/CODEOWNERS` and the aesthetics policy are all inside
`toolchain_digest`, so a receipt cannot outlive any of them.

## Browser evidence

**1,451 rendered states**, including all **544 decoder layers** across nine checkpoints in both
label modes, **56 responsive viewport/lens/theme combinations** and **36 device-scale cases at
1.25 and 2**. **0 geometry defects, 0 runtime JavaScript errors, 0 external runtime requests.**
Model/layer/scenario controls, search, keyboard navigation, tracing, focus, zoom, URL restore,
SVG/XML rendering and JSON exports passed.

This result now reproduces under **macOS system fonts** as well as the Linux CI fallback, which
is meaningful because no fonts are bundled and every geometry check is built on measured
`getBBox()` bounds. The layout does not depend on a single font stack.

The layer matrix is now derived from the application's own model table rather than a literal in
`tests/browser.test.py`. The literal listed eight checkpoints and 504 layers, so when a ninth
shipped the suite went on sweeping the same eight and reported success over a matrix that no longer
described the artifact. A hardcoded inventory cannot notice what it omits.

**One of the figures above was wrong, and this record published it.** `device_scale_states` was
written as the literal `32` in a report dict whose every other entry is derived from what the run
did. The loop behind it is two device-scale ratios by every checkpoint by both themes, so the
true value became 36 when the ninth checkpoint shipped, and `tests/browser-cases.json` — written
by the same run — recorded `device-scale-1.25: 18` and `device-scale-2: 18` the whole time. The
receipt disagreed with its own case file, and the wrong number reached
`tests/browser-results.json`, the signed engineering receipt and this document.

Nothing caught it because nothing was looking: `verification/tests/test_reported_numbers.py`
reconciled aesthetics scenarios, schematics scenes, decoder layers and both receipt test counts,
and not one browser figure. The count is now derived from the recorded cases, all three browser
figures are reconciled against the receipt, and a separate test compares the receipt's
device-scale count against the case file the same run produced.

The lens and hardware axes are derived through the same door. `VIEWS` and the roofline SKU list
were literals in a file whose own docstring argues against literals; an eighth lens or a fifth
SKU would have been swept in no state while the suite's final assertion passed over the smaller
matrix. Both values are unchanged today — seven lenses, four SKUs — which is the point: a
derivation that reproduces the literal exactly is the only evidence that the literal was still
true.

It is not Safari/Firefox certification, all-system-font coverage, an exhaustive state-space proof
or GPU execution.

## Aesthetics evidence

`npm run verify:aesthetics` passes: **427 checks, 0 failures** across 182 scenarios and 120
distinct measurements. The first run of this gate failed on 21 defects that shipped in the
delivered artifact; all are now fixed at the source rather than by moving a threshold.

- **17 WCAG AA contrast failures**, worst `--subtle` on `--bg` at **3.28:1** against a 4.5:1
  requirement in the light "paper" theme, some of it inside the exported SVG. Five tokens were
  retuned; the measured ratios are now 4.63–6.96:1 and are pinned in a test, so the algorithm stays
  calibrated rather than merely passing.
- **4 rendered sizes below the floor** (`.source-role` 8px, `.status-pill` 8.5px). The chrome type
  scale removed the whole class: the smallest chrome text is now 10px.
- **24 ad-hoc font sizes became 8**, and 31 ad-hoc spacing values became one 2/4px rhythm. Both are
  now enforced statically, so they cannot silently return.

AA (4.5:1) is the floor everywhere. Reading-size **chrome** prose is held to AAA (7:1), and the
written exposition described below is inside that measurement rather than beside it: its cells and
citations were added to the probe's chrome selector list, so the 12px prose is measured at the 7:1
threshold (**14.19:1** dark, **12.21:1** light) and the 11px citations at 4.5:1. Canvas
labels stay at AA deliberately: they sit on the semantic tint fills that carry the colour grammar
of `docs/design/OBSERVATORY.md`, and forcing 7:1 there would wash that grammar out of the diagram.

Position must not depend on theme, and does not: composition is byte-identical between dark and
light in every scenario. All **182** measured scenarios were each compared against a recorded
signature and all match — the measured, the compared and the baselined counts are the same number,
which is the only way this check means anything. That sentence is now the gate's own PASS message,
computed from the compared set, rather than a property of this document.

It was twice not. The baseline first held 156 signatures covering the eight original checkpoints,
and the ninth checkpoint's 22 had no drift protection at all while the gate reported PASS over
every scenario it measured. Re-recording the baseline closed that instance and left the mechanism
intact: `aesthetics.py` asked `if label in recorded` for drift and
`all(recorded.get(l,s)==s for l,s in current.items())` for stability, which is vacuously true for
any label the baseline has never seen, so `COMPOSITION_STABLE` reported `len(current)` scenarios as
matching whether or not they were compared and `current - recorded` was covered by nothing. A tenth
checkpoint would have added unbaselined labels, left them uncompared, and reported the full
measured count as matching anyway. The
comparison is now a tested pure function: `COMPOSITION_NEW` FAILs on a measured scenario the
baseline has never seen — the mirror of `COMPOSITION_MISSING`, and FAIL for the same reason, that a
baseline stale against our own bytes is something we can know — and `COMPOSITION_STABLE` counts only
what it compared. The promotion of the current baseline was checked in both directions when it was
made: every one of the 22 new signatures belongs to a genuinely new `dsv41` scenario, and **not one
of the 156 existing signatures moved**, which rules out a shared change having leaked into the other
eight presets under cover of the new model.

The layer sample is deliberate and worth naming, because it is narrower than the other gates: this
gate measures decoder layers 1, 2 and 4, which for `dsv41` are two `swa` layers and one
`csa2-reuse` layer. The `csa2-full` and `csa2-reindex` block layouts are covered by
`verify:schematics` over all 544 layers and by the browser suite over all 544 in both label modes,
not by rendered-text measurement here.

Ratios are computed from specified colours against a resolved backdrop stack. That is an upper
bound on legibility, not a guarantee of it: anti-aliasing and stem weight are not modelled, and an
element whose backdrop cannot be resolved is reported BLOCKED rather than passed.

**The measured set is now bound to the page, and its gaps are recorded rather than implied.** The
probe used to select chrome text with a hardcoded selector string that nothing checked: renaming a
class removed it from measurement while the gate went on reporting PASS over a page it no longer
looked at — the same shape of defect this gate exists to catch, one layer up. Two static checks
close it. `CHROME_PROBE_LIVE` requires every class the probe measures by to still exist in
`src/styles.css` **and** in the markup that emits it, so a rename fails the gate instead of
disappearing. `CHROME_COVERAGE_OK` requires every reading-size chrome rule in the stylesheet —
all **43** between the 12px reading floor and the 24px large-text relaxation — to be either in the
measured set (**15** selectors) or in `UNMEASURED_CHROME` (**33**).

That second list is debt, not an exemption. It is what the gate does **not** measure, and probing
those selectors by hand finds **8 distinct reading-size chrome elements between 4.89:1 and 6.87:1**
against the 7:1 that reading-size prose is held to — `--muted` and `--accent` at 12px on
`--surface`, `--canvas`, `--bg` and `--accent-bg`, plus the 14px skip link. They are pre-existing
and are not fixed here: clearing them is a palette retune of the whole application, which does not
belong in a commit that adds an exposition. `UNMEASURED_CHROME` is inside `toolchain_digest`, so
adding a line to it invalidates every engineering receipt. The residual hole is named too: a new
element reusing a token/size pair already recorded there would be allowed, because the list is
keyed by selector rather than by measured ratio.

## The delivered page, in more than one state

`npm run verify:presentations` is the only suite that opens the artifact from a real `file://`
origin, and until now it measured exactly one state: `page.goto` lands on the default
`ds3`/`model`/`block` page and nothing navigated away from it. So the focal pages — which carry
prose and a data table that exist in no other state — had their headings, control names, table
semantics and citations checked by nothing, and the report's "9 headings ordered" was a true
statement about a different page.

`focal_states()` now reads the live model table and returns every checkpoint with an
encoder/decoder split, so the list cannot go stale the way `tests/browser.test.py`'s hardcoded
eight-checkpoint matrix did. The suite measures **2 states** and reports them in the receipt:
`default` (9 headings, 132 controls, 4 citations) and `dsv41/ced` (**10 headings ordered, 84
controls named, 1 table with scoped headers over 8 rows, 29 citations named**). `PROSE_SCOPE`
fails if no focal state was measured, so "we only saw the page we opened" cannot recur silently.

A new probe covers what a data table needs and no other check looks at: a real `<thead>`, `scope`
on every column header, a `<th scope="row">` per body row, and external citations that have an
accessible name and `rel="noopener"`. A five-column table of bare `<td>`s is perfectly legible,
passes every contrast and geometry check, and is unnavigable. Its negatives are exercised against
synthetic markup in a browser, not asserted.

## Diagram semantics

`npm run verify:schematics` checks what the diagram means rather than where its boxes landed. The
scene IR comes from the diagram builders and, per scene, `core.mjs` independently states what is
true; the two are compared. 757 scenes, 3,129 checks, 544 decoder layers, no browser. Of those
3,129, **238 are explicit not-applicable passes** — `SCENE_MODEL_SKIP` 213 and `MHC_SKIP` 25, both
counted in the report's own `passed_codes` — so 2,889 are applicable checks and the rest are a check
saying, in the report rather than by silence, that it did not apply here.

It also judges the causal encoder-decoder architecture itself, under 25 codes that did not exist
before. Two are worth naming because of what they compare rather than what they assert.
`CED_MODE_DISAGREEMENT` derives the 40-layer mode assignment twice -- once from the pinned
`kv_source_layer_ids` / `index_source_layer_ids` arrays, once generated from the technical
report's §4.2.1 group structure, which is given the depths and the group sizes and no layer ids at
all -- and requires the two to agree. `KV_BUDGET_DISAGREEMENT` re-derives main KV per token in
Python, 720 B over four Full-mode layers, and compares it against what `cacheLedger` computes in
JavaScript; `KV_RESIDUAL_IMPLAUSIBLE` then holds the published 890 B/token to implying between
0.5 and 1.0 bytes per indexer channel, which the reported figure does at 0.531. The bound is the
FP4-to-FP8 range and is deliberately not narrowed, because the report states no indexer precision
and a tighter bound would be a guess wearing a check's clothing.

Every judge in this gate reads only our own bytes, and none of them returns BLOCKED. That matters
because `aggregate` turns a single BLOCKED into exit 2 and this gate is one of the four the Pages
deploy requires, so a gate that could block would stop the page shipping for something that is not
a software defect. Three checks that legitimately block -- the publisher transcription, the
published active-parameter ratio, and claim-model coverage with nothing to compare -- were moved to
the scientific audit for that reason.

What enforces it, precisely. `verification/ced.py` has no schematics-domain BLOCKED constructor and
no keyword that reaches one: its `_one` helper takes no `status` parameter, so writing a
schematics-domain BLOCKED requires writing new code rather than passing an argument, and a test
pins that signature. `schematics.py`'s own `_ok`/`_bad` likewise construct only PASS and FAIL. The
judges the gate wires in are held in two exported lists, `SCHEMATICS_JUDGES` and `SCENE_JUDGES`,
which `run()` iterates and which `test_no_schematics_judge_can_emit_blocked` and
`test_no_scene_judge_can_emit_blocked` iterate too -- so wiring a judge brings it under the
property test without anyone remembering to list it. Those tests sweep 31 deliberately broken model
shapes against seven inventory states, including absent and degenerate ones, require every finding
to be PASS or FAIL, and require both outcomes to occur so the sweep cannot pass by exercising
nothing.

Two residues, named rather than papered over. The five judges defined inside `schematics.py`
itself are not in those lists; they are covered by that module having no BLOCKED constructor and by
`test_the_gate_report_carries_only_pass_and_fail`, which reads the real report's status counts.
And `aggregate` returns BLOCKED for an empty finding list, so a gate that produced no findings at
all would block; it produces 3,129 across 757 scenes, and the scene count is itself asserted.

## The written exposition

The CED page carries prose, not only a diagram: eight records naming what each published mechanism
replaces, how it works, one measured consequence and the bytes it is read from. Prose is where an
unsupported assertion is easiest to write and hardest to see, so two rules make it checkable rather
than merely well-written.

**No figure is a string.** Each record's `effect` is a *function* of the model, evaluated at render
time from `cacheLedger`, `servingLedger`, `parameterLedger` and `layerPartition`. A literal cannot
drift from the data because there is no literal to drift: `src/innovations.mjs` contains no
architectural number at all. `tests/core.test.mjs` asserts that every `effect` is a function, and
then mutates one field per record and requires that record's rendered sentence to move — a record
whose exposition ignored the field it is about would pass a type check while still being a literal
in disguise.

**The rule covers the authored sentences too, and it did not at first.** `replaces` and `mechanism`
are the only prose here, and they shipped a draft saying "the bottom half of the stack is a causal
encoder" — which asserts `encoderLayers === layers / 2` in a place nothing can check, directly
beside a derived figure that the mutation test rendered as "10 of 40". Another said "one E4M3 scale
per sixteen channels", duplicating `core.mjs`'s `1/16`, and named "FP4 E2M1" and "FP8" where
`kvFormat` and `swaKvFormat` already hold them. Those sentences now carry **no quantity at all** —
not a digit, and not a cardinal spelled as a word, which a digit scan alone would have missed — and
a test enforces both halves: the shipped prose is clean, and the four sentences the rule was written
for are rejected by it. The formats moved into `effect`, where they follow the model table.

**Every record cites sources that resolve.** `COVERAGE_UNCITED` requires at least one source id per
record and requires each one to resolve; an id that does not resolve renders as nothing, which reads
on the page as an uncited assertion rather than as an error.

**The exposition is checked against the computation, not against itself.** `COVERAGE_DERIVED` reads
the numbers out of the rendered sentence and requires figures derived outside the page to be among
them. Figures are compared as numbers, not matched as substrings, so a row writing 1,720 does not
satisfy an assertion about 720.

**5 of the 8 rows** are covered, and the gate's own PASS message says so rather than leaving the
scope to be inferred. The FP4 row is checked against `derived_main_kv_bytes_per_token`, the *same*
function `KV_BUDGET_DISAGREEMENT` holds `cacheLedger` to. The `ced`, `hier-indexer`, `mhc` and
`dspark` rows are checked against configuration fields that `FIELD_MAP` pins to JSON pointers
inside the publisher's own config document and that `CED_PUBLISHER_AGREEMENT` re-derives from those
bytes — a provenance chain that does not pass through `core.mjs` at all. The judge checks the
`FIELD_MAP` membership rather than trusting the claim, so a field that stops being pinned stops
counting as evidence.

The remaining **3** genuinely cannot be reached this way, and are recorded in
`UNPINNED_COVERAGE_ROWS` with the check that does hold each quantity: `csa2` renders
`layerPartition`, our own rule, held by `CED_MODE_DISAGREEMENT` and `CED_PARTITION_SUM`; `engram`
renders a `parameterLedger` output, held by `ENGRAM_TOTAL_DERIVATION`; `swa-replay` renders
`cacheLedger`'s bounded term, held by `KV_BOUNDED_INVARIANT` across two contexts. A second Python
implementation of any of those would compare two transcriptions of one rule instead of two
derivations. A test requires every row to be in one list or the other, so a row cannot be silently
unchecked.

The FP4 row is the one that needed this. The publisher's headline is **890 B/token**; the Full-mode
layers account for **720 B/token**; the **170 B** difference is indexer K at a precision the report
never states. The row prints all three, and `COVERAGE_DERIVED` fails if it prints only the
published total — which is the version a reader would have no way to question.

`PARAM_BAND_RATIO` closes the last of three implementations of the encoder/decoder
compression-ratio rule. `ratioOf` in `src/diagram-meta.mjs` is checked against the drawing by
`CED_BAND_RATIO` and `cacheLedger`'s copy by `KV_BUDGET_DISAGREEMENT`, but `parameterLedger`'s copy
was unchecked for a structural reason rather than an oversight: only `function.parameterLedger`'s
source hash reached the inventory, and a hash says a function changed, never whether it is right.
The ledger is now sampled twice — once at the declared band ratios, once with every ratio at 1 —
and the difference is exactly the compressor gate weights the ratio rule owns, **7,864,320
parameters** over three gated encoder bands at 5120 × 512, re-derived in Python. Turning
`l < encoderLayers` into `l <= encoderLayers` gates a fourth band and fails the check; removing the
condition gates none and fails it too. A zero-versus-zero comparison is treated as a failure, not
as agreement.

Three rules were removed or narrowed during authoring because the data showed they were not
contracts the IR makes: wire endpoints are semantic rather than geometric, wires fanning out from
one node legitimately share a trunk, and unwired nodes are annotation cards rather than defects.
Each was corrected rather than accommodated by widening a tolerance.

Row alignment holds in its strong form: `norm1` and `attention` sit at pixel-identical heights in
all nine presets, including the CSA2 block, whose `attention` box was deliberately authored at the
same 600/58 geometry the other eight use. Of 55 shared operations, 10 occupy a second row and every
one has a structural cause — Kimi's latent MoE inserts a stage, Flash-Next's QSA history sits
higher, and the 93/94-layer checkpoints need another row of layer tiles. A second row is permitted;
a third is not.

One over-assertion was found in inspector prose rather than in a node and is now gone: the
candidate-pool node told encoder Full layers 3, 9 and 15 that their indexer scores were "drawn from
the candidate set layer 21 establishes", which is both anachronistic and wrong in kind — the
report's Hierarchical Sparse Indexer is used only in the decoder of the causal encoder-decoder, so
an encoder scoring layer has no candidate pool and ranks the whole causally visible range. The pool
node and every sentence naming it are now gated on the layer reaching the candidate source, and a
test asserts that no unpooled layer states the pool's shape or claims a source layer supplies it.
Naming the pool in order to deny it is still allowed, because that is information a reader needs.

The gate also carries an mHC-specific invariant (`verification/ced.py`): a four-stream residual
drawn as a bare `+` passes every geometric check there is, so the residual rail must state the
stream count `core.mjs` declares and no `kind:'sum'` node may close the residual path. It is
judged on the 63 `dsv41` scenes and is a graph property, not a reviewer's impression.

`verification/ced.py` now also carries `CED_SCENE_TOPOLOGY` for the encoder/decoder page. That page
is the one view whose whole subject is a relation rather than a box, so the relation is checked:
`tools/export-scenes.mjs` reports the 40 per-layer kinds `core.layerKind` produces and nothing
else, `ced.py` re-derives the band runs from them in a second implementation, and the drawing is
judged against that — the bands must tile layers 1–40 exactly, exactly one edge may run from
`enc-out` into `dec-full`, no edge may run from a decoder node back into an encoder node, the
candidate pool must feed exactly the four decoder-band Reindex layers and be established by exactly
one source, and the Engram and DSpark nodes must name every layer they claim to reach. A back edge
would make this an ordinary shared-KV Transformer, and a caption saying otherwise is not a check.
Ten negative tests in `verification/tests/test_ced.py` show each of those can actually fail.

## Delivered artifacts

`npm run verify:presentations` is the only suite that opens the artifact from a real `file://`
origin — every other suite injects with `set_content()`, so the path a reader takes had never been
exercised. Its 33 checks validate the exported SVG and the exported scenario against a schema the
page has declared all along with nothing defining it, check accessible names, heading order and
visible focus, emulate print and reduced motion, and consume `previews/manifest.json` so previews
cannot silently depict an artifact that no longer ships.

The artifact loads nothing from an external origin. Its 23 external URLs are arXiv and vendor-doc
citations the reader clicks, which is the provenance feature, not a portability defect.

**A fresh preview set is not a complete one, and this gate asked only the first question.**
`tests/capture.py` built its case list from a hardcoded eight-model literal, so no published
preview depicted the ninth checkpoint in any state — not its decoder block and not its
encoder/decoder page — while `PREVIEW_FRESH` compared one artifact digest and reported the previews
that did exist as current. A manifest depicting eight of nine checkpoints is exactly as fresh as one
depicting all nine.

Closing that left the same defect one layer in, and this document overstated the fix. It said the
capture script reads the focus segments out of the rendered page, so its cases are the
application's own inventory. It read them and then discarded them: `if stage=='ced'`. So the only
focal pages ever published were the encoder/decoder ones, and `dsv41/model/residual` — the mHC
4-stream residual this checkpoint exists to draw — was in no preview, nor were
`dsv41/model/inputs`, `q38flash/model/inputs` or `q3827/model/inputs`. `judge_preview_coverage`
asked for a decoder block plus, where applicable, a `ced` page, so it reported PASS over a
requirement written to the same literal, and `focal_states` emitted the same single stage, so
those three pages were measured by this gate in no state at all.

`core.focalStages` is now the one derivation of which pages a checkpoint offers, and the capture
script, this gate's coverage judge, its focal-state list, both exporters, the browser matrix and
the aesthetics sweep all read it. `PREVIEW_COVERAGE` requires every model in the live table and
every focal page that model offers; `PROSE_SCOPE` names the states it measured. **All nine
checkpoints now appear in the 33 published previews, including every one of the 9 focal pages they
offer**, and this gate measures **21 delivered states**: Foundations, dedicated entry states, and every focal page in
the application.

**The manifest's own per-file digests had never been read, and the coverage question was only ever
asked in one direction.** `tests/capture.py` records an `svg_sha256` for every capture; across
every `.py` and `.mjs` in this repository there was exactly one reference to that key — the line
that writes it. `PREVIEW_FRESH` compares a single artifact digest and `PREVIEW_COVERAGE` counts
models, so a published SVG that had been hand-edited, truncated or left behind by an older build
satisfied both while the manifest recorded precisely what was wrong with it. `PREVIEW_INTEGRITY`
now verifies each published SVG against its recorded digest: **30 files, each hashing to what the
manifest attests**, and a comparison count of zero reports BLOCKED rather than a pass.

The reversed direction was worse, because it had no check at all. `PREVIEW_COVERAGE` walks the
manifest asking whether each model appears; nothing walked `previews/` asking whether each
published file is accounted for. Four files were not: `mobile.png`, captured deliberately but
recorded in no manifest entry; `test-export.svg` and `test-scenario.json`, download scratch the
browser suite wrote into the published directory; and `observatory-initial.png` and `overview.png`,
the first written by a scratch script with no gate behind it and the second produced by nothing at
all — last touched at the 3.0.0 baseline, so it depicted an eight-model page for three releases in
the directory this project publishes as evidence. `PREVIEW_INVENTORY` now reports any published
file no capture record names. The mobile capture is recorded (with a viewport and a state, and no
`svg_sha256`, because it exports no SVG — an absent digest, never a fabricated one), the scratch
writes go under `tests/`, and the two orphans are gone.

This is the same shape of defect as the browser matrix that swept eight checkpoints, the
composition baseline that held signatures for eight, the chrome probe whose selector list omitted
the new page, the focal-state list that measured only the page it opened, and the claim manifest
that carried eight model contracts for nine models — each recorded above or below. Every one was
found on this branch, which makes it a class rather than an incident: a check written against a
fixed set of eight things reports success when a ninth arrives. The remedy that works is the same
each time — derive the set from the live model table, and make some check assert coverage rather
than only freshness.

## The contract inventory covered eight of nine checkpoints

The tenth instance of that class, and the one in the most load-bearing artifact here.
`observatory.json` carried exactly one `model.*` contract per model — `model.ds2` through
`model.kimi-k3`, eight of them — and eight `model`-domain runtime obligations, and both stayed at
eight while 482 claims for a ninth checkpoint were authored. This document reported
`Domain contracts / runtime obligations | 31 / 31` over nine checkpoints, and the publication
report showed eight `CONTRACT_CONSISTENT | model` passes. DeepSeek-V4.1-Flash — the only model here
with four attention modes, two separately-quantized KV caches and a 4-stream residual — had no
contract record to be internally consistent with and no obligation to be NOT_RUN against. Its
absence was indistinguishable from having nothing to prove.

Nothing in the audit could report it. `RUNTIME_MATRIX` builds its required set from
`{gpu, scale_up, scale_out, communication}` and excludes the `model` domain. `RUNTIME_COVERAGE`
asks only that each *domain* carry at least one obligation, and `model` carried eight.
`DOMAIN_UNCOVERED` fires only on a domain with zero contracts. And no test asserted that a *model*
has a contract at all — `CLAIM_MODEL_COVERAGE`, added on this branch, is exactly this enforcement
one level up and was never applied to contracts.

Both halves are closed. `model.dsv41` and `runtime.model.dsv41` are authored in the shape of the
existing eight: the record's `layer_groups` are derived from `layerKind` over all 40 layers and
partition `1..40` exactly once across `swa`, `csa2-full`, `csa2-reindex` and `csa2-reuse`; every
scalar is *bound* to its `MODELS.dsv41.*` input rather than transcribed; `latent` equals `d`,
exactly as it does for the seven other presets with no compressed expert path. Nothing was invented
to make the record parse. And `judge_contract_model_coverage` now requires every model in the live
inventory to have both a model-domain contract and a model-domain obligation, reading the
association off each contract's own bindings and claims rather than off a naming convention —
`model.kimi-k3` speaks for `MODELS.kimi3`, so a check keyed on `f'model.{mid}'` would have failed
on the wrong thing. It reports FAIL, not BLOCKED: a model our own exporter emits with no record
beside it is a wiring defect we can know about. Like its two neighbours it runs in the publication
audit rather than in a Pages-deploying gate, because its zero-comparison path is BLOCKED.

The honest consequence is a slightly worse-looking report: **32 contracts and 32 obligations**,
one more `CONTRACT_EVIDENCE` BLOCKED, two more `REVIEW_REQUIRED` BLOCKED, and **32 of 32
obligations NOT_RUN** after private hardware receipts were withdrawn. That is the point. The nine
`CONTRACT_MODEL_COVERAGE` passes are the only new PASSes, and they assert coverage of the
inventory, not correctness of anything in it.

## Scientific inventory

| Item | Count |
|---|---:|
| Input leaves | 587 |
| Function/file identities | 32 |
| Distinct sampled scene contracts | 2883 |
| Claims including domain obligations | 3525 |
| Candidate source entries | 76 |
| Accepted raw-byte snapshots | 74 |
| Scientifically certified claims under the signed policy | 0 |
| Passed presentation bindings | 0 |
| Domain contracts / runtime obligations | 32 / 32 |

The publication audit has **6,120 BLOCKED findings, 38 PASS and no FAIL contradictions**, read
from the current source and consumer bytes with a fresh engineering receipt:

| Code | Status | Count |
|---|---|---:|
| `REVIEW_REQUIRED` | BLOCKED | 3,557 |
| `CLAIM_ANCHOR` | BLOCKED | 2,334 |
| `CLAIM_ROLE` | BLOCKED | 89 |
| `MODEL_DUAL_CODE` | BLOCKED | 77 |
| `CONTRACT_EVIDENCE` | BLOCKED | 32 |
| `CONTRACT_CONSISTENT` | PASS | 23 |
| `CONTRACT_MODEL_COVERAGE` | PASS | 9 |
| `DOCUMENTED_NO_SOURCE` | BLOCKED | 16 |
| `DOMAIN_INPUT` | BLOCKED | 9 |
| `SOURCE_UNFETCHED` | BLOCKED | 2 |
| `SOURCE_CACHE` | BLOCKED | 2 |
| `CLAIM_SOURCE_BLOCKED` | BLOCKED | 2 |
| `RECEIPT_UNATTESTED` | BLOCKED | 1 |
| `ENGINEERING_CURRENT` | PASS | 1 |

Most of these are cascading source dependencies and absent independent reviews; this is **not** a
claim that 6,120 facts are false. No review identity, accepted source hash or execution receipt was
invented.

For the record, the evidence-anchor work began at **9,037 findings (9,013 BLOCKED)** with
`CLAIM_ANCHOR` at **5,899**. Anchoring configuration citations as resolved JSON pointers, anchoring
paper and vendor-document citations as unique quotes, dropping 148 citations that were false
provenance, and letting derived scene claims carry provenance through their dependencies together
removed 3,853 BLOCKED findings without a single fabricated anchor.

Runtime certification was run on these bytes and reports the same population plus its
obligations: **6,113 BLOCKED, 41 PASS, 29 NOT_RUN, no FAIL.**

**Adding a ninth checkpoint did not make any receipt stale, and an earlier draft of this section
was wrong to predict that it would.** The prediction came from reading the coupling rule instead of
running the gate. Both obligation classes are immune to a claim-manifest edit, for two different
reasons: the thirteen hardware-attribute obligations deliberately do not bind `lock_sha256` at all,
and the eighteen behavioural ones have no receipt to retire — they are `RUNTIME_NOT_RUN`, and
`runner.py` reports a missing experiment before it ever reaches the staleness branch. So the gate
shows **zero `RUNTIME_STALE`** findings. The mechanism is real and is described under *Known and
unclosed*; it fires when a manifest edit invalidates an obligation that actually holds a receipt,
which is not the case here.

Private hardware captures and their receipts are withheld from the public distribution.
No redacted capture is accepted as evidence. All runtime obligations remain NOT_RUN until
new shareable evidence is captured and independently reviewed.

**Verified scientific claims: 0.** Two claims previously passed — the stylesheet and the HTML
entry point — because `presentation` claims skipped review entirely. Every claim is now reviewed,
so those green cells are gone. Calling a user-visible value presentational is itself a judgement,
and it is the judgement most worth checking.

**Qwen presets have no publisher implementation, and that is now resolved rather than blocked.**
`huggingface/transformers` was previously registered as `publisher_code`, so `transformers` plus
`sglang` satisfied "publisher implementation AND independent framework implementation" using two
third-party reimplementations and no publisher at all. transformers is now `framework_code`, and
dual-code independence compares organizations rather than repository names. Under the recorded
`evidence/publisher-code/audit.json` exception described below, the Qwen checkpoints meet the
corroboration requirement through independent implementations from three organizations. For the
eight Qwen, DeepSeek V2/V3 and Kimi presets, `MODEL_DUAL_CODE` and `CLAIM_ROLE` are **0** findings.

**DeepSeek V4.1 now has a pinned independent serving implementation.** SGLang's
`dsv4.1` development branch at `ac1a56987c5c552f9f4284f5a98db5d65541fafa` includes an
explicit V4.1 configuration adapter, guarded model execution and cache layout. The adapter
maps V4.1 to the shared `deepseek_v4.py` entry class; filename similarity alone is not the
identity proof. Three new source snapshots and exact AST anchors cover the added cache
representation and serving explanation. Full claim-by-claim corroboration and signatures
remain incomplete; source availability does not close the publication gate.

## A correction to this project's own reasoning

While making derived scene claims carry provenance through their dependencies, this project claimed
the change left the gate harder to fool than it had been before. **That was wrong, and the claim is
withdrawn.**

The anti-laundering property — a derived claim cannot reach PASS while any dependency is
unverified — was already carried by the **dependency fixpoint**, which predates the change and is
untouched by it. In isolation the new role check is **more permissive** than the one it replaced: it
lets a role be satisfied by a dependency's referent rather than demanding the claim carry its own.

What the change actually bought is narrower and worth stating exactly: it made it possible to
remove **3,561** copied citations from derived scene claims — entries that were duplicates of their
inputs' citations with the unanchorable ones left null — **without manufacturing false role gaps**.
Deleting those copies under the old rule would have produced `CLAIM_ROLE` findings describing
missing evidence that was never missing. That is a gain in truthfulness of the report, not a gain
in resistance to attack.

## What the remaining `CLAIM_ANCHOR` findings are

2,334 evidence entries name a source but no selector within it. They are not oversights, and
`docs/verifiers/ANCHOR_RESIDUE.md` records a decision for all **3,125** residue rows across **144**
classes — re-pointed, dropped, reclassified, or explicitly unresolved with a stated reason.

**1,586** of the 2,334 are genuine citations on derived `scene.*` claims that no declared dependency
supplies, kept because deleting them would strip 360 claims of their only evidence for a role their
domain requires:

- **856** GPU vendor-document citations across **214** claims — the `h100`, `h200`, `b200` and
  `gb300` product pages.
- **730** communication citations across **146** claims — `nccl-api`, `nvshmem-api`, `nixl-api`,
  `nccl` and `nvshmem`.

They stay, and they keep reporting BLOCKED, which is the correct fail-closed report: the citation is
true provenance for what the diagram draws, and it is still not a selector.

The ninth checkpoint ends this count at 2,334 against 2,335 before it, and the near-zero net
movement is two different things cancelling. Nineteen citations gained a selector and four were
dropped; then an audit of **every one of the 89 `python_symbol` anchors on `MODELS.dsv41.*`**
removed 20 that resolved against spans which do not carry the value they were cited for, and
re-pointed 6 onto spans that do. Removing a false anchor raises `CLAIM_ANCHOR`, and that is the
right direction: an anchor whose bytes do not state the claim is worse than no anchor, because it
reports as evidence. What was anchored, each recorded in `ANCHOR_RESIDUE.md`:

- **Configuration pointers (3).** `compressEncoder` and `compressDecoder` as indexed
  `compress_ratios` pointers — the publisher documents that field as one compression ratio per
  layer — and `swigluLimit`, whose relation stays `documents` rather than `equal` because upstream
  holds `10.0` against the displayed `10`: the same number, not the same bytes.
- **Model-card quotes (6).** `total`, `active`, `activeDecode`, `kvBytesPerToken` and `kvFormat` on
  the card's own sentences, and `encoderLayers` on "a 40-layer Transformer organized as a 20-layer
  causal encoder followed by a 20-layer decoder".
- **Publisher-code symbols (10).** `EngramLayout` for the five Engram fields it reads,
  `hc_split_sinkhorn_kernel` for `streams` and `sinkhornIters`, `fp4_act_quant` for `kvFormat`, and
  `act_quant` plus `Attention._window_kv` for `swaKvFormat`. Each digest was resolved by
  `verification.sources acquire` against re-fetched bytes whose digests equalled the accepted ones.
- **Dropped (4).** The configuration citation on `id`, `name`, `family` and `sources`, which no
  document states.

The audit's own table is in `ANCHOR_RESIDUE.md`, per symbol. The largest corrections: `Gate`
carried `expertFF`, `ff`, `shared` and `denseLayers`, though the router computes no FFN width and no
expert count — `MoE` does, and asserts `n_shared_experts == 1` outright; `Aligner`, a two-layer
projector, carried the vision layer count, head count, FFN width and patch size, which belong to
`ViT`, `MLP` and `PatchEmbed`; `Transformer` carried `architecture`, `generation`, `repo` and
`summary`, which are our own labels and prose, and `maxContext`/`configMax`, where the span holds
`max_seq_len` — a runtime cache cap — rather than `max_position_embeddings`. Three absence or
derived claims (`qkNorm`, `swaLayers`, `ngram`) lost anchors for the same reason the configuration
citations on such claims never had them: an absence read off a span that merely lacks the mechanism
is an inference, and a product no span states is arithmetic. Six anchors moved rather than
disappearing, onto the span that does carry the value: `indexSourceLayers` and `kvSourceLayers` to
`Attention`, `dsparkExperts` and `dsparkTopK` to `ModelArgs.get_moe_config`, `visionFF` to `MLP`,
`visionPatch` to `PatchEmbed`. `swigluLimit` was the first instance of this class found and it
moved the same way, from `Gate` to `Expert`.

Those 63 standing anchors, and the 6 re-pointed ones, now carry `contains` tokens — the attribute
or key whose presence in the span is what makes that symbol the right symbol, checked by
`select_anchor` at resolution rather than by a reviewer. All **69 resolved, 0 refused**, and the
false binding this class kept producing is now unrecordable: `Gate` with
`contains: ['n_activated_experts']` resolves, `Gate` with `contains: ['args.moe_inter_dim']` is
refused. `contains` is mandatory for `span` and `text_quote` anchors and optional for
`python_symbol`, which is the asymmetry that allowed it; closing that in `select_anchor` would
refuse the **5,837** `python_symbol` anchors on the other eight checkpoints until each is authored,
so the figure is recorded here and the decision is not taken silently.

One non-move belongs in the record too. `encoderLayers`' configuration and code citations stay
unanchored: `config.json` marks a boundary at layer 20 but never calls it an encoder/decoder split,
and `Transformer.__init__` builds one flat `nn.ModuleList`. Only the card states the split, so only
the card carries the anchor.
`CLAIM_ROLE` and `MODEL_DUAL_CODE` remained blocked in that earlier audit because no exact
framework source had been pinned. The SGLang addition below supersedes that source inventory.

The remaining 748 sit on `MODELS.*`, `HARDWARE.*`, contract records, functions and files, and each
falls inside a class in `ANCHOR_RESIDUE.md`. One of them was added deliberately:
`MODELS.dsv41.encoderLayers` used to cite `dsv41-code`'s `Transformer` with relation `implements`,
which was false — `Transformer.__init__` builds one flat `nn.ModuleList` and encodes the 20/20
encoder/decoder split nowhere. That anchor is still `null` and the configuration citation has none
either, so both keep reporting `CLAIM_ANCHOR` BLOCKED; the field is anchored only on the model card,
which states the split in words. An honest BLOCKED beats a pointer that resolves against bytes which
do not say what it claims, and a source that does say it is not a substitute for the two that do
not.

## Two discrepancies between our data and its sources

Found while authoring anchors, recorded rather than silenced, and both left visible in
`ANCHOR_RESIDUE.md`:

- **Vocabulary.** The Qwen3 report states a **151,669**-token vocabulary; the app displays
  **151,936**. Both are right and they measure different things: the report's figure is the
  tokenizer's BBPE vocabulary, while 151,936 is `vocab_size` in the shipped configuration — the
  padded height of the embedding matrix, which is the number that costs parameters and HBM. The app
  anchors the padded figure to the configuration with an `equal` pointer, and the paper citation on
  those three claims was **dropped**, because leaving it would imply the report corroborates a
  number it never mentions.
- **Context length.** The Qwen3 report's tables give Qwen3-8B **128K** context; the app displays
  **32,768**. Native versus extended: the app names "the native context used by this preset", and
  the report states 32,768 as the pre-training sequence length for all models, reaching 128K
  afterwards with YaRN and Dual Chunk Attention. The shipped configuration carries a third number
  again, `max_position_embeddings` = 40,960. The three affected citations were **re-pointed** onto
  the report's native-context sentence.

A third, smaller mismatch is recorded the same way: `MODELS.q3827.linearGate` displays `SiLU` where
Qwen's configuration says `"swish"`. They are the same activation under two names (SiLU is
Swish-1), so the pointer is a true anchor and it stands — but its relation was deliberately left
`relation: documents` rather than upgraded to `equal`, because `equal` compares upstream bytes with
the displayed value and `"swish"` is not `"SiLU"`. Upgrading it would make the gate assert something
false in order to look greener.

## Known and unclosed

Recorded so they are not mistaken for oversights:

- **Build attestation is not present in the local receipt.** `RECEIPT_UNATTESTED` remains
  BLOCKED. The public snapshot excludes private hardware captures; configuring public CI
  and its attestation remains a separate release action.
- The engineering receipt is **self-produced**. This is now a finding rather than a sentence:
  `RECEIPT_UNATTESTED` BLOCKS on any receipt without a build attestation. CI attests the exact
  tested artifact; branch protection remains the repository owner's to configure, and
  `.github/CODEOWNERS` has no force without it.
- **PDF sources cannot currently be pinned at all.** `verification/sources.py:30` and
  `verification/evidence.py:140` both call `raw.decode('utf-8','strict')` before any other check, and
  every anchor kind in `select_anchor` operates on decoded text lines, so a PDF is rejected at
  acquisition rather than at anchoring. Supporting one honestly would need the PDF itself pinned as
  the source of record, an extracted-text artifact whose digest is bound to the PDF's, the extractor
  inside `toolchain_digest`, proof that the extraction is byte-deterministic, and a new
  `pdf_text_quote` anchor kind — and even then the anchor would prove a fact is in *our extraction*,
  not in the PDF. Three arXiv papers were re-pinned to full-text HTML instead. Any future paper
  without an arXiv HTML rendering forces this decision rather than deferring it.
- **A cap in the quote-authoring tool.** `QUOTES` in `tools/anchor-documents.py` keys on
  `(source, field)` and not on model, so a source cited by several checkpoints can only carry facts
  that are identical across all of them — `q3-paper` backs q38, q330 and q3235 at once,
  `hopper-tuning` backs h100 and h200, `blackwell-tuning` backs b200 and gb300. This is why
  per-model paper facts stayed unanchored: layer counts, hidden sizes and expert counts for
  individual Qwen models. 52 residue rows sit on such a shared key; re-keying on
  `(source, model, field)` is estimated to unlock roughly **24** of them and to turn the other 28
  into an honest "this source does not state this figure". Recorded, deliberately not fixed.
- **A stale receipt must be re-probed rather than re-stamped.** For a **behavioural** obligation
  `lock_sha256` still binds the receipt to the target and source manifests; five manifest edits
  during the anchor work moved past the lock the three GPU receipts were captured under, and
  `verify:runtime` correctly turned into three `RUNTIME_STALE` FAIL findings — a gate failure, not a
  bookkeeping nuisance. The repair was to re-probe all three hosts (H100-SXM, B200, GB300) read-only
  on the private fleet and rebuild the receipts. Every attested value came back byte-identical, but
  that was an outcome of the probe, not a licence to skip it. Editing a recorded digest in place
  would have been fabrication. That remedy is unchanged; what changed is which edits demand it.
- **Hardware receipts bind the hardware, not the claim manifest.** Previously every receipt bound
  `lock_sha256` and `consumer_sha256`, so a cosmetic diagram edit retired the three GPU receipts and
  required re-probing real hardware — which happened three times while relabelling one wire. That
  coupling was too broad in one direction only: an H100's HBM capacity does not depend on our claim
  set or our source files. Obligations in the hardware-attribute domains (`gpu`, `scale_up`,
  `scale_out` — 13 of 32) therefore bind
  - `probe_schema_sha256`, the hardware data model in `verification/schemas/hardware.schema.json`
    (mismatch → `RUNTIME_SCHEMA` **BLOCKED**: a superseded model is missing evidence, not a
    contradiction);
  - `contract_sha256`, the contract record with the application's own bindings applied (mismatch →
    `RUNTIME_CONTRACT_DRIFT` **FAIL**: the contract would assert something the probe never saw);
  - `hardware_sha256`, the attested half of the capture the receipt names in `hardware_artifact`.
    The gate **re-derives** it from those bytes, which are themselves among the receipt's checksummed
    artifacts, so a hand-written digest is `RUNTIME_HARDWARE_DRIFT` **FAIL** rather than an unchecked
    claim; an absent field or an unpinned record is `RUNTIME_HARDWARE` **BLOCKED**.

  The other 19 obligations attest what our code *does* and keep binding the lock and the consumer
  inventory exactly as before; `verification/tests/test_runtime_binding.py` proves both directions,
  one obligation at a time. **A re-probe is now required when the hardware data model changes, when
  the hardware contract record changes, or when the receipt expires** — not when a claim, a diagram
  or a source file changes. This was demonstrated rather than asserted: with the three receipts
  captured and promoted, editing an unrelated claim's `statement` in `observatory.json` left
  `verify:runtime` at exit 2 with the same finding counts and zero `RUNTIME_STALE`,
  `RUNTIME_SCHEMA` or `RUNTIME_CONTRACT_DRIFT` findings.
- **The expiry window is kept for hardware receipts, deliberately.** Nothing in the gate can detect
  that someone re-capped a GPU's power limit, reflashed a NIC or swapped a card between two probes
  of the same host: every digest would agree, because each only describes what the last probe saw.
  Expiry (30 days; 7 for `measurement`) is the only mechanism that forces another look, so
  decoupling these receipts from the manifest is not a reason to drop it.
- **Five `--query-gpu` field names in the probe are not implemented by any driver on the fleet.**
  The comprehensive probe was authored without hardware to test it against. On all three hosts
  (driver 535.161.08 and 580.105.08) `nvidia-smi` refused `pci.link.gen.max`, `pci.link.width.max`,
  `clocks.memory`, `pci.link.gen.current` and `pci.link.width.current` — it refuses the **whole**
  query over one unknown name. The probe now keeps each refusal as a checksummed artifact, drops
  exactly the names the driver refused, asks again, and records each dropped field in the capture's
  `unavailable` list with the driver's own sentence. Nothing was substituted from a datasheet, and
  the field names were left as authored rather than quietly corrected, so the refusal stays visible
  in the evidence. (`pcie.link.*` and `clocks.mem` are accepted by these drivers; adopting them is a
  deliberate re-probe, not a silent edit.)
- **`driver.attached_gpus` is a container-visible count, not a property of the machine.** It is read
  from `nvidia-smi -q` inside whatever cgroup the probe ran in, so `NVIDIA_VISIBLE_DEVICES` or a
  device-restricted container changes it — and therefore changes `hardware_sha256` — without
  anything changing on the physical host. The recorded counts (8 for the H100 and B200 hosts, 4 for
  the GB300 host) describe what the probe could see. The SKU-scoped counts the contracts assert
  (`gpu_count` 1, or 2 for a GB300 superchip) are deliberately independent of it.
- **Five contracts remain stubs** — training, serving, numerics, storage and measurement. Each now
  states in the manifest exactly what it needs. Fabricating shard digests or tokenizer revisions to
  make a record parse would be strictly worse than an honest stub.
- **Three communication contracts describe observed behaviour**, so they need a runtime receipt
  rather than a binding, and say so.
- **Qwen ships configuration and no modeling code**, verified against the Hugging Face model API
  and recorded in `evidence/publisher-code/audit.json`: none of the five Qwen checkpoint
  repositories contains a single `.py` file, while DeepSeek V2/V3 and Kimi K3 do. Demanding a
  publisher implementation for those checkpoints would block them forever rather than describe
  anything true.
  For them the corroboration requirement is met instead by **independent implementations from at
  least two different organizations** — `huggingface/transformers`, `vllm-project/vllm` and
  `sgl-project/sglang` — each anchored to the same architectural symbol so the citations are a
  comparison rather than a list. This is a narrow exception keyed to the recorded audit, not a
  general relaxation: a publisher that does ship code still cannot be corroborated by two
  reimplementations, and two repositories of one organization never suffice.
- **Two sources remain unfetched** — `ib-opensm` and `dynamo-nixl` — because their URLs redirect and
  the acquirer refuses to follow a redirect without inspecting the new origin. Pin the resolved
  origin; do not relax the guard.
- **`REVIEW_REQUIRED` is 3,552 findings** and is independent of everything above.
  No amount of source acquisition or hardware access reduces it: it needs a reviewer identity in
  the protected trust root and an independent signer who is not the author.
- **Independent V4.1 source is available, but full corroboration is incomplete.** The pinned
  SGLang development branch supplies the exact V4.1 dispatch and cache implementation.
  Remaining `MODEL_DUAL_CODE` and `CLAIM_ROLE` findings require per-consumer evidence;
  a shared source file cannot automatically certify every model claim.
- **The indexer term is now independently derived.** SGLang's packed indexer pool stores
  64 payload bytes plus 4 scale bytes per compressed token. Source ownership and compression
  yield 170 B/input token; together with 720 B/token main KV this reconciles the reported
  890 B/token. The new `KV_INDEXER_DISAGREEMENT` and `KV_TOTAL_DISAGREEMENT` checks compare
  both the component and exported combined subtotal. This is complete-pair amortized payload;
  window state, pending compression pairs, pages and allocation reserves remain separate.
- **No DSpark parameter count is published.** The configuration pins the draft block size, the
  draft expert count and the top-k; no document states the module's weights. The inspector says so
  on the page — the count is unpublished and excluded from every weight total in this atlas — and
  `parameterLedger` contains no DSpark term. Inventing one to make the published parameter total
  reconcile is the failure this bullet records rather than commits.
- **The technical report cannot be byte-pinned, so nothing is anchored to it.**
  `DeepSeek_V41_Tech_Report.pdf` is displayed as a source in the application and is deliberately
  absent from `verification/manifests/sources.json`: `sources.py` decodes strictly as UTF-8, so a
  PDF is refused at acquisition rather than at anchoring, as the PDF bullet above describes. Its
  `§2.2 / §2.3 / §4.2.1` anchor in `SOURCES` is a pointer for a reader, not evidence a gate can
  check, and the report's own §4.2.1 group structure reaches a check only through the independent
  re-derivation in `CED_MODE_DISAGREEMENT`.
- **`CED_PUBLISHER_AGREEMENT` cannot run in CI.** It needs the acquired upstream bytes, and
  `.verification-cache/` is git-ignored by design, so CI holds none and the check reports BLOCKED
  there — naming the acquire command, never a vacuous PASS and never a comparison count of zero
  reported as agreement. The transcription comparison therefore has local evidence only, which is
  why it lives in the publication profile rather than in a software gate.
- **Eight reading-size chrome elements are below the AAA floor the record holds prose to.** Probing
  the `UNMEASURED_CHROME` list by hand finds **8 distinct elements between 4.89:1 and 6.87:1**
  against 7:1 — `--muted` and `--accent` at 12px on four backdrops, plus the 14px skip link. They
  are pre-existing and unfixed here: clearing them is an application-wide `--muted`/`--accent`
  retune, not an edit that belongs in the commit that adds a checkpoint. They are recorded debt
  inside `toolchain_digest`, not an exemption.
- **`contains` is still optional for `python_symbol` anchors.** Making it mandatory in
  `select_anchor` is what would make a symbol cited for a value its span does not carry
  unrecordable rather than merely findable by audit — an audit removed 20 such bindings on this
  checkpoint. Requiring it now would refuse
  the **5,837** `python_symbol` anchors on the other eight checkpoints until each is authored, so
  the figure is recorded and the decision is deliberately not taken silently.
- **61 `dsv41` citations remain unanchored** — 30 on the five publisher-code files, 26 on the
  configuration and 5 on the model card — each inside a class with a recorded decision in
  `docs/verifiers/ANCHOR_RESIDUE.md`. They are `CLAIM_ANCHOR` BLOCKED, which is the correct report:
  an absence read off a span that merely lacks the mechanism is an inference, a product no span
  states is arithmetic, and our own labels and prose are not in any publisher document at all.

## Transcription against publisher bytes

`MODELS.dsv41` is a hand transcription of the publisher's `config.json`, and transcription error
is the dominant failure mode when adding an architecture. `CED_PUBLISHER_AGREEMENT` re-derives
54 of its 81 fields from the acquired publisher bytes -- 45 by JSON pointer, 9 by an expression
over the same document -- and compares them, with 0 unresolved pointers and 0 disagreements. The
other 25 fields are recorded with the reason each has no publisher counterpart, and
`CED_FIELD_UNMAPPED` fails on any field in neither list, so adding a field to the model table
forces a decision about how it gets verified.

It runs in the **publication** profile (`npm run verify:science`), not in a software gate. The
check needs the acquired upstream bytes; `.verification-cache/` is git-ignored by design, so CI
holds none, and it reports BLOCKED naming the acquire command wherever they are absent -- never a
vacuous PASS, and never a comparison count of zero reported as agreement. It also blocks on
*partial* agreement: a mapped field that stops resolving, through an upstream rename or a
truncated document, is a field nothing compared, and reporting the fields that did resolve as
agreement would be the same defect in smaller print.

Two further checks live there for the same structural reason rather than for want of a software
home. `CED_ACTIVE_RATIO` compares the published 8B/16B actives against the 20/40 depth split and
reports BLOCKED, not FAIL, on a mismatch, because the actives are published rounded to whole
billions and the embedding and Engram terms are depth-independent. `CLAIM_MODEL_COVERAGE` requires
every claim naming two or more models to name all nine -- 25 do -- and blocks if it finds no
multi-model claim at all, since zero comparisons is not agreement. Both have an honest BLOCKED
outcome, and a *software* gate returning BLOCKED would stop the page shipping, which is precisely
what the amended Pages contract above exists to prevent. Publication is already BLOCKED offline,
so that is its expected state there. No local coverage is lost: every local `verify:science` run
executes all three, and CI could never have executed the transcription comparison either way.

## Artifact identity

Portable HTML SHA-256: `2b3a2c86a0376a614e8c784e78da44f4197f34349ea2f26ae07fd07e6f5bd05c`

Scientific lock SHA-256: `a1b0ace724bf4faa166505e066c1510c835a311755323fbe6297ea80c54bb890`

Verifier identity SHA-256: `dd1b225e750cff45a13e661becff7e98e85db0a6da830218c2108f6715fe4834`

Complete command logs and receipts: `verification/reports/`. Full browser matrix:
`tests/browser-cases.json`. Re-run after changing application, verifier or evidence; old receipts
are not transferable.

## Reproduce

```sh
uv sync --frozen
uv run --frozen playwright install chromium
npm run verify:schematics    # exit 0
npm run verify:aesthetics    # exit 0
npm run verify:presentations # exit 0
npm run verify:engineering   # exit 0
npm run verify:science       # exit 2 while evidence and reviews are outstanding
npm run verify:runtime       # exit 2 while experiments are unrun
```

`npm run verify` requires engineering, schematics, aesthetics, presentations AND publication, and
remains fail-closed on all five.

**Pages deploys on the four software gates.** This is an amended contract, changed deliberately
and recorded here. Deployment used to require the publication gate as well — which is BLOCKED
until an independent reviewer signs, so the page could never ship, and the artifact upload was
in fact unreachable: it sat inside the publication job *after* `verify:science`, which exits 2.
Deployment now requires `engineering`, `schematics`, `aesthetics`, `presentations` and a
`package` job that exists so building the deployable artifact is never entangled with the
scientific audit. Each must genuinely pass; none is `continue-on-error`. The `publication` job
still runs on every push and still reports BLOCKED — the published page states its own
scientific status rather than being withheld, which is the point: a reader can see both that
the software is verified and that the science is not.

**Publication completeness is an offline gate, and CI gates on contradictions only.** The
publication audit answers two questions. *Completeness* — is every claim backed by cached upstream
bytes and an independent signed review? — cannot run in CI by construction: `.verification-cache/`
is git-ignored because redistribution of the source bytes is not authorized, so CI would report
`verified_sources: 0`; and no reviewer identity exists, because inventing one is forbidden. A CI job
asking that question could only ever be red, and a check that can never pass is noise a reader stops
reading. It used to be exactly that: a permanently failing `publication` job whose red X was
indistinguishable from a contradiction, because `aggregate()` collapses BLOCKED and FAIL into one
exit code and GitHub renders both the same.

So completeness runs on a trusted machine — `npm run verify:science`, exit 2, **6,120 BLOCKED and
38 PASS findings, 0 FAIL** — and its report is committed as `verification/reports/publication.json`.
This document states its figures, and `verification/tests/test_reported_numbers.py`, which runs
inside the *green* engineering gate, fails if the record drifts from the report **or if the report
was generated on other bytes**: it re-derives the scientific lock and the consumer inventory from the
working tree and compares them to the two digests the report binds. "Offline" cannot quietly become
"stale". BLOCKED is enforced by a check that can pass, instead of announced by one that never could.

*Contradictions* — does any claim disagree with the evidence it cites? — is the half CI can honestly
answer, and `publication-contradictions` answers it on every push, on CI's own bytes: it runs the
full audit, fails on exit 1, on any FAIL finding, or on a missing or empty report, and accepts exit
2. That is not `continue-on-error` and is stricter than one. Nothing deploys on it — a test asserts
it can never enter the deploy gate's requirements, and that no job named `publication` exists to be
mistaken for a completeness result.

Success is not all-green; success is each gate telling the truth about what it actually checked.


## Dedicated V4.1 landing

DeepSeek V4.1 Flash ships at `dist/deepseek-v4.1.html`, built from the shared shell and opening
on the whole-network CED view. Atlas navigation, fixed checkpoint identity, serving phase
inspection, reloads, exports, return navigation and mobile layout are exercised in Chromium.
The same navigation also passed through `npm start` over HTTP. All registered portable files participate
in engineering identity, repeated builds, Pages packaging and attestation. Dedicated previews
carry that page's own artifact hash. Publication and runtime status remain separate.

Corrected scientific consumers: shared post-RoPE KV and inverse output RoPE; previous pre-vector
versus current post-vector/Sinkhorn mHC ownership; mHC comparison identity; bounded sliding-window
state and derived FP4 indexer storage; actual exported phase-depth validation. The active
manifest binds these consumers to the existing pinned publisher anchors and shared ledgers.
Existing source snapshots, independent review subjects and trust roots are unchanged. Private hardware receipts were subsequently withdrawn for public distribution.
Three SGLang snapshots were added at ac1a56987c5c552f9f4284f5a98db5d65541fafa, with exact
V4.1 dispatch and storage anchors. Main KV and packed indexer payload reconcile to 890 B/token;
window state and allocation overhead remain separate. [Research and source scope](docs/design/V41-SERVING-RESEARCH.md)
record the development-branch serving implementation and Miles plugin. Neither is a runtime receipt.

Dedicated HTML SHA-256: `f51f000ce609a3f5122a985dd49e404b3d6d43162abbef8d3498b884af0e12c6`

CED phase depths follow the report-level scheme. The pinned reference forward still visits all backbone layers; encoder-only prefill execution is not certified by that code. This distinction is visible in the CED caption and serving inspector.


## Book accounting

`npm run verify:accounting` reports 399 accounting scenarios: 309 agreement,
82 explained_difference, 186 unsupported and 0 contradiction findings. These are
arithmetic classifications, with explicit units, denominators and exclusions, not
publication or runtime acceptance. The independent Python calculations reproduce the
anchored cache example as 6,710,886,400 bytes, 6.7108864 decimal GB and 6.25 binary GiB.
The published table's GiB label is recorded as an explained difference.

The 16-byte training baseline remains unchanged. An additional FP32 gradient accumulation
buffer would make 20 bytes per parameter. CSA2 unique cache owners and repeated execution
reads remain separate. Both portable pages expose these assumptions through native details
controls, tested with Chromium keyboard navigation, narrow viewport, print and offline delivery.

The engineering run skipped four receipt-dependent checks while creating its receipt, five
optional runtime-candidate tests, and the accepted-hardware test class because private receipts
are withheld. Synthetic runtime binding tests ran and passed.
The receipt-dependent public-record checks passed on the completed receipt afterward.

## Public distribution privacy

Raw hardware captures and accepted runtime receipts are withheld, not redacted and re-signed.
The originals remain in an ignored private local backup. Synthetic test captures do not certify
any machine. `npm run verify:privacy` scans current publishable files, and `npm run package:public`
creates a snapshot without Git history, ignored files, local ownership or filesystem timestamps.
The pre-scrub Git history contains personal commit metadata and historical private captures.
Existing public branches require a history rewrite and an audit of every reachable commit;
private recovery bundles and unrevised local worktree refs must never be pushed. The clean
archive also supports initializing a new repository. Public citations and code-owner handles
remain as attribution and review ownership.

## Dedicated model pages

The landing directory links to Qwen3.8, Kimi K3, and DeepSeek V4.1. Qwen3.8 keeps its 27B and
Flash-Next variants on one page; the other pages fix their checkpoint identity. A shared
registry drives page builds, artifact identity, browser routing and preview selection.
Chromium exercises real landing links, variant switches, reloads, identity enforcement,
scenario exports and return links. All eight portable files are included in the software gates.

- `deepseek-v4.1.html` SHA-256: `f51f000ce609a3f5122a985dd49e404b3d6d43162abbef8d3498b884af0e12c6`
- `index.html` SHA-256: `2b3a2c86a0376a614e8c784e78da44f4197f34349ea2f26ae07fd07e6f5bd05c`
- `kimi-k3.html` SHA-256: `2cdd9cb673acfab4828c472ee830d493310fefbe2e552699531410822264b961`
- `qwen3.8.html` SHA-256: `cc1b7819c3dcb149d8c249975e8dc627fbff27935536941cf07234402adec75b`

Publication coverage: 3557 REVIEW_REQUIRED, 2334 CLAIM_ANCHOR.

## Shared systems sections

Site navigation provides Home, GPU Hardware and Networking. Model tabs expose Architecture,
Training, Serving and Evidence. Compare has no navigation/search entry or published preview;
its pure scene remains covered by internal legacy sweeps. Old Compare bookmarks open Architecture.
Hardware and Networking use canonical shared pages without model identity banners. Networking
exposes a workload-model selector, and links preserve the originating scenario. Old model-page
system bookmarks redirect to these shared pages. Repeated local arrow-key navigation keeps focus
in the model tabs. All eight files receive artifact identity and portable delivery checks.

- `hardware.html` SHA-256: `9b2801273fdf4f7f5d87df6d92f455e6cc945f63c003c6d34aec18aa2aa171cb`
- `networking.html` SHA-256: `94e35f459861081cdea9c1df015da1c5460e927f073e85ca7537f50a88dcf738`

- `foundations.html` SHA-256: `b4c050bf9e6a73021c506f29cca4e4d02089b871f2e8f8b3ecee1925e5d8e868`

- `data-representation.html` SHA-256: `239a76a51b602b1eefcb74757c513c826ec01ccd06792ca8075ac9b2afa56e17`

## Landing directory and data reference

The static homepage links to Modeling Foundations (DeepSeek V2/V3 and Qwen3), the Qwen3.8,
DeepSeek V4.1 Flash and Kimi K3 pages, and shared GPU Hardware, Networking and Data
Representation sections. Eight portable artifacts are delivered. The directory keeps in-page
anchors local and migrates legacy model/scenario fragments to their canonical page.

Chromium verifies real links, exact family membership, reloads, offline use, keyboard focus,
heading order, responsive widths, reading contrast and print for both static documents.
The current capture manifest includes 33 previews, including the homepage and data reference.

Four general data-topic consumers bind the exact displayed prose and reference URLs.
Those URLs are reading references, not accepted upstream evidence: acquisition, precise
anchors and independent interpretation review are pending (`DOCUMENTED_NO_SOURCE` BLOCKED).
No source authority allowlist, reviewed subject, trust root or accepted source byte was changed.
