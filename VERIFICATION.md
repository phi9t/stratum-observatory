# STRATUM Observatory 3.0 — delivered-artifact verification

Generated from the fresh engineering receipt and the five adjacent gate reports in
`verification/reports/`. Every figure below is read out of one of those files;
`verification/tests/test_reported_numbers.py` fails if this document and the reports drift apart
again.

## Separate results

| Gate | Result | Exit | What actually ran |
|---|---|---:|---|
| Engineering | PASS | 0 | 320 verifier tests, 42 model/graph tests, two byte-identical builds, actual Chromium suite |
| Schematics | PASS | 0 | 684 scenes including all 504 decoder layers; 2,737 checks against core.mjs |
| Presentations | PASS | 0 | 12 checks — exports, accessibility, print, portability and preview freshness from a real file:// origin |
| Aesthetics | PASS | 0 | 156 scenarios, 359 checks; type, spacing, contrast, clearance and composition |
| Scientific publication | BLOCKED | 2 | Eight-model consumer inventory and strict evidence/review audit; **no FAIL** |
| Runtime certification | BLOCKED | 2 | 28 of 31 obligations NOT_RUN; H100, B200 and GB300 attested from the armada fleet; **no FAIL** |

## Toolchain

CPython 3.13.12 and Playwright 1.55.0, both provisioned by uv from `uv.lock`.
OpenSSL 3.0.17 verifies review signatures. `uv sync --frozen` then
`uv run --frozen playwright install chromium` reproduces it. The interpreter, the lock, the
Python pin, `tests/`, `.github/CODEOWNERS` and the aesthetics policy are all inside
`toolchain_digest`, so a receipt cannot outlive any of them.

## Browser evidence

**1,336 rendered states**, including all **504 decoder layers** across eight checkpoints in both
label modes, **56 responsive viewport/lens/theme combinations** and **32 device-scale cases at
1.25 and 2**. **0 geometry defects, 0 runtime JavaScript errors, 0 external runtime requests.**
Model/layer/scenario controls, search, keyboard navigation, tracing, focus, zoom, URL restore,
SVG/XML rendering and JSON exports passed.

This result now reproduces under **macOS system fonts** as well as the Linux CI fallback, which
is meaningful because no fonts are bundled and every geometry check is built on measured
`getBBox()` bounds. The layout does not depend on a single font stack.

It is not Safari/Firefox certification, all-system-font coverage, an exhaustive state-space proof
or GPU execution.

## Aesthetics evidence

`npm run verify:aesthetics` passes: **359 checks, 0 failures** across 156 scenarios and 100
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

AA (4.5:1) is the floor everywhere. Reading-size **chrome** prose is held to AAA (7:1). Canvas
labels stay at AA deliberately: they sit on the semantic tint fills that carry the colour grammar
of `docs/design/OBSERVATORY.md`, and forcing 7:1 there would wash that grammar out of the diagram.

Position must not depend on theme, and does not: composition is byte-identical between dark and
light in every scenario. All 156 match the recorded composition baseline, which is what confirms
the type and colour work was purely presentational and touched no scene geometry.

Ratios are computed from specified colours against a resolved backdrop stack. That is an upper
bound on legibility, not a guarantee of it: anti-aliasing and stem weight are not modelled, and an
element whose backdrop cannot be resolved is reported BLOCKED rather than passed.

## Diagram semantics

`npm run verify:schematics` checks what the diagram means rather than where its boxes landed. The
scene IR comes from the diagram builders and, per scene, `core.mjs` independently states what is
true; the two are compared. 684 scenes, 2,737 checks, 504 decoder layers, no browser.

Three rules were removed or narrowed during authoring because the data showed they were not
contracts the IR makes: wire endpoints are semantic rather than geometric, wires fanning out from
one node legitimately share a trunk, and unwired nodes are annotation cards rather than defects.
Each was corrected rather than accommodated by widening a tolerance.

Row alignment holds in its strong form: `norm1` and `attention` sit at pixel-identical heights in
all eight presets. Ten operations occupy a second row and every one has a structural cause — Kimi's
latent MoE inserts a stage, Flash-Next's QSA history sits higher, and the 93/94-layer checkpoints
need another row of layer tiles. A second row is permitted; a third is not.

## Delivered artifacts

`npm run verify:presentations` is the only suite that opens the artifact from a real `file://`
origin — every other suite injects with `set_content()`, so the path a reader takes had never been
exercised. Its 12 checks validate the exported SVG and the exported scenario against a schema the
page has declared all along with nothing defining it, check accessible names, heading order and
visible focus, emulate print and reduced motion, and consume `previews/manifest.json` so previews
cannot silently depict an artifact that no longer ships.

The artifact loads nothing from an external origin. Its 23 external URLs are arXiv and vendor-doc
citations the reader clicks, which is the provenance feature, not a portability defect.

## Scientific inventory

| Item | Count |
|---|---:|
| Input leaves | 455 |
| Function/file identities | 27 |
| Distinct sampled scene contracts | 2520 |
| Claims including domain obligations | 3025 |
| Candidate source entries | 62 |
| Accepted raw-byte snapshots | 60 |
| Scientifically certified claims under the signed policy | 0 |
| Passed presentation bindings | 0 |
| Domain contracts / runtime obligations | 31 / 31 |

The publication audit has **5,160 BLOCKED findings, 24 PASS and no FAIL contradictions**:

| Code | Status | Count |
|---|---|---:|
| `REVIEW_REQUIRED` | BLOCKED | 3,056 |
| `CLAIM_ANCHOR` | BLOCKED | 2,046 |
| `CONTRACT_EVIDENCE` | BLOCKED | 31 |
| `CONTRACT_CONSISTENT` | PASS | 23 |
| `DOCUMENTED_NO_SOURCE` | BLOCKED | 12 |
| `DOMAIN_INPUT` | BLOCKED | 8 |
| `SOURCE_UNFETCHED` | BLOCKED | 2 |
| `SOURCE_CACHE` | BLOCKED | 2 |
| `CLAIM_SOURCE_BLOCKED` | BLOCKED | 2 |
| `RECEIPT_UNATTESTED` | BLOCKED | 1 |
| `ENGINEERING_CURRENT` | PASS | 1 |

Most of these are cascading source dependencies and absent independent reviews; this is **not** a
claim that 5,160 facts are false. No review identity, accepted source hash or execution receipt was
invented.

For the record, the evidence-anchor work began at **9,037 findings (9,013 BLOCKED)** with
`CLAIM_ANCHOR` at **5,899**. Anchoring configuration citations as resolved JSON pointers, anchoring
paper and vendor-document citations as unique quotes, dropping 148 citations that were false
provenance, and letting derived scene claims carry provenance through their dependencies together
removed 3,853 BLOCKED findings without a single fabricated anchor.

Runtime certification reports the same population plus its obligations: **5,163 BLOCKED, 27 PASS,
28 NOT_RUN, no FAIL.**

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
corroboration requirement through independent implementations from three organizations.
`MODEL_DUAL_CODE` and `CLAIM_ROLE` are both **0** findings.

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

2,046 evidence entries name a source but no selector within it. They are not oversights, and
`docs/verifiers/ANCHOR_RESIDUE.md` records a decision for all **2,987** residue rows across **128**
classes — re-pointed, dropped, reclassified, or explicitly unresolved with a stated reason.

**1,439** of the 2,046 are genuine citations on derived `scene.*` claims that no declared dependency
supplies, kept because deleting them would strip 327 claims of their only evidence for a role their
domain requires:

- **784** GPU vendor-document citations across **196** claims — the `h100`, `h200`, `b200` and
  `gb300` product pages.
- **655** communication citations across **131** claims — `nccl-api`, `nvshmem-api`, `nixl-api`,
  `nccl` and `nvshmem`.

They stay, and they keep reporting BLOCKED, which is the correct fail-closed report: the citation is
true provenance for what the diagram draws, and it is still not a selector.

The remaining 607 sit on `MODELS.*`, `HARDWARE.*`, contract records, functions and files, and each
falls inside a class in `ANCHOR_RESIDUE.md`.

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
- **Runtime receipts are invalidated by any manifest edit, and must be re-probed rather than
  re-stamped.** `lock_sha256` binds each receipt to the target and source manifests; five manifest
  edits during the anchor work moved past the lock the three GPU receipts were captured under, and
  `verify:runtime` correctly turned into three `RUNTIME_STALE` FAIL findings — a gate failure, not a
  bookkeeping nuisance. The repair was to re-probe all three hosts (H100-SXM, B200, GB300) read-only
  on the private fleet and rebuild the receipts against the current lock. Every attested value came
  back byte-identical, but that was an outcome of the probe, not a licence to skip it. Editing the
  recorded `lock_sha256` in place would have been fabrication.
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
- **`REVIEW_REQUIRED` is 3,056 findings** and is independent of everything above. No amount of
  source acquisition or hardware access reduces it: it needs a reviewer identity in the protected
  trust root and an independent signer who is not the author.

## Artifact identity

Portable HTML SHA-256: `d10c1446481c4b9b98ee7fbce93b563dec375f480d425431dab64f1ff5e27391`

Scientific lock SHA-256: `aa8a032aad955b3bfc2dd9da454dbf40f4e5e9e4824417d74dd0816a5c9920b4`

Verifier identity SHA-256: `0f067402e63855fd9d4ab5fe760a6e2b4e5bbff635fe30ca89ff1f6a7e0eb263`

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

`npm run verify` requires engineering, schematics, aesthetics, presentations AND publication. The
Pages workflow remains fail-closed and does not deploy. Success is not all-green; success is each
gate telling the truth about what it actually checked.
