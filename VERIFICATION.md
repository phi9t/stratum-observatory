# STRATUM Observatory 3.0 — delivered-artifact verification

Generated from the fresh engineering receipt and the three adjacent gate reports in
`verification/reports/`.

## Separate results

| Gate | Result | Exit | What actually ran |
|---|---|---:|---|
| Engineering | PASS | 0 | 233 verifier tests, 40 model/graph tests, two byte-identical builds, actual Chromium suite |
| Schematics | PASS | 0 | 684 scenes including all 504 decoder layers; 2,737 checks against core.mjs |
| Presentations | PASS | 0 | Exports, accessibility, print, portability and preview freshness from a real file:// origin |
| Aesthetics | PASS | 0 | 156 scenarios; type, spacing, contrast, clearance and composition |
| Scientific publication | BLOCKED | 2 | Eight-model consumer inventory and strict evidence/review audit |
| Runtime certification | BLOCKED | 2 | 28 of 31 obligations NOT_RUN; H100, B200 and GB300 attested from the armada fleet |

## Toolchain

CPython 3.13.12 and Playwright 1.55.0, both provisioned by uv from `uv.lock`.
OpenSSL 3.6.2 verifies review signatures. `uv sync --frozen` then
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

`npm run verify:aesthetics` passes: **207 checks, 0 failures** across 112 scenarios. The first run
of this gate failed on 21 defects that shipped in the delivered artifact; all are now fixed at the
source rather than by moving a threshold.

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

Composition is byte-identical between dark and light across all 56 scenarios, and the recorded
baseline did not move through the entire redesign — confirming the change was purely presentational
and touched no scene geometry.

Ratios are computed from specified colours against a resolved backdrop stack. That is an upper
bound on legibility, not a guarantee of it: anti-aliasing and stem weight are not modelled, and an
element whose backdrop cannot be resolved is reported BLOCKED rather than passed.

## Diagram semantics

`npm run verify:schematics` checks what the diagram means rather than where its boxes landed. The
scene IR comes from the diagram builders and, per scene, `core.mjs` independently states what is
true; the two are compared. 684 scenes, 2,737 checks, no browser.

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
exercised. It validates the exported SVG and the exported scenario against a schema the page has
declared all along with nothing defining it, checks accessible names, heading order and visible
focus, emulates print and reduced motion for the first time, and consumes `previews/manifest.json`
so previews cannot silently depict an artifact that no longer ships.

The artifact loads nothing from an external origin. Its 23 external URLs are arXiv and vendor-doc
citations the reader clicks, which is the provenance feature, not a portability defect.

## Scientific inventory

| Item | Count |
|---|---:|
| Input leaves | 455 |
| Function/file identities | 27 |
| Distinct sampled scene contracts | 2520 |
| Claims including domain obligations | 3025 |
| Candidate source entries | 56 |
| Accepted raw-byte snapshots | 0 |
| Scientifically certified claims under the signed policy | 0 |
| Domain contracts / runtime obligations | 31 / 31 |

The publication audit has **15,737 BLOCKED findings and no FAIL contradictions**. `DOMAIN_INPUT` fell from 12 to 8 as contracts were bound to the application or given measured records. Many are
cascading source dependencies and absent independent reviews; this is **not** a claim that 15,740
facts are false. No review identity, accepted source hash or execution receipt was invented.

**Verified scientific claims: 0.** Two claims previously passed — the stylesheet and the HTML
entry point — because `presentation` claims skipped review entirely. Every claim is now reviewed,
so those green cells are gone. Calling a user-visible value presentational is itself a judgement,
and it is the judgement most worth checking.

**Qwen presets have no publisher implementation.** `huggingface/transformers` was previously
registered as `publisher_code`, so `transformers` plus `sglang` satisfied "publisher
implementation AND independent framework implementation" using two third-party reimplementations
and no publisher at all. transformers is now `framework_code`, and dual-code independence compares
organizations rather than repository names. Qwen model claims are consequently and correctly
BLOCKED on `MODEL_DUAL_CODE` and `CLAIM_ROLE`.

## Known and unclosed

Recorded so they are not mistaken for oversights:

- The engineering receipt is **self-produced**. This is now a finding rather than a sentence:
  `RECEIPT_UNATTESTED` BLOCKS on any receipt without a build attestation. CI attests the exact
  tested artifact; branch protection remains the repository owner's to configure, and
  `.github/CODEOWNERS` has no force without it.
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
- **Two sources remain unfetched** because their URLs redirect and the acquirer refuses to follow
  a redirect without inspecting the new origin. Pin the resolved origin; do not relax the guard.
- **5,631 evidence entries name a source but no selector within it** (`CLAIM_ANCHOR`). These are
  papers and vendor documents, which need span or quote anchors rather than JSON pointers.
- **`REVIEW_REQUIRED` is 3,056 findings** and is independent of everything above. No amount of
  source acquisition or hardware access reduces it: it needs a reviewer identity in the protected
  trust root and an independent signer who is not the author.

## Artifact identity

Portable HTML SHA-256: `83bc7a5982f07ae108f4db029c3713b3fa933d2f51715d4c32e3e2d873c4ce54`

Scientific lock SHA-256: `b0c0255c43600c6e847e62d86f4891b06a3f14cc9f83ccb3d1445eaddf4afe38`

Verifier identity SHA-256: `516cbb59f1477a4467878298d11fa0cbdb970ea97aeddf4b3f0f8f9fa3bdbf97`

Complete command logs and receipts: `verification/reports/`. Full browser matrix:
`tests/browser-cases.json`. Re-run after changing application, verifier or evidence; old receipts
are not transferable.

## Reproduce

```sh
uv sync --frozen
uv run --frozen playwright install chromium
npm run verify:aesthetics    # exit 0
npm run verify:engineering   # exit 0
npm run verify:science       # exit 2 while evidence and reviews are outstanding
npm run verify:runtime       # exit 2 while experiments are unrun
```

`npm run verify` requires engineering, schematics, aesthetics, presentations AND publication. The Pages workflow remains fail-closed and
does not deploy. Success is not all-green; success is each gate telling the truth about what it
actually checked.
