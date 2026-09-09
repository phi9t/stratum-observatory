# Evidence anchors and the review ceremony

Date: 2026-09-09
Status: approved design, not yet implemented

## Problem

The publication gate reports **9,013 BLOCKED findings and no FAIL contradictions**. Two codes are
99% of that number:

| Code | Count | What it is |
|---|---:|---|
| `CLAIM_ANCHOR` | 5,899 | evidence entries whose `anchor` is `null` |
| `REVIEW_REQUIRED` | 3,056 | every claim (3,025) plus every contract (31) |
| `CONTRACT_EVIDENCE` | 31 | cascade from claims that are not verified |
| `DOCUMENTED_NO_SOURCE` | 12 | `VIEWS`, 7 `VIEW_NAMES.*`, 4 contract-claims |
| `DOMAIN_INPUT` | 8 | 3 communication receipts, 5 stub contracts |
| `SOURCE_*` / `CLAIM_SOURCE_BLOCKED` | 6 | the two redirect-blocked sources |
| `RECEIPT_UNATTESTED` | 1 | self-produced engineering receipt |

The anchor debt is not distributed the way the raw count suggests:

| | claims | `CLAIM_ANCHOR` | `REVIEW_REQUIRED` | anything else |
|---|---:|---:|---:|---|
| `scene.*` derived contracts | 2,520 | 5,000 | 2,520 | none |
| atomic claims | 505 | 899 | 536 | all 82 remaining findings |

The manifest holds 5,901 `null` anchors while the report shows 5,899 `CLAIM_ANCHOR` findings, and
901 atomic `null` anchors against 899 findings. The two-entry difference is the pair of citations
naming the redirect-blocked sources: those raise `CLAIM_SOURCE_BLOCKED` and the audit stops before
reaching the anchor check. Counts of *work to do* below use the manifest figures; counts of
*findings* use the report.

Every scene claim already carries four dependencies (`MODELS.<model>.d`,
`MODELS.<model>.layers`, `file.src/core.mjs`, `file.src/diagrams.mjs`). Its citation list is a
**copy** of its inputs' citations, with the code entries anchored and the configuration, paper and
vendor-document entries left `null`. A scene claim's `expected` is a digest of rendered panel text;
citing the DeepSeek-V2 paper as `documents` for it is not provenance. Copied evidence is not the
same thing as derived-from-verified-inputs.

## Outcome

Phases 1 and 2 take BLOCKED from **9,013 to roughly 3,110**, after which the remaining backlog is
almost entirely `REVIEW_REQUIRED`. That is the goal: reach the state where the review ceremony is
the whole remaining story, and then build a ceremony a human can actually complete.

**Publication remains BLOCKED even if every scope is signed.** Five stub contracts, two
redirect-blocked sources and `RECEIPT_UNATTESTED` are untouched by this work. Any plan that implies
a green publication gate is wrong.

## Phase 1 — Anchor the 901 atomic citations

505 atomic claims: 362 model, 82 gpu, 45 visualization, 16 spread across the remaining domains. 410
are classified `documented`, so a concrete source anchor is exactly what they owe.

The 901 unanchored entries by source role: 277 `publisher_config`, 272 `paper`, 237
`vendor_document`, 62 `publisher_code`, 49 `framework_code`, 4 `library_code`.

### 1.1 Re-pin three papers to full text

`ds2-paper`, `ds3-paper` and `q3-paper` are pinned to arXiv `/abs/` landing pages. Those are
abstract pages: layer counts, head dimensions and expert geometry are not on them, and the terms
that are present (`671B`, `MLA`) occur three to four times, which fails `text_quote`'s uniqueness
rule. Availability was verified on 2026-09-09:

| Source | Full text | Size |
|---|---|---:|
| `ds2-paper` | `https://arxiv.org/html/2405.04434v5` | 727 KB |
| `ds3-paper` | `https://arxiv.org/html/2412.19437v2` | 550 KB |
| `q3-paper` | `https://arxiv.org/html/2505.09388v1` | 986 KB |

`q3-paper` has no v2 or v3 HTML — both return 404, so v1 is the only full-text rendering.
`k3-paper` is already pinned to `/html/`, so this is the established pattern rather than a new one.

Re-pinning goes through `sources.py`'s candidate-and-promote path, never an in-place edit of an
accepted source. The identity change is recorded explicitly so it reads as a correction of a
citation that pointed at the wrong document, not as a silent swap of accepted bytes.

While there: the `version` field on these sources still reads "exact document snapshot pending" on
bytes that are in fact locked and verified. Correct it.

**PDF was considered and rejected for now.** `sources.py:30` and `evidence.py:140` both decode
strictly as UTF-8 before any other check, and every anchor kind in `select_anchor` operates on
decoded text lines, so a PDF is rejected at acquisition. Supporting PDF honestly means pinning the
PDF as the source of record, storing an extracted-text artifact whose digest is bound to the PDF's
digest, placing the extractor inside `toolchain_digest`, proving extraction is byte-deterministic,
and adding a `pdf_text_quote` kind — a subsystem that also inserts a transform between the pinned
bytes and the quote, so the anchor would prove a fact is in *our extraction*. Relaxing the decode
guard is not an option. Recorded in `VERIFICATION.md` as a known limit: any future pre-2024 paper
without an arXiv HTML rendering will force this decision.

### 1.2 Configuration sources become JSON pointers

605 unanchored entries hang off `MODELS.*` claims. Where the field has a counterpart in the
publisher's own configuration, the anchor is a `json_pointer`, matching the 86 pointer anchors that
already exist.

A field-to-pointer mapping table drives a tool. **No anchor is written that has not been proven to
resolve**: the tool resolves each candidate against the cached bytes first, and where
`relation: equal`, the resolved value must equal the claim's `expected` or the entry is reported
rather than written.

### 1.3 Papers and vendor documents become quote anchors

209 unanchored entries hang off `HARDWARE.*` claims across 11 fields each. `text_quote` already
exists and requires both a unique quote and explicit `contains` tokens, so the quote is an
integrity anchor and the meaning still rests on review.

### 1.4 The residue is a reported finding, not a rounding error

Some citations point at a source that provably does not contain the fact. An NVIDIA product page
carries `80GB` and `3.35TB/s` exactly once each, but says nothing about compute capability, PCI
device IDs or MIG. Three honest resolutions, and only these three:

1. **Re-point** to a source that does state it — the `docs.nvidia.com` tuning and system guides
   are HTML and already in the catalog.
2. **Drop the citation** and let `CLAIM_ROLE` fail loudly, which is a true statement about the
   evidence.
3. **Reclassify**, but only within the existing `PRESENTATION_SUFFIXES` whitelist
   (`id`, `name`, `family`, `sources`, `blurb`). Fields like `MODELS.*.summary` and `MODELS.*.repo`
   are our own labels rather than upstream facts but are *not* on the whitelist; widening that
   whitelist is a policy-root change and is out of scope here.

Inventing a quote to satisfy the check is forbidden. The residue count is reported before any
decision is taken about it, and is not silently absorbed.

## Phase 2 — Provenance flows through dependencies

This is the policy-root change. It lives in `verification/runner.py`.

### 2.1 The rule

For a claim classified `derived`, `required_roles` and `MODEL_DUAL_CODE` are satisfied by the union
of roles across its **verified** dependencies — verified meaning `claim_status == PASS`. An
unverified input can never lend a role.

This is strictly harder to fool than today's behaviour. Right now a copied citation satisfies the
role check on a scene claim whether or not the input claim it was copied from ever verifies.

### 2.2 The restructuring it forces

Role evaluation currently happens in the first pass over claims, before the dependency fixpoint at
`runner.py`'s `DEPENDENCY_BLOCKED` loop. It has to move after the fixpoint, which means the loop
now feeds role resolution rather than only propagating blocks.

That loop is the load-bearing evaluation order of the entire gate. **Characterization tests lock
the current behaviour before it is touched**, and they stay afterwards.

### 2.3 Then strip the copied citations

With roles carried by dependencies, the unanchorable copied entries are removed from the 2,520
scene claims. The anchored code entries stay where they are genuine.

### 2.4 Negative tests, written first

- An unverified dependency cannot satisfy a role.
- A role that no dependency holds cannot appear on the dependent claim.
- A dependency cycle cannot manufacture verification.
- `MODEL_DUAL_CODE` still refuses two repositories of one organization when reached through the
  dependency path.
- A claim with no dependencies and no evidence still fails, as `DERIVATION_INPUTS` requires.

Net effect: `CLAIM_ANCHOR` on scene claims goes 5,000 → 0. Those claims remain BLOCKED on
`REVIEW_REQUIRED`, which is the real reason.

## Phase 3 — The review subsystem

`check_review` in `evidence.py` requires: the reviewer's public key present and unrevoked in
`verification/trust/reviewers.json` (currently `{"reviewers": {}}`), `verdict == accept`, a
`subject_sha256` binding the exact claim, an unexpired timezone-aware expiry, `author != reviewer`,
and a valid Ed25519 signature over the canonical body. Its strength comes from the trust root being
protected and from the reviewer holding a private key the author does not.

### 3.1 Packets a reviewer can actually read

`review_requests.py` emits `{id, kind, subject_sha256, statement}`. A reviewer holding that would
have to reconstruct every piece of evidence from the repository themselves.

New `verification/review_packets.py` emits a self-contained packet per subject: the statement, the
`subject_sha256`, the consumer value, and for each evidence entry the **resolved selected text**
together with its source identity and anchor. The packet is readable without the repository.

### 3.2 Scoped reviews

3,056 individual signing acts is not a workflow a human completes, and 2,520 machine-generated
scene contracts are one judgement rather than 2,520.

The review body gains `scope: {selector, count, scope_root}`. Verification recomputes `scope_root`
over the subject digests of **all claims in the current manifest matching `selector`**. Adding a
claim that matches the selector, or changing any member's evidence or consumer value, changes the
root and invalidates the review. The body stays small, no inclusion-proof machinery is needed, and
the "binds this exact subject" property is preserved.

Negative test: a claim outside the enumerated selector cannot claim coverage; a selector broad
enough to sweep in unrelated claims changes the root and fails.

### 3.3 Signing is the reviewer's act

`verification/sign_review.py` builds the canonical body and signs it with a key path supplied by
the operator. **The human reviewer runs it.** A signature produced by the agent that authored the
claims would carry no information, whatever the `author` field said.

Key generation, fingerprint recording, `CODEOWNERS` protection of the trust root and the revocation
path are documented in a new `docs/verifiers/REVIEW_CEREMONY.md`. The private key is generated by
the reviewer, never enters the repository, and only the public key and its fingerprint are
committed.

### 3.4 The honesty record

The reviewer identity will be held by the same human who authors the claims, under a distinct
identity. This satisfies `author != reviewer` as written. `VERIFICATION.md` will state plainly that
**independence here is procedural, not actual** — a single party holds both roles. Recording that
is the difference between a gate that reports its own strength honestly and one that launders a
mechanism into a guarantee.

## Phase 4 — Truth-in-reporting

`VERIFICATION.md` has drifted from the reports it claims to summarise:

- "15,737 BLOCKED findings" — now 9,013.
- "5,631 evidence entries name a source but no selector" — now 5,899.
- "Accepted raw-byte snapshots | 0" — now 60 of 62 sources verified.
- "Qwen model claims are consequently and correctly BLOCKED on `MODEL_DUAL_CODE` and `CLAIM_ROLE`"
  — both are now 0.
- The aesthetics section says 112 scenarios while the table above it says 156.
- Artifact, lock and verifier identity hashes need re-derivation from the fresh receipt.

Refresh from the actual reports, and add the PDF-extraction limit from Phase 1 to the known-and-
unclosed section.

## Verification

Per phase: a failing behavioural or negative test first, implement, then

```sh
npm run test:verifiers && npm test
npm run verify:engineering    # fresh receipt; every phase invalidates the previous one
npm run verify:science        # exit 2, BLOCKED
npm run verify:runtime        # exit 2, BLOCKED
npm run verify:schematics && npm run verify:aesthetics && npm run verify:presentations
```

Success is:

- No new `FAIL` codes anywhere. `MODEL_DUAL_CODE` stays 0.
- Engineering, schematics, aesthetics and presentations stay PASS at exit 0.
- Science and runtime stay BLOCKED at exit 2, with `CLAIM_ANCHOR` reduced to the reported residue.
- Every anchor written resolves against real cached bytes; none was authored to satisfy a check.

## Risks

- **Phase 2 restructures the runner's evaluation order**, which is the load-bearing loop of the
  whole gate. Characterization tests go in before the change and stay after it.
- **The Phase 1 residue could be large.** If a substantial share of vendor citations prove
  unsupportable, Phase 1 becomes a re-sourcing task rather than an anchoring task. The count is
  reported before the decision, not absorbed into it.
- **Scoped reviews are a new trust primitive.** A selector that is too broad is a laundering path.
  The negative tests are the guard, and the selector is recomputed from the live manifest rather
  than trusted from the body.
- **Phase ordering matters for receipts.** Every phase touches `verification/**` and therefore
  invalidates `verifier_sha256`; Phase 1 also moves `lock_sha256`. Re-run `verify:engineering`
  before quoting any science or runtime result.
