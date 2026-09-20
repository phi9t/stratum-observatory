# Scientific verification — stratum-observatory-3

**runtime: BLOCKED**

Generated: 2026-09-20T19:37:21.675772+00:00

This report distinguishes software/contract tests from evidence-backed science. Missing evidence is not a contradiction, and a green engineering run is not scientific certification.

## Coverage

| Item | Count |
|---|---:|
| inputs | 604 |
| functions_and_files | 60 |
| scene_contracts | 3109 |
| claims | 3797 |
| sources | 83 |
| verified_sources | 81 |
| verified_scientific_claims | 0 |
| passed_presentation_bindings | 0 |

## Gates

| Domain | Status |
|---|---|
| evidence | BLOCKED |
| model | BLOCKED |
| gpu | BLOCKED |
| scale_up | BLOCKED |
| scale_out | BLOCKED |
| communication | BLOCKED |
| training | BLOCKED |
| serving | BLOCKED |
| numerics | BLOCKED |
| storage | BLOCKED |
| measurement | BLOCKED |
| visualization | BLOCKED |
| release | BLOCKED |

## Findings by failure mode

| Code | Status | Count |
|---|---|---:|
| CED_ACTIVE_RATIO | PASS | 1 |
| CED_FIELD_UNMAPPED | PASS | 1 |
| CED_PUBLISHER_AGREEMENT | PASS | 1 |
| CLAIM_ANCHOR | BLOCKED | 2334 |
| CLAIM_MODEL_COVERAGE | PASS | 1 |
| CLAIM_ROLE | BLOCKED | 320 |
| CLAIM_SOURCE_BLOCKED | BLOCKED | 2 |
| CONTRACT_CONSISTENT | PASS | 24 |
| CONTRACT_EVIDENCE | BLOCKED | 32 |
| CONTRACT_MODEL_COVERAGE | PASS | 9 |
| DERIVATION_INPUTS | BLOCKED | 1 |
| DOCUMENTED_NO_SOURCE | BLOCKED | 17 |
| DOMAIN_INPUT | BLOCKED | 8 |
| ENGINEERING_CURRENT | PASS | 1 |
| MODEL_DUAL_CODE | BLOCKED | 304 |
| RECEIPT_UNATTESTED | BLOCKED | 1 |
| REVIEW_REQUIRED | BLOCKED | 3829 |
| RUNTIME_NOT_RUN | NOT_RUN | 32 |
| SOURCE_CACHE | BLOCKED | 2 |
| SOURCE_UNFETCHED | BLOCKED | 2 |

## First unresolved findings

**SOURCE_UNFETCHED / ib-opensm** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / ib-opensm** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / dynamo-nixl** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / dynamo-nixl** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**REVIEW_REQUIRED / DEFAULT_STATE.batch** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.compare** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.context** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.detail** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.fabric** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.hardware** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.intensity** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.kvBytes** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.layer** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.library** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.model** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.oversub** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.phase** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.placement** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.ranks** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.stage** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.stateBytes** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.theme** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.trace** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

**REVIEW_REQUIRED / DEFAULT_STATE.traffic** — No signed semantic review. A reviewed boolean is not evidence. Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.

The adjacent JSON report contains every finding and consumer ID. No raw source, signed review, GPU run or deployment measurement is inferred from a URL or a fixture.
