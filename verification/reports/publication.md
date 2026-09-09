# Scientific verification — stratum-observatory-3

**publication: BLOCKED**

Generated: 2026-09-09T06:18:50.609377+00:00

This report distinguishes software/contract tests from evidence-backed science. Missing evidence is not a contradiction, and a green engineering run is not scientific certification.

## Coverage

| Item | Count |
|---|---:|
| inputs | 455 |
| functions_and_files | 27 |
| scene_contracts | 2520 |
| claims | 3025 |
| sources | 56 |
| verified_sources | 0 |
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
| CLAIM_ROLE | BLOCKED | 1219 |
| CLAIM_SOURCE_BLOCKED | BLOCKED | 10079 |
| CONTRACT_CONSISTENT | PASS | 19 |
| CONTRACT_EVIDENCE | BLOCKED | 31 |
| DOCUMENTED_NO_SOURCE | BLOCKED | 12 |
| DOMAIN_INPUT | BLOCKED | 12 |
| ENGINEERING_CURRENT | PASS | 1 |
| MODEL_DUAL_CODE | BLOCKED | 1219 |
| REVIEW_REQUIRED | BLOCKED | 3056 |
| SOURCE_CACHE | BLOCKED | 56 |
| SOURCE_UNFETCHED | BLOCKED | 56 |

## First unresolved findings

**SOURCE_UNFETCHED / hf-config** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / hf-config** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / hf-model** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / hf-model** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / sglang-model** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / sglang-model** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / sglang-config** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / sglang-config** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / sglang-vision** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / sglang-vision** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / nccl-api** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / nccl-api** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / nccl-collectives** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / nccl-collectives** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / nvshmem-api** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / nvshmem-api** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / nvshmem-quiet** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / nvshmem-quiet** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / nixl-api** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / nixl-api** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / nixl-agent** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / nixl-agent** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_UNFETCHED / cuda-cc** — No authenticated acquisition snapshot/checksum is locked. Acquire exact pinned bytes; review the candidate lock.

**SOURCE_CACHE / cuda-cc** — Exact source bytes are absent from the offline cache. Acquire exact pinned bytes; review the candidate lock.

The adjacent JSON report contains every finding and consumer ID. No raw source, signed review, GPU run or deployment measurement is inferred from a URL or a fixture.
