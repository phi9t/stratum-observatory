# Evidence, review and receipt protocol

## Three distinct gates

**Engineering** proves that the checked-out software passes its tests and produces the tested artifact. **Publication** additionally requires complete registered consumers, source integrity/anchors, domain contracts and independent semantic reviews. **Runtime** additionally requires artifact-backed execution on every declared SKU/fabric/library scenario. Runtime status must never be implied by publication status.

## Sources and anchors

Git sources require an authorized publisher/framework/library repository, full 40-hex commit, exact path and matching HTTPS URL. Documentation requires an edition or retrieval-snapshot identity, origin and content hash. Each source records license/redistribution constraints. Raw cache bytes are SHA-256 checked; a manifest with only URLs is incomplete.

Supported selectors are JSON Pointer with identity/length extraction; exact qualified Python AST symbol; bounded line span with required tokens; and an exact unique text quotation. Each selected region has its own digest. AST parsing does not execute Python and rejects missing/ambiguous symbols; a comment containing a class name does not satisfy a code anchor. C/C++ references initially use bounded, hashed spans; this is not a C++ semantic parser.

A JSON `equal` relation is compared to the local consumer value. Other relations need semantic review of actual operands, branch guards, inheritance, aliases and dispatch. Hashes and token occurrence do not prove mathematical meaning. Derived claims carry an acyclic input-claim graph and inherit unverified dependencies.

## Acquisition is explicit, not trust-on-first-use

```sh
npm run verify:sources -- \
  --candidate verification/candidates/2026-09-06-k3.json \
  --only hf-config --only hf-model --only sglang-model
```

Acquisition uses bounded HTTPS reads, no credentials or remote execution, exact pinned git URLs, an 8 MiB source limit and a content-addressed local cache. Redirects are rejected for inspection rather than followed into an unreviewed origin. This may require resolving an approved canonical source path before acquisition; do not remove the origin guard merely to make a fetch succeed.

The output is a **candidate** containing catalog, target and results. It never overwrites accepted manifests, expected consumer values or review signatures. Inspect any upstream value/anchor change; resolve missing selectors; compare publisher and framework semantics; then deliberately promote the reviewed candidate catalog/target. A new checksum is not automatically a new approved truth.

The cache is ignored by Git by default to avoid redistributing full third-party material. Provision a controlled, content-addressed evidence store or legally permitted vendored cache in CI. The initial workflow intentionally blocks until that is supplied; it does not silently download mutable documentation during publication.

## Signed independent review

`verification/trust/reviewers.json` maps authorized reviewer IDs to repository-relative Ed25519 public keys. This release contains no private keys and invents no authorized reviewers. Add a real public key only through protected review. Keys can be revoked. Author and reviewer must differ.

A claim review signs a canonical JSON body:

```json
{
  "author": "implementer-id",
  "reviewer": "independent-reviewer-id",
  "subject_sha256": "<digest of exact claim plus its source records>",
  "verdict": "accept",
  "expires_at": "<timezone-qualified expiry>"
}
```

The base64 Ed25519 signature and body are stored under the exact claim ID in `manifests/reviews.json`. For a domain contract, sign `digest(contract)`. Source commit/bytes, selector, local expected value or code hash changes invalidate the claim subject. Contracts bind constants, tolerances, applicability and consumer mappings. Expired, revoked, unknown, tampered and self-reviews do not pass.

Export **unsigned** requests with:

```sh
python3 -m verification.review_requests \
  --claim MODELS.kimi3.d --out verification/candidates/width-review.json
```

Compute/sign canonical bytes with `verification.evidence.canonical`, not pretty-printed JSON. The reviewer must inspect the source and interpretation before signing. This tool exports digests only; it cannot approve its own work. OpenSSL verifies signatures; an unsigned `reviewed: true` field is ignored as proof.

## Runtime receipts

Each obligation names its domain, exact contract and required environment fields. There are separate obligations for each GPU SKU, system topology, IB/RoCE profile and communication library. A B200 result cannot certify H200, and one NCCL result cannot certify NIXL.

A receipt under `manifests/runtime.json[obligation_id]` includes `executed_at`, successful `exit_code`, `lock_sha256`, `consumer_sha256`, required environment, checked domain `record`, and raw `artifacts` with repository-relative paths and SHA-256 values.

What retires a receipt depends on what it attests. **Behavioural** obligations (`model`, `communication`, `training`, `serving`, `numerics`, `storage`, `measurement`, `visualization`, `release`) attest what our code does, so their receipts must carry the *current* `lock_sha256` and `consumer_sha256`; a stale one is `RUNTIME_STALE` FAIL. **Hardware-attribute** obligations (`gpu`, `scale_up`, `scale_out`) attest what a machine or fabric is, which no edit to our claim set or source files can change. Those receipts record the lock and consumer digests as capture-time provenance but are not compared against them; they bind three digests instead:

| Field | Meaning | On mismatch |
|---|---|---|
| `probe_schema_sha256` | `verification.hardware.schema_sha256()` — the hardware data model the capture used | `RUNTIME_SCHEMA` **BLOCKED**: a superseded data model is missing evidence, so re-probe |
| `contract_sha256` | `verification.runner.contract_sha256(contract, inventory['fields'])` — the contract record, bindings applied, that the probe saw | `RUNTIME_CONTRACT_DRIFT` **FAIL**: the contract now asserts something the probe never saw |
| `hardware_sha256` | the attested-half digest from `verification/schemas/hardware.schema.json`, carried so two captures of one host can be compared | absent is `RUNTIME_HARDWARE` **BLOCKED**; it is carried, not re-derived, because the gate does not hold the hardware record |

Everything else applies to both kinds: artifact digests, the `record_artifact` pointer binding, `RUNTIME_ENVIRONMENT`, `RUNTIME_TARGET`'s per-key contract comparison, `exit_code`, and the expiry window. The window stays for hardware receipts too: nothing here can detect a re-capped power limit or a swapped NIC between two probes, so expiry is the only thing that forces another look. Its reviewed subject is `digest({obligation, receipt})`; the corresponding independent signature is stored in `reviews.json`. It must also bind the normalized record to a checksummed JSON artifact via `record_artifact: {path, pointer}`. Large binary traces may accompany that normalized record.

Receipts must identify actual commands, inputs, outputs, errors and exact software/backend/driver configuration. The checker validates digests, schemas, domain assertions and identity; the trusted runner/reviewer attests that the artifacts came from the stated experiment. It cannot cryptographically prove physical execution from an arbitrary uploaded log. Never substitute a unit-test fixture for a real receipt.

## Boundaries

The JSON schemas support authoring; the checked-in evaluator implements the subset actually used. Runtime freshness is a policy, not evidence of historical falsehood. `NOT_APPLICABLE` is reserved, with no blanket waiver implementation in this release. A new target/family needs explicit policy review and adapters; the framework is not a universal compiler, theorem prover, simulator or autonomous source-truth oracle.
