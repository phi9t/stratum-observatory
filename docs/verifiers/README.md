# STRATUM Scientific Verification Framework

This is an additive verifier layer for the nine-checkpoint Observatory target (`manifests/observatory.json`), reusable through explicit adapters for other atlases. `manifests/kimi-k3.json` is historical and is not the current consumer inventory. It separates **tested software**, **source-backed scientific interpretation**, and **observed execution**. It does not infer one from another.

## Commands and gates

```sh
npm run test:verifiers       # Pure CPU verifier tests, including adversarial cases
npm run verify:engineering  # Verifier + atlas tests, two builds, actual Chromium suite
npm run verify:science      # Offline publication gate; reports every missing obligation
npm run verify:runtime      # Publication requirements + exact SKU/fabric/backend executions
npm run verify              # engineering, schematics, aesthetics, presentations AND publication; no silent bypass
```

A pass exits `0`; a contradiction/malformed input exits `1`; missing evidence or unrun work exits `2`. Reports are written even when the audit is blocked. `BLOCKED` means not established, **not necessarily false**. `NOT_RUN` is not a passing test. `NOT_APPLICABLE` is reserved; this release has no blanket waiver switch.

**Initial migration:** all delivered scientific core values, exported functions, protected files and sampled scene contracts are inventoried. The source catalog has exact repository commits and documentation locations, but no full upstream byte snapshots or signed human reviews have been fabricated. Fine-grained implementation anchors and several scenario contracts remain unresolved. The publication gate therefore correctly blocks the initial atlas. Read `verification/reports/publication.md`; do not advertise the original website as scientifically certified.

## Coverage and ownership

| Verifier | Contract enforced | Additional evidence needed for execution claims |
|---|---|---|
| Evidence/provenance | Immutable code pins; raw hashes; typed anchors; independent roles; consumer coverage; derivation DAG; signed reviews | Trusted runner/signing identity, protected policy root |
| Model | Layer partition; routing counts; tensor contractions; residual shapes; publisher + framework code | Reference/framework forward, cache, decode and gradient comparisons |
| GPU | SKU/scope; CC; HBM units; dense/sparse peaks; resource bounds; native math and compiler target | Device attributes, actual compiled kernels, driver/MIG/clock identity |
| Scale-up | GPU/CPU/tray/switch conservation; NVLink versus scale-out; NIC affinity | Peer-access/topology/NUMA/IMEX/fabric-partition probes |
| Scale-out | IB/RoCE addressing; MTU; one-way rates; oversubscription; congestion policy | HCA/firmware/OFED/rdma-core and switch evidence; RDMA correctness/counters |
| Communication software | NCCL/NVSHMEM/NIXL roles, backend/version, lifetime and completion order | Selected transport/plugin trace; ordering, reuse and error-path tests |
| Training | Rank/group mesh; global batch; optimizer-state budget; schedule DAG | Distributed update equivalence, uneven loss normalization, restart |
| Serving | Complete hybrid state; common prefix boundary; isolation; handoff; memory replicas | Chunk/prefill/decode equivalence, cancellation and concurrent restore |
| Numerics/kernels | Packed weights/scales; finite outputs; bounded reference-relative error | Real kernel/reference outputs, long-horizon drift, declared error policy |
| Storage/data/recovery | Shard completeness; hash/byte conservation; atomic commit; RNG/sampler/tokenizer pins | Interrupted write and deterministic resume; dataset/tokenizer manifests |
| Measurement | Measured versus derived; raw samples; denominators; environment; mean reconciliation | Raw latency/throughput/counter traces; tail estimators need added tested adapters |
| Visualization | Scene-to-claim binding; geometry; dependency endpoints; orthogonal paths | Exact exported artifact and browser/font/viewport evidence |
| Release/reproducibility | Source/test/build/lock identity; deterministic HTML; zero runtime requests | Required CI status and protected publication environment |

Failure handling, cancellation, isolation and compatibility are cross-cutting obligations, not an unverified “production ready” badge.

## Repository structure

`verification/evidence.py` handles integrity, source anchors and review signatures. `domains/contracts.py` implements typed checks. `runner.py` assembles coverage and gates. `engineering.py` runs fresh local tests and binds the artifact. `sources.py` performs explicit candidate acquisition. `tools/export-verification.mjs` reads the actual application instead of a hand-copied parameter list.

`manifests/observatory.json` contains consumer claims, domain contracts and the runtime matrix; `manifests/kimi-k3.json` is the superseded Kimi-only target, kept for reference only. `sources.json` is the source catalog. `reviews.json` and `runtime.json` start empty. `trust/reviewers.json` intentionally contains no invented public keys. `schemas/` contains machine-readable shapes; the checked-in schema evaluator implements only the subset used by these files.

Read [MODEL.md](MODEL.md), [GPU.md](GPU.md), [NETWORK.md](NETWORK.md), [OTHER_DOMAINS.md](OTHER_DOMAINS.md), [EVIDENCE.md](EVIDENCE.md), and [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md).

## Trust boundary

SHA-256 establishes byte identity, not scientific truth. An AST symbol proves code exists, not that a branch runs. Two independent repositories do not constitute independent mathematics. An Ed25519 review authenticates an authorized review, not an infallible conclusion. Runtime certification requires raw, checksummed experiments on the declared stack and a trusted signer.

Protect verifier code, schemas, manifests, workflows and reviewer keys through required review and branch rules. No in-repository checker can defend against an administrator replacing both the checker and its trust root. This bundle prepares CI workflows; it does not configure remote branch protection or install reviewer identities.
