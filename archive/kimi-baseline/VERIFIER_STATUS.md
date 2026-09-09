# STRATUM scientific-verifier delivery status

Verified at **2026-09-06T18:17:33.157736+00:00**.

## Three results, not one

| Gate | Result | Evidence |
|---|---|---|
| Engineering | **PASS** | 113 verifier tests; 25 original atlas tests; two byte-identical builds; actual Chromium suite |
| Scientific publication | **BLOCKED**, exit 2 | Missing upstream snapshots, precise implementation anchors, independent reviews and scenario records |
| Runtime certification | **BLOCKED**, exit 2 | Publication prerequisites plus 24 declared experiments marked NOT_RUN |

Engineering browser checks covered **302 rendered configurations** and **40 responsive combinations**; geometry defects, runtime errors and external runtime requests were all zero. SVG/JSON exports and interactions passed. This is Chromium coverage, not exhaustive cross-browser or all-combinations certification.

## What is implemented

An additive Python/Node verifier system, strict manifests/schemas, immutable source acquisition candidates, digest/anchor checks, independent Ed25519 review validation, domain contract checks, actual core/scene inventory, raw-artifact-bound execution receipts, a read-only system probe, detailed coding-agent specs and engineering/publication/deploy CI dependencies. The visual atlas and its calculations are unchanged; B200/B300/GB300 are verifier profiles, not newly added UI presets.

The code contains **13 verification areas**, **24 domain contract records** and **24 explicit runtime obligations**. Additional architectures require adapters, evidence and reviewed policy. This is not a universal semantic proof engine or a suite of completed GPU benchmarks.

## Exact audit coverage

| Inventory | Count |
|---|---:|
| Core/scenario/hardware input leaves | 75 |
| Exported functions and protected file bodies | 20 |
| Distinct sampled scene contracts | 458 |
| Consumer bindings | 553 |
| Claims including domain-policy obligations | 576 |
| Candidate upstream source entries | 25 |
| Verified source snapshots in this delivery | 0 |
| Scientifically verified claims under the new policy | 0 |
| Passing presentation-only bindings | 21 |

The publication report contains **1748 blocking findings**. These include cascading source dependencies and missing approvals; they are not 1748 demonstrated false facts. Local consistency checks and presentation bindings are reported separately from scientific claims. No evidence threshold was weakened to obtain a green result.

## Pending evidence, in order

1. Acquire the exact catalogued publisher/framework/library files and vendor documents into a controlled, content-addressed cache. A real `hf-config` acquisition attempt here returned temporary DNS failure and exit 2. The tool did not substitute partial search snippets, mutable main, or an invented file hash.
2. Resolve the per-field code symbols/spans, JSON selectors and semantics, including publisher implementation plus an independent framework implementation for model claims. Check actual consumers/branches; config JSON alone is insufficient.
3. Complete source-bound domain records: B300/GB300 memory, selected NCCL/NVSHMEM/NIXL backends/ordering, training, serving, numerics, storage, measurement, visualization and release scenarios. Every missing input is named in the report.
4. Establish trusted independent reviewers and sign the exact claim/contract subjects. No trusted identities or signatures were invented. Protect verifier, manifests, workflow and trust-root changes in the real repository.
5. Run actual differential model, GPU/kernel, topology/fabric, library-ordering and recovery/measurement experiments on the declared stacks. Capture normalized JSON plus raw traces, hashes, environment and independent signed receipts. No such experiments ran here.

The read-only local inventory probe also returned exit 2: GPU/CUDA/verbs tools were absent, and RDMA netlink access was unavailable. Its output is inventory evidence only, never successful device conformance.

## Reproduce

```sh
npm run verify:engineering
npm run verify:science
npm run verify:runtime
```

The latter two intentionally exit 2 on this initial bundle. `npm run verify` requires engineering **and** publication. Pages deployment depends on both CI jobs; this archive does not configure GitHub remotely.

## Artifact identity

- Portable HTML SHA-256: `df1044c65e6b2f3d55f8fa7656b08d8758f8cbec3d74ac2fcb5f994ad12d3bc4`
- Scientific lock SHA-256: `259e90a2be6f7c20f7759bc9bf401a19cf9a33d13a7657287104be5c30bcff9e`
- Consumer digest: `fbd6e195f45608d0aa4e3e565572776b47fac3ba4e9bdda6fb6d8c42382077b2`
- Verifier digest: `c92c0da727b59848bb0736fd04ab425b0fc0706564402e98079881c7c44c8b98`

The local engineering receipt is freshness-limited to 24 hours by policy; rerun it on another checkout or after changes. Raw command logs, complete JSON findings and machine-readable receipts are under `verification/reports/`. The source/byte/anchor/signature checks are not a substitute for interpretation or trustworthy execution provenance.
