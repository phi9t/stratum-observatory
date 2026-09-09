# Scientific verification layer — design

**Approval:** executable CI gates + machine-readable evidence + human-readable specs; Kimi K3 is the first target. Existing rendering is unchanged. This is an additive verifier system, not a claim of GPU conformance.

## Trust boundaries

1. Core exports, function bodies, and diagram text are local inputs, never evidence for themselves.
2. An upstream URL is a discovery hint. Evidence requires pinned identity, actual cached bytes, SHA-256, exact JSON pointer or code symbol/span, and matching consumer bindings.
3. Architectural inputs require BOTH publisher/Hugging Face code and an independent serving/training implementation. Configuration is additional evidence, not a substitute for code. Derived claims inherit all dependencies.
4. Hash/anchor checks establish integrity and traceability, not semantic equivalence. Structural interpretations need a reviewer-signed digest; numerical behavior needs differential execution evidence. A CPU test is not an H100/Blackwell test.
5. Empty coverage, missing evidence, unsupported backends, unavailable GPUs, and unauthenticated attestations never pass a required gate.

## Modules

`verification/evidence.py`: safe cache paths, pins, anchors, value extraction, evidence DAG, coverage and review binding. `verification/domains/`: typed model, GPU, scale-out, communication, training, serving, numerical, storage, measurement and diagram contracts. `verification/runner.py`: unified results and gate profiles. `verification/sources.py`: explicit bounded source acquisition and drift reports; never silently update a lock. `tools/export-verification.mjs`: actual local exports/function hashes/scene inventory. `verification/manifests/`: target bindings and source catalog. `verification/schemas/`: versioned interchange schemas. Tests are standard-library Python plus the existing Node/Chromium tests.

## Profiles

**engineering:** implementation tests, deterministic build, existing browser checks. **publication:** complete claim coverage, valid pinned evidence and necessary reviews, domain contracts, engineering receipt bound to the same artifact. **runtime:** publication + specified CPU/GPU/multi-node conformance receipts for the exact software/hardware/backend tuple. Report PASS, FAIL, BLOCKED, NOT_RUN and NOT_APPLICABLE separately; NOT_RUN never satisfies a required gate.

## Coverage

Evidence, model graph/config/weights/tokenizer/vision, GPU silicon/package/system and scale-up, IB/RoCE scale-out, NCCL/NVSHMEM/NIXL semantics, training/parallelism, serving/cache isolation, numerics/kernels/quantization, storage/checkpoints, measurement/reliability, visualization and reproducible release. Each domain owns machine assertions, source obligations, empirical obligations, negative tests and remediation.

## Acceptance

A falsified source hash, orphan field, stale function body, missing framework anchor, GB300-as-single-GPU, duplex/aggregate-rate mistake, RoCE-mode mismatch, enqueue-as-completion, stale cache boundary, unsafe checkpoint publication, unsupported numerical mode or stale receipt must fail or block. Merely adding a source URL or `reviewed: true` cannot turn a blocked model into a verified model. Initial target audits must retain real failures.
