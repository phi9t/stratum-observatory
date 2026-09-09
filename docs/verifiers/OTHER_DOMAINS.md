# Cross-layer verifier contracts

These domains are registered and executable. A validator checking a synthetic record is not the same as a populated, source-backed atlas scenario or a real runtime experiment. Missing target records remain blockers in the initial migration.

## Training and distributed execution

Bind training configuration, optimizer implementation, parallel groups, schedule and checkpoint format to the exact framework revision. Rank coordinates must form a bijective mesh. TP/PP/DP axes and overlapping EP/CP subgroups must not be naively multiplied twice. Global batch is microbatch × accumulation × the relevant data-parallel degree, not all ranks. Token-weighted loss normalization matters for uneven sequence lengths.

Persistent state accounting must name optimizer, master weights, gradient dtype, moments, sharding and offload. Muon must not inherit an Adam ledger merely because both are optimizers. Activation, temporary gather, communication and allocator buffers need separate peak-lifetime accounting. The current contract checks byte reconciliation and a schedule DAG; it is not a complete pipeline simulator.

Runtime acceptance compares a small single-device update with distributed/sharded updates, including clipping/accumulation and uneven tokens. Test gather-before-use, gradient-reduction-before-update, activation recomputation equivalence, failure during a step, and restart with the same sampler/RNG state. Report phase-specific assumptions for agentic RL, rollout/train weight versions and on/off-policy data; do not infer proprietary training topology from a model config.

## Serving, caching and memory ownership

The cache contract is indexed by model revision, layer, representation/dtype, request namespace and token boundary. Hybrid token KV, recurrent matrices and convolution histories must all be restored at a mutually compatible boundary. Mutable recurrent state is private or copy-on-write; prefix sharing must not leak tenant or request state.

Handoff order is explicit: registration → producer readiness → transfer submission → completion → consumer readiness → release. Split prefill/decode ordinarily requires two independently resident weight replicas; an alternate sharing arrangement needs its own contract. Sum logical weight/cache components separately from allocator pages, quantization scales, workspaces, graph-capture pools, output reserve and fragmentation. A logical byte estimate alone never certifies fit.

Runtime acceptance includes full versus chunked prefill, token decode, prefix mismatch, heterogeneous attention layers, reset, cancellation, speculative rollback when supported, concurrent prefix reuse, paging/eviction, replica/weight-version mismatch and transport failure. TTFT, ITL and queueing remain measurements rather than cache-size formulas.

## Kernels, numerical semantics and quantization

Specify storage, compute and accumulation formats separately. Account for packed tails, group scales, metadata/alignment and exclusions. The implemented MXFP4 profile checks four bits plus one byte scale per 32 values; it must not silently stand in for NVFP4 or prove native Hopper FP4 execution.

Differential checks reject NaN/Inf, empty or mismatched outputs and error beyond a declared absolute/relative budget. The precommitted tolerance policy is bound into the reviewed contract; a runtime record may not relax it to pass. Actual/reference arrays must come from checked runtime artifacts. Include ill-conditioned normalization, routing ties, zero/large activations, recurrent long-horizon drift, chunk boundaries and gradient checks where relevant.

A source-level operator count is not a kernel count. Fusion, quant/dequant, padding, layout permutation, PDL, CUDA Graphs and hidden synchronization require compiler/dispatch or trace evidence. Performance claims need the measurement verifier as well as numerical correctness.

## Storage, data, tokenization and restart

Bind tokenizer/preprocessing and dataset versions, special-token IDs, modality packing, label/loss masks, sequence packing boundaries and data cursor. Do not compare training/serving traces with incompatible tokenizer or chat-template revisions. Model/vision architecture claims still require their model-code anchors; a dataset name is not an input-contract proof.

A checkpoint has an explicit shard manifest with hash, size and unique identity. Its commit marker is not published until required shards are durable. Exact training restart includes weights, optimizer, RNG, sampler/cursor and step state. A weights-only checkpoint is a different artifact, not a failed training-state verifier; introduce a separately scoped contract before claiming that workflow is supported.

Runtime fault cases include partial write, missing/corrupt shard, stale manifest, resharding, interrupted download, changed tokenizer and sampler restore. Large tensor payloads should be checked through safe formats and streaming hashes. Never deserialize untrusted pickle or execute remote model code inside an offline evidence job.

## Measurement and cost models

Every result declares documented, derived, assumed or measured status; units, denominator, environment, clock/power mode, warmup and sample count. A measured claim requires raw artifacts. Timeouts and failures stay in request accounting. A payload/rate bound cannot masquerade as measured latency. FLOPs, activation traffic, HBM traffic, link bytes, wire overhead and application tokens/s are different quantities.

The initial executable estimator checks arithmetic means and consistency with raw samples. It intentionally rejects unsupported percentile/bootstrap estimators rather than manufacturing p99 confidence. Add independently tested quantile/CI adapters before certifying tail-latency plots. Measurement receipts expire after seven days for the current-state runtime profile; other execution receipts have a 30-day limit. Historical results need a separately scoped archival profile, not a fresh-date label.

## Visualization and scientific projection

Nodes, tensor labels, operators, state, links and equations have registered claim bindings. `tools/export-verification.mjs` fingerprints every core field, exported function, protected file, all 504 decoder layers across the eight checkpoints and representative training/serving/hardware/network variants. Whole-file fingerprints cover branches outside the sampled matrix; this is a change detector, not complete semantic static analysis.

The existing browser suite measures rendered SVG bounds, crossings/overlap categories, responsive overflow, interaction, URL restoration and exports. New diagrams must retain readable label lanes, explicit state rails, orthogonal edges and stable layouts. A collision-free picture is not proof that the depicted math is right; the model and systems claim gates remain mandatory.

## Release, isolation and reproducibility

Engineering runs verifier tests, atlas tests, two deterministic builds and the real Chromium suite. Its receipt binds exact consumer/function hashes, verifier/schema/workflow code, source-lock identity and the produced HTML. Old reports cannot substitute for a fresh execution. Publication requires engineering plus scientific evidence; deployment depends on both jobs.

No source acquisition, private signing key, remote-code execution or destructive system probe occurs in untrusted pull-request jobs. Protect reviewer keys and verifier policy with external branch rules/code-owner approval. Hash/signature checks, path/symlink rejection, immutable source roles, duplicate/nonfinite JSON rejection, explicit cache limits and fixture rejection are cross-cutting supply-chain controls. This is not a general application-security penetration test or a certification of an unobserved cluster.
