# DeepSeek V4.1 dedicated delivery

The user selected a dedicated page within STRATUM. The architecture implementation was
already on main at a3e94b7. This change adds a portable entry point from the shared shell,
defaulting to the whole-network CED view, with fixed checkpoint identity and a return link.
The shared accounting and STRATUM programmatic interface remain available.

## Correctness review

- CSA2 uses shared post-RoPE KV and inverse RoPE before grouped output projection.
  Pinned evidence: Attention._window_kv, Attention._compress_kv, Attention.forward.
- mHC reads with the previous pre vector and writes with the current post vector and
  Sinkhorn residual matrix. Pinned evidence: Block.hc_pre, hc_post, hc_mixes, forward.
  Comparison describes mHC on either side.
- Serving includes bounded SWA state and includes the packed indexer payload with allocation exclusions.
  Phase inspection and its verifier consume the actual serving ledger outputs.
- Ordinary checkpoint switching preserves the lens. Development serves the same built
  entry points as deployment. Reload, export, mobile layout and return navigation are tested.

## Evidence migration

The active manifest registers each changed consumer with its exact fingerprint and explicit
model, function and file dependencies. Corrected operators cite existing pinned
Attention/Block/Transformer anchors. File fingerprints identify local changes; they do not
approve scientific claims. Prior identities remain in git and the historical inventory.
Accepted source bytes, signed review subjects, reviewer keys and hardware receipts are unchanged.

## Delivery contract

Both HTML files are bound by engineering receipts, repeated builds, attestation and packaging.
Browser, aesthetics, presentation and preview checks cover the dedicated entry point.
Fresh results belong in VERIFICATION.md and verification/reports. Independent scientific
review and unperformed runtime experiments remain separate blockers.


## Review closeout

The review found and corrected ordinary model selection resetting the lens and a missing
root development route. The file-origin reader journey covers the former; an HTTP round trip
through `npm start` verified the latter. Delivery review also required mandatory per-page
preview hashes, complete presentation checks on the dedicated page, and fail-closed receipt
binding when a page or its digest map is absent. Negative tests cover missing delivery identity.

The candidate composition baseline was inspected against the generated CED and mHC previews.
Only the corrected V4.1 block, residual, comparison and serving compositions changed; new
prefill scenarios were added. No typography, spacing, contrast or crossing threshold changed.


## Report versus reference execution

The CED phase-depth model follows the reported execution scheme. The pinned publisher
Transformer.forward still iterates every backbone layer and then computes logits and samples
an output token. Its code does not establish the encoder-only prefill optimization. The CED
caption and serving inspector make this explicit; phase accounting remains a report-level model,
not measured reference-runtime behavior. First-token sampling after forward is supported by
that implementation and remains in the request lifecycle.

## Independent serving evidence added before landing

The LMSYS article led to exact SGLang and Miles development-branch implementations, recorded
in [the source research](V41-SERVING-RESEARCH.md). Three new SGLang snapshots were acquired
through the bounded candidate workflow and promoted as unsigned, source-bound evidence; prior
snapshots and trust roots remain unchanged. The cache ledger now includes FP4 indexer payload
and byte scales. Separate Python checks reject missing or wrong indexer storage and a combined
subtotal that omits it. The serving inspector distinguishes published CED phase accounting from
optional approximate replay. The innovation table calls its bounded-state row what it measures,
rather than implying that fixed window storage implements replay. Miles and Engram placement
are documented research inputs; this landing adds no claimed training deployment or GPU result.
