# DeepSeek V4.1 mechanism deep dives — proposed scope

Status: approved by the user: shareable chapter views in the existing V4.1 page.

The V4.1 overview remains the entry point and explains how the mechanisms connect.
Add separate Encoder, Decoder, Engram, CSA2, and dSpark chapters. Recommended delivery
is shareable chapter views within the existing portable V4.1 HTML, using the existing
URL state, scene inspector, trace, SVG export, themes, and offline behavior. The alternative
is a separate portable HTML for every chapter; it provides separate file URLs but increases
navigation and artifact coverage without changing the underlying learning material.

## Reader contract

Each chapter answers a concrete question, offers a focused mechanism diagram and worked
trace, identifies tensor shapes and state ownership, and links to exact source evidence.
Changing a supported scenario must update the explanation as well as the diagram. Static
prose must not duplicate architectural quantities already held by the model or ledgers.
Unsupported runtime behavior is labeled as unavailable rather than simulated as verified.

- **Encoder:** how causal prompt processing produces the encoder output; which layers
  write compressed caches, which select or reuse attention positions, and where Engram
  is injected. Explain causality, residual streams, compression, and the handoff. Keep
  reported encoder-only prefill separate from the reference implementation's full forward.
- **Decoder:** how a generated position traverses the decoder, reading retained state and
  reusing cache owners and selections. Follow the handoff through attention, experts,
  residual updates, output logits, and sampling. Avoid drawing a conventional separate
  cross-attention module that the implementation does not contain.
- **Engram:** token history to n-gram hashes, table lookup, contextual processing, and
  residual injection. Show sequence boundaries and image-token handling. Distinguish
  table residency, selected rows, and learned projections. Treat host/device placement
  as a separately sourced implementation choice.
- **CSA2:** compare Full, Reindex, and Reuse responsibilities within this mechanism;
  distinguish the local window from compressed storage and the selected read set. Trace
  cache ownership, hierarchical candidate selection, and causality through representative
  encoder and decoder layers. Selected tokens are not the cache's storage capacity.
- **dSpark:** MTP draft inputs, the target-layer attention-input features consumed by
  draft blocks, draft experts, future-position proposal, and target verification. Keep
  draft positions, draft depth, and target depth distinct. Explain accepted/rejected
  continuation only as far as source evidence establishes the rule; do not fabricate
  an acceptance implementation, acceptance rate, speedup, or unpublished parameter count.

## Implementation and evidence boundaries

Extend model-supported focal stages and their scene builders together. Preserve the
canonical V4.1 identity, current overview and old URLs. The model/source table and pure
core functions own facts and derived values; the browser owns navigation and presentation.
A dedicated mechanism module can keep these deeper scenes out of the general overview.
Existing hardware and networking remain shared site sections.

Every added scene or explanation must have an exact consumer binding and source chain.
The research map distinguishes inspected cached source bytes from live public verification,
report-level statements, and missing implementations. Source absence remains BLOCKED;
no source hash, reviewer identity or runtime receipt is invented.

## Verification and delivery

Write behavioral and negative cases for chapter navigation, model-stage compatibility,
cache/read ownership, causal boundaries, and draft-versus-target separation. Sweep every
new focal stage through scene semantics, browser geometry, export, accessibility, print,
mobile and offline checks. Generate previews from current portable artifacts. Run all four
software gates and separate scientific/runtime gates, refresh their public record, then
publish the verified portable files to the existing Pages repository.
