# Levels of detail

A page descends from the whole model to single operations. Every level below is
required; the section names and order can change to suit the model, and a distinctive
mechanism (a sparse indexer, a memory module, a second residual stream) gets its own
section.

| Level | Section | Must contain |
|---|---|---|
| Model | **The whole circuit** | One hand-laid overview drawing: inputs, embedding, the layer stack (collapsed), final norm, head, outputs; the main dependency direction left to right. A stage ledger under it: each stage with one sentence and its output tensor. |
| Data | **Data representation** | What enters and leaves as tensors: token ids, embeddings, positions, logits; and what a decode step keeps (the cache), with widths. Use `TensorScale` when relative sizes are the point (a low-rank latent next to the full width, a compressed cache next to an uncompressed one). |
| Layer | **The stack** | `LayerMap` with every layer selectable and coloured by kind. Selecting a layer changes the block view, the sentence, and anything layer-dependent. A table of what differs by kind. |
| Block | **Anatomy of a block** | `OpGraph` of one layer for the selected kind: norms, sublayers, residual adds, in execution order. If kinds differ structurally, the graph changes with the selection. |
| Sublayer | **Attention** (one section per attention variant) | `OpGraph` at op level: every projection, reshape, norm, positional step, mask, softmax, cache read/write, output projection. A playable figure for the attention pattern (`AttentionGrid`) and for the positional scheme. |
| Sublayer | **Feed-forward / experts** | `OpGraph` of the FFN at op level. For MoE: the router as its own `OpGraph` (score function, bias, group selection, top-k, normalisation, scale), the expert FFN, shared experts drawn apart, and `MoeRouter` running the checkpoint's real rule. |
| Op | (inside every `OpGraph`) | Each op's inspector: what it does, formula, input and output shapes, parameter shapes, pinned source refs in every implementation read, and a note where implementations or modes differ. |
| Run | **Run it** | The clean reference executed in the browser at toy width with seeded weights: at least one computed figure per distinctive mechanism, with controls. See `reference-implementation.md`. |
| Weights | **Where the weights live** | `Ledger` from derived facts, reconciled with the hub total. One sentence on what dominates and why. |
| Serving | **What serving changes** | The decode-time view: what is cached per token and per layer and its size as a function of batch and context (derived, with sliders); what the serving implementation fuses, skips or reorders. |
| Provenance | **Sources and boundaries** | Table of pinned sources (role, repo, commit, file). What the page does not cover (other modalities, training losses, quantisation, sampling) stated plainly. |

Every section also ends with the study aids described in `study-material.md`.

## What "op level" means

An op is one thing the code does that changes values or meaning: a linear projection,
a norm, an activation, a reshape that introduces or removes an axis, a rotation, a
mask, a softmax, an add, a multiply, a top-k, a gather, a concatenation, a cache read
or write. One line of `forward` is usually one op. A `view`/`transpose` pair that only
moves axes for a matmul can be folded into its neighbour and mentioned in the note.

Typical counts: a block graph has 6–10 ops; an attention graph 12–25; a router 6–10;
an FFN 3–6. If your attention graph has five boxes, you have drawn the paper figure,
not the implementation.

## Shapes

Every op's inspector gives real shapes with symbolic batch and sequence:
`[B, S, 7168]`, `[B, heads, S, head_dim]` with the numbers from facts. When a shape
depends on a control (selected layer, context length), compute it.

## Scope

Draw what this checkpoint runs. A branch the config disables is mentioned once in the
relevant inspector note with the field that disables it, not drawn. A second modality
tower or an auxiliary head may be scoped out if the page says so in the boundary note
and `parameterNote` accounts for its parameters.
