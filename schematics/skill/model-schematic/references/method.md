# Method: reading the code

Read every class in the reference file and every `forward` line by line. As you read,
record in `NOTES.md`:

**Structure**
- The module tree: what contains what, and how many of each.
- What changes with layer index: dense vs MoE layers, alternating attention kinds,
  layers that share or reuse state, layers with extra modules. Find the exact
  condition in the code (`if layer_idx >= …`, a config list, a modulo).
- Residual scheme: pre-norm or post-norm, anything beyond `x + f(x)`.

**Every forward, in order**
- Each operation and the shape after it. Write shapes symbolically (`B`, `S` for
  sequence, then real widths).
- Reshapes, splits and concatenations, and what the new axes mean.
- What is added, multiplied, masked, scaled, clamped, and by what.
- Which activation, which norm and its epsilon, whether projections have biases.
- Positional information: where it enters (embedding add, rotation of q/k, bias in
  attention), on which channels, with what base and scaling.

**State and data**
- Tokenisation and special handling at the input; the embedding; the final norm and
  head; weight tying.
- What a decode step caches (keys and values in which form and width, compressed
  latents, indices) and what is recomputed.
- Train-only branches (auxiliary losses, dropout, extra heads) and inference-only
  branches.

**Things worth a figure of their own**
Look for facts the code implies that a reader would not guess:
- a reshape that conserves elements while changing meaning;
- a dimension that is surprisingly small or large relative to its neighbours (a
  bottleneck, a low-rank path);
- where most of the parameters are, and how lopsided that is;
- which tokens can and cannot see each other;
- a selection step (top-k, routing, gating) and what happens to what is not selected;
- anything cached in a different form from how it is used.

Each of these is a candidate for a to-scale drawing or a playable widget. A page with
none of them is a config table with boxes around it.

**Then compare with the serving implementation** and note every structural
difference (see `fidelity.md`).

Only after this, write `facts.ts` and plan sections. If you find yourself drawing
before `NOTES.md` covers every class, stop and go back.
