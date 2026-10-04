# Playability

A control earns its place when moving it teaches something a static figure cannot.
Every control drives real state computed from facts; nothing is decorative.

## Required

1. **Layer selection** (`LayerMap`), stored in the URL (`useUrlNumber("layer", …)`).
   It must visibly change the page: the block graph for that layer's kind, the
   sentence, per-layer values.
2. **Op stepping** in every `OpGraph`: click, Previous/Next, arrow keys. The kit does
   this; your job is ops listed in execution order with complete inspectors.
3. **One widget per distinctive mechanism.** Pick what this model is known for:
   - attention pattern → `AttentionGrid` with a query slider (causal, sliding window,
     sparse top-k, block-sparse);
   - rotary positions → `RopeDials` with a position slider;
   - MoE routing → `MoeRouter` + `routeToken` with the checkpoint's scoring function,
     top-k, groups, normalisation and scale from facts, and a "route another token"
     button that changes the seed;
   - low-rank or compressed paths → `TensorScale`, optionally with a control that
     switches between what is computed and what is cached;
   - anything else → build a small SVG figure from `Node`/`Wire` driven by a `Slider`
     or `Segments`. Keep the arithmetic in a pure function next to the facts and test
     it.
4. **Size controls** for the serving section: batch and context sliders driving a
   derived cache size. Show the formula's terms in a `Readout`.
5. **URL state.** Every control uses `useUrlNumber` or `useUrlChoice` with a key
   unique on the page. Reloading the URL restores the exact view; the gate tests this.

## The pattern for every interactive figure

```
control  →  one piece of URL state
figure   →  re-renders; highlights via data-attributes, not new colours
<Sentence>  restates the current state in words (one per figure)
<Readout>   exact values for the current state
```

The sentence is not optional: it is what makes the figure readable without colour and
what a screen reader announces.

## Synthetic inputs

Widgets that need inputs (router logits, attention scores) use seeded synthetic values
from `kit/lib/rng`. Say so in the caption: the *rule* is the checkpoint's, the
*numbers going in* are synthetic. Never present synthetic output as model output.

## Defaults

The default state of every control is what most readers will see, so it must already
show the point: a query in the middle of the sequence, not position 0 where the causal
mask hides everything; a rotary position where the dials have turned; a layer of the
most interesting kind; a seed whose result has visible structure.

## Do not

- Add a control that changes nothing meaningful (a theme toggle counts as nothing).
- Animate on a timer. State changes only when the reader acts.
- Hide required content behind a control: the default state must be a complete,
  correct picture.
