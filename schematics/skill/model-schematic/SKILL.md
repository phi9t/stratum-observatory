---
name: model-schematic
description: Build a model schematic — a single-page, playable, source-faithful architecture deep dive for one neural-network checkpoint, drawn at model, layer, block and op level from its real implementation. Use when asked to visualize, diagram or explain a model architecture, forward pass or data representation as a web page.
---

# Model schematic

You are building one page for one checkpoint: `schematics/models/<slug>/`. A reader
should be able to start at the whole model, pick any layer, open its block, and step
through it one operation at a time, with every shape, formula and number traceable to
pinned source code. It should also be beautiful.

Two things make these pages good, and both are easy to skip:

- **The page is drawn from code you have actually read.** Not from memory of the
  paper, not from the model family in general. The most valuable content is what the
  code implies and nobody wrote down.
- **You look at what you built.** SVG text overflows, labels collide and dark mode
  breaks in ways you cannot see in source. Screenshot, look, fix, repeat.

Work inside `schematics/`. Read `README.md` there first: it defines the folder
contract and the gates. The worked example is `models/gpt2/`; copy its patterns, not
its content.

**Before designing anything, open every image in `examples/`** (next to this file).
They are screenshots of the worked example and of the widget catalogue: that is what
a finished section looks like, and your page is judged against them. Your sections
should be at least as clear, as spacious and as consistent.

## Procedure

1. **Scaffold and pin.** `npm run new -- <slug> <hf-repo>`. Add sources to
   `sources.json`: the checkpoint's `config.json`, one *reference* implementation
   (publisher code or transformers), at least one independent *serving*
   implementation (vllm or sglang), and a training one if it exists. Then
   `npm run fetch -- <slug>`. How to find and choose them: `references/fidelity.md`.
2. **Read the reference implementation in full**, then the serving one. Take notes as
   you go in `models/<slug>/NOTES.md`: every module class, what its `forward` does in
   order, every reshape, what is cached, what differs by layer index, what the serving
   code does differently. `references/method.md` lists what to look for.
3. **Write `facts.ts`.** Every number the page will show: `cfg` for config values,
   `code` for things only the code states, `derive` for everything computed. Include
   the parameter ledger and `parameterTotal`.
4. **Plan the sections** against `references/levels-of-detail.md`, which lists the
   required levels and what each must contain. Write `coverage.json` as you go.
5. **Build the sections** with the kit (`references/kit.md`). Style rules are in
   `references/style.md`; interaction requirements in `references/playability.md`;
   how to make it teach in `references/study-material.md`.
6. **Check, look, fix.** `npm run format -- <slug>`, then `npm run check -- <slug>`
   until every gate passes or is honestly skipped. Then `npm run shoot -- <slug>` and
   open every PNG in `runs/shots/<slug>/` (light, dark, mobile). Put each beside the
   matching image in `examples/` and fix whatever makes yours worse: crowding, tangled
   wires, a figure with nothing said about it, uneven spacing. Repeat. Expect three
   rounds. Passing gates means nothing is broken; it does not mean the page is good.
7. **Self-review** with `references/review-rubric.md`, then write
   `models/<slug>/REPORT.md`: what is drawn, what is excluded and why, what you could
   not verify, and the final check table.

## Rules that are not negotiable

- One exact checkpoint. Never describe "the family".
- No number in a section file that does not come from `facts.ts`. No invented branch,
  op or shape to make a diagram tidier. If you are unsure what the code does, read it
  again; if it is still unclear, say so on the page.
- Do not paste upstream source into the page or the repo. Link to pinned lines instead
  (the kit does this from `symbols.json`).
- Do not edit `kit/`, `tools/` or the skill to make a gate pass. If a gate is wrong,
  say so in `REPORT.md`.
- A gate that cannot run is reported as skipped, with the reason. Never as passed.
