# Review rubric

Score each line pass / fail with one sentence of evidence. A page is done when every
line passes or the failure is a stated, justified scope decision.

## Fidelity
1. Checkpoint identity is exact (repo and commit) and shown on the page.
2. `npm run check` : pins, facts, coverage, parameters pass (or parameters has a
   numeric, specific `parameterNote`).
3. Pick five ops at random across the attention and FFN graphs. For each, open the
   pinned source at the cited lines: the op exists there, in that order, with that
   shape. Record which five.
4. Every layer kind the code distinguishes appears in the layer map, with the exact
   condition from the code.
5. The serving section states at least one concrete thing the serving implementation
   does differently, with a ref, or states what was compared and found identical.
6. No number in the prose or figures is absent from `facts.ts`.

## Coverage of detail
7. All levels in `levels-of-detail.md` are present.
8. The attention graph has every projection, norm, positional step, mask and cache
   interaction as its own op; the FFN/MoE graph likewise.
9. Every op inspector has what/formula/shapes/refs filled; none is a placeholder.
10. The page contains at least two observations a reader would not get from the
    config alone.

## Playability
11. Layer selection changes the block view and the sentence.
12. Each distinctive mechanism has a widget that runs the checkpoint's real rule.
13. Every control changes the page; the URL restores the view after reload.

## Aesthetics
14. Screenshots reviewed in light, dark and mobile; no clipped or overlapping text.
15. Lane colours are used consistently with the stated legend.
16. Each section's lede states its point; captions say what to notice.
17. The page reads as one design.

## Honesty
18. `REPORT.md` lists exclusions with reasons, unverified items, and skipped gates.
