# Style

The look is a technical monograph: quiet paper, precise drawings, serif headings, mono
numbers. `kit/styles/schematic.css` implements it; you get most of it by using kit
components and not fighting them. Do not write new CSS unless a custom figure needs a
class, and then use only the existing tokens (`var(--ms-…)`).

## Page

- **Hero**: mono eyebrow (`Model schematic · <checkpoint>`), the model name large in
  serif, one sentence saying what the page shows, a spec strip of five or six values
  from facts, and a pill naming what the page is drawn from.
- **Sections**: the eyebrow is the pinned source being drawn (the kit renders it from
  `source` refs). The title is serif and plain ("Anatomy of a block", not "Deep dive
  into blocks"). The lede is one or two sentences that state the point of the section:
  what the reader will know after it. Then figure, then tables.
- Sections are separated by a hairline and space. No cards, no numbered badges, no
  icons, no emoji.

## Colour

Colour is meaning, assigned once per page and kept:

| Lane | Use for |
|---|---|
| `merged` (ink) | the main residual stream and neutral structure |
| `high` (blue) | the first named path: attention, queries, or the primary branch |
| `low` (orange) | the second: feed-forward, experts, keys/values, or a second branch |
| `instance` (violet) | side inputs and state: caches, positions, gates, routing |
| `skip` (teal) | residual and skip connections, checkpoints, auxiliary taps |
| yellow mark | the current selection only |

Decide the mapping before drawing, state it in a legend near the first figure, and do
not reuse a lane for a second meaning. Three lanes plus ink is usually enough.

## Drawings

- Fixed pixel size, never scaled down; minimum text 11px. `OpGraph` sizes boxes from
  their text. In hand-laid figures use `<Node>` without `w` (it measures) or
  `nodeWidth()`; never guess a width.
- Tensors and state are square-cornered and tinted; operations are rounded; residual
  wires are dashed teal and run above the row; sums and gates are small circles.
- Wires are horizontal and vertical. Leave room: empty space is where wires go.
- Labels are short. Box title: the operation (`RMSNorm`, `q projection`, `top-k`).
  Second line: the output width or shape. Everything else belongs in the inspector.
- Keep a hand-laid overview under 1120px wide. Keep each `OpGraph` to one band.
- The overview is the page's first impression and must be more than a row of boxes.
  Use `StackOverview` (layer tiles coloured by the same kinds as the layer map, side
  inputs, legend), or hand-lay one as `models/gpt2/sections/Overview.tsx` does when
  the model has structure `StackOverview` cannot show (a tie, two towers, a memory).
- Show the shape of a mechanism before its parts. When a sublayer is split into
  several graphs, open the section with one small topology graph (at most eight
  boxes, two or three rows) in which each box is one of the detailed graphs below:
  the reader must be able to see, in one figure, which paths run in parallel and
  where they meet. Splitting without this turns a circuit into a list.
- A section with several figures needs a sub-heading and a connecting sentence for
  each. Never stack figure after figure with nothing between them.
- Draw to scale when sizes are the point; say what is not to scale.

## Words

- Write for an engineer who has not read the code. Say what a thing does before
  naming it. Define a term once, where it first appears.
- State facts plainly. No "leverages", "robust", "powerful", "seamlessly". No hedging
  that is not a real uncertainty.
- Numbers: plain digits, no thousands separators (`129280`, `[B, S, 7168]`). Use
  `approx()` only beside the exact value. Use `n()` and `shape()` from the kit.
- Captions say what to notice, not what the figure is.

## Code

Write section files the way you would write any source a colleague will read: one
statement per line, one op per object. Run `npm run format -- <slug>` before
checking; the `format` gate fails files prettier would change.

## Check with your eyes

After `npm run shoot`, look at every image and ask:

- Does any text touch or cross a box edge, another label, or a wire?
- Is anything unreadable in dark mode (a hard-coded colour, a tint with no contrast)?
- Is the first screen of each section understandable without scrolling a figure?
- On mobile, does every figure scroll sideways inside its panel while the page does
  not?
- Is there a figure that is just boxes in a row with no information a table would not
  give better? Replace it with a `Pipe` or a table.
- Does the page look like one design, or like several? Uneven spacing and a section
  with three different figure styles are the usual causes.
