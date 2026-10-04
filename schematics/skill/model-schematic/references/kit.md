# The kit

Import everything from `../../kit` (or `../../../kit` inside `sections/`). The
source is short and commented: read `kit/components/*.tsx` when a prop is unclear.
`models/gpt2/` is a complete page built with it. `kit/gallery/Page.tsx` is a widget
catalogue (`npm run build -- gallery`, then open `dist/gallery.html`) showing `LayerMap`
with several kinds, `MoeRouter`, `AttentionGrid` patterns and `RopeDials` wired to URL
state; its numbers are placeholders, so copy the wiring, never the values.

## Page skeleton

```tsx
// Page.tsx
import { Schematic, n } from "../../kit";
import * as f from "./facts";
import { Overview } from "./sections/Overview";
// …

export const sections = [
  { id: "overview", label: "Overview" },
  { id: "data", label: "Data" },
  // …
] as const;
export const sectionIds = sections.map((s) => s.id);   // required export

export function Page() {
  return (
    <Schematic
      name="GPT-2"
      eyebrow="Model schematic · openai-community/gpt2"
      tagline="…one sentence…"
      spec={[{ value: n(f.layers.value), label: "layers" }, …]}
      authority={<>Drawn from <SourceRef source="reference" symbol="GPT2Model" /></>}
      sections={sections}
    >
      <Overview />
      …
    </Schematic>
  );
}
```

Each section file exports one component that returns
`<Section id title source={[{source, symbol}]} lede>…</Section>`. The `id` must be in
`sections`. `source` refs become the linked eyebrow.

## Components

| Component | Use |
|---|---|
| `Schematic`, `Section`, `Sub`, `Prose` | Page frame, section, sub-heading, paragraph (≤ 44rem). |
| `Figure` | Panel around one SVG. `OpGraph` wraps itself; use `Figure` for your own SVGs and for widgets (`AttentionGrid`, `MoeRouter`, `RopeDials`, `TensorScale`). |
| `OpGraph` | A graph of operations with stepping and an inspector. The main tool. |
| `StackOverview` | The page's first figure: stages left and right of a plate of layer tiles coloured by kind, side inputs below, legend. Pass `selected` so it follows the layer map. |
| `LayerMap` | Selectable tile per layer, coloured by kind, with a counted legend. |
| `Pipes` / `Pipe` | A linear sequence as chips and pills (`tensorItem`, `opItem`). Wraps on mobile. |
| `Tensor` | Inline named tensor chip. |
| `Table` | Hairline table; `dense` for long numeric tables; `current` marks a row. |
| `Ledger` | Parameter or byte budget with share bars and a total. |
| `TensorScale` | Tensors to scale across stages, joined by ribbons. |
| `AttentionGrid` | Query × key grid from a `cell(q, k)` function; click a row or drive `query` from a slider. |
| `MoeRouter` + `routeToken` | Routing demo; `routeToken` is the pure arithmetic. |
| `RopeDials` | Rotation of chosen frequency pairs at a position. |
| `Slider`, `Segments`, `Stepper`, `Button` | Controls. |
| `Sentence`, `Readout` | The words and exact values under an interactive figure. |
| `FactList`, `Boundary` | Short facts with hairlines; the scope note. |
| `Takeaways`, `Check`, `Glossary` | Study aids: what to remember, a self-check question with a hidden answer, terms defined once. See `study-material.md`. |
| `Node`, `Wire`, `Junction`, `ArrowDefs`, `ColumnLabel`, `Note`, `nodeWidth` | Hand-laid SVG. |
| `SourceRef` | Link to a pinned symbol with its line range. |

`ref` (from `kit/ref/tensor.ts`) is the tensor library for the clean reference
implementation: `ref.Ctx`, `affine`, `matmul`, `rmsNorm`, `softmax`, `topk`, … It runs
in shape mode or value mode; see `reference-implementation.md`. `Heatmap` draws a
computed matrix.

Helpers: `cfg`, `code`, `derive`, `n`, `approx`, `shape` (facts);
`useUrlNumber(key, fallback, min, max)`, `useUrlChoice(key, fallback, allowed)` (URL
state); `rng`, `normals` (seeded synthetic inputs).

## OpGraph

```tsx
const ops: Op[] = [
  { id: "x", label: "hidden", sub: shape("B", "S", d), kind: "tensor", col: 0, row: 0,
    detail: { what: "Residual stream entering the block.", output: ["x", shape("B", "S", d)],
              refs: [{ source: "reference", symbol: "GPT2Block" }] } },
  { id: "ln1", label: "LayerNorm", sub: n(d), col: 1, row: 0, lane: "merged",
    detail: { what: "…", formula: "y = (x − mean) / sqrt(var + eps) · g + b",
              inputs: [["x", shape("B", "S", d)]], output: ["y", shape("B", "S", d)],
              params: [["weight", shape(d)], ["bias", shape(d)]],
              refs: [{ source: "reference", symbol: "GPT2Block" }] } },
  { id: "add1", label: "add", kind: "sum", col: 3, row: 0, detail: { … } },
];
const edges: Edge[] = [
  { from: "x", to: "ln1" },
  { from: "x", to: "add1", kind: "skip" },      // routed above the row, dashed teal
  { from: "attn", to: "cache", kind: "state" }, // dotted violet
];
<OpGraph id="block" title="One transformer block" ops={ops} edges={edges} />
```

- `col`/`row` place ops on a grid; the kit sizes boxes from text and routes
  orthogonal wires. Put the main path on row 0 left to right; side paths (a gate, a
  cache, a parallel branch) on row 1 or 2.
- `kind`: `op` (default), `tensor`, `state` (dashed, for caches), `sum`, `mul`.
- List ops in execution order: stepping follows the array.
- `id` (the prop) is the URL key and must be unique per page.
- `sub` is short (a width or a shape). Long text goes in `detail`.
- `width` is the op's output width per token. Give it on every data-path op: the
  gauge under the box shows bottlenecks and expansions at a glance.
- A graph should fit one band (about 1240px: roughly eight columns). A single-row
  chain that is longer wraps cleanly onto a second band. A multi-row graph that wraps
  tangles, and the layout gate fails it: split it into two graphs at a natural seam
  (e.g. "build q, k, v" and "attend and project").
- `heading` puts a sub-heading above the graph. Give one whenever a section has more
  than one graph, and put a sentence of prose between graphs saying how they connect.
- `sub` is at most 24 characters: one width or one shape. Two shapes joined with "|"
  do not fit; name the op after what it makes and put both shapes in `detail.output`.
- Build ops with a function of the selected layer or variant when they differ, so the
  graph is always the true one.

## Hand-laid figures

Use for the whole-model overview. Pattern (see `models/gpt2/sections/Overview.tsx`):
an `<svg className="ms-fig" role="img" aria-labelledby>` with a `<title>`,
`<ArrowDefs id>`, `ColumnLabel`s, `Node`s and `Wire`s. Geometry numbers are allowed as
JSX props and in `const LAYOUT… = {…}` objects; every label and count still comes from
facts.

## URL keys

`layer`, `op` ids, `q`, `seed`, `batch`, `ctx` are conventional. Clamp ranges from
facts (`useUrlNumber("layer", 0, 0, layers.value - 1)`).
