import { Check, n, OpGraph, Section, shape, Takeaways, type Edge, type Op } from "../../../kit";
import * as f from "../facts";
import { useLayer } from "./Stack";

export function Block() {
  const [layer] = useLayer();
  const d = f.hidden.value;
  const x = shape("B", "S", d);
  const ref = [
    { source: "reference", symbol: "GPT2Block.forward" },
    { source: "serving", symbol: "GPT2Block.forward" },
  ];
  const ops: Op[] = [
    {
      id: "in",
      width: d,
      label: "hidden_states",
      sub: x,
      kind: "tensor",
      col: 0,
      row: 0,
      detail: { what: `The residual stream entering layer ${n(layer)}.`, output: ["hidden_states", x], refs: ref },
    },
    {
      id: "ln1",
      width: d,
      label: "LayerNorm ln_1",
      sub: n(d),
      col: 1,
      row: 0,
      detail: {
        what: "Normalises each position's vector before attention reads it. The stream itself is left untouched: this is pre-norm.",
        formula: `y = (x − mean(x)) / sqrt(var(x) + ${String(f.normEps.value)}) · weight + bias`,
        inputs: [["x", x]],
        output: ["y", x],
        params: [
          ["weight", shape(d)],
          ["bias", shape(d)],
        ],
        refs: ref,
      },
    },
    {
      id: "attn",
      width: d,
      label: "self-attention",
      sub: n(d),
      lane: "high",
      col: 2,
      row: 0,
      detail: {
        what: "Every position gathers information from itself and earlier positions. Drawn op by op in the next section.",
        inputs: [["x", x]],
        output: ["attn_output", x],
        refs: [
          { source: "reference", symbol: "GPT2Attention.forward" },
          { source: "serving", symbol: "GPT2Attention.forward" },
        ],
      },
    },
    {
      id: "add1",
      label: "add",
      kind: "sum",
      col: 3,
      row: 0,
      detail: {
        what: "Adds attention's output to the stream it started from.",
        formula: "hidden_states = attn_output + residual",
        inputs: [
          ["attn_output", x],
          ["residual", x],
        ],
        output: ["hidden_states", x],
        refs: ref,
      },
    },
    {
      id: "ln2",
      width: d,
      label: "LayerNorm ln_2",
      sub: n(d),
      col: 4,
      row: 0,
      detail: {
        what: "Normalises the updated stream before the MLP reads it.",
        formula: `y = (x − mean(x)) / sqrt(var(x) + ${String(f.normEps.value)}) · weight + bias`,
        inputs: [["x", x]],
        output: ["y", x],
        params: [
          ["weight", shape(d)],
          ["bias", shape(d)],
        ],
        refs: ref,
      },
    },
    {
      id: "mlp",
      width: d,
      label: "MLP",
      sub: `${n(d)} → ${n(f.mlpWidth.value)} → ${n(d)}`,
      lane: "low",
      col: 5,
      row: 0,
      detail: {
        what: "Transforms each position on its own; no information moves between positions here.",
        inputs: [["x", x]],
        output: ["feed_forward_hidden_states", x],
        refs: [
          { source: "reference", symbol: "GPT2MLP.forward" },
          { source: "serving", symbol: "GPT2MLP.forward" },
        ],
      },
    },
    {
      id: "add2",
      label: "add",
      kind: "sum",
      col: 6,
      row: 0,
      detail: {
        what: "Adds the MLP's output to the stream.",
        formula: "hidden_states = residual + feed_forward_hidden_states",
        inputs: [
          ["feed_forward_hidden_states", x],
          ["residual", x],
        ],
        output: ["hidden_states", x],
        refs: ref,
      },
    },
    {
      id: "out",
      width: d,
      label: "hidden_states",
      sub: x,
      kind: "tensor",
      col: 7,
      row: 0,
      detail: {
        what:
          layer === f.layers.value - 1
            ? "The stream leaving the last block, on its way to the final LayerNorm."
            : `The stream leaving layer ${n(layer)}, entering layer ${n(layer + 1)}.`,
        output: ["hidden_states", x],
        refs: ref,
        note: "The reference block also has a cross-attention branch; it exists only when config.add_cross_attention is set, which this checkpoint does not do.",
      },
    },
  ];
  const edges: Edge[] = [
    { from: "in", to: "ln1" },
    { from: "ln1", to: "attn" },
    { from: "attn", to: "add1" },
    { from: "in", to: "add1", kind: "skip" },
    { from: "add1", to: "ln2" },
    { from: "ln2", to: "mlp" },
    { from: "mlp", to: "add2" },
    { from: "add1", to: "add2", kind: "skip" },
    { from: "add2", to: "out" },
  ];
  return (
    <Section
      id="block"
      lede="Two sublayers, each wrapped the same way: normalise a copy, transform it, add the result back. The stream is only ever added to."
      source={[{ source: "reference", symbol: "GPT2Block" }]}
      title="Anatomy of a block"
    >
      <OpGraph
        caption="Click a box, or step with the buttons, to inspect one operation. The bar under a box is the width of its output, relative to the widest in that figure; here every operation keeps the stream's width."
        edges={edges}
        id="block"
        ops={ops}
        title={`GPT-2 block, layer ${n(layer)}`}
      />
      <Check question="If you deleted both LayerNorms inside a block, would any tensor change shape?">
        No. A LayerNorm rescales each position's vector and keeps its width, so every shape in the drawing would stay{" "}
        {shape("B", "S", d)}. What changes is the scale of what attention and the MLP read, which is why training
        without them is unstable.
      </Check>
      <Takeaways
        items={[
          "A block never overwrites the stream. It adds two corrections to it: one from attention, one from the MLP.",
          "Normalisation sits on the branch, before each sublayer, not on the stream. That is what pre-norm means.",
        ]}
      />
    </Section>
  );
}
