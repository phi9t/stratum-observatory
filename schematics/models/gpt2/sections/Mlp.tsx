import { n, OpGraph, Section, shape, Takeaways, type Edge, type Op } from "../../../kit";
import * as f from "../facts";

export function Mlp() {
  const d = f.hidden.value;
  const w = f.mlpWidth.value;
  const x = shape("B", "S", d);
  const wide = shape("B", "S", w);
  const ref = [
    { source: "reference", symbol: "GPT2MLP.forward" },
    { source: "serving", symbol: "GPT2MLP.forward" },
  ];
  const ops: Op[] = [
    {
      id: "m-in",
      width: d,
      label: "normed stream",
      sub: x,
      kind: "tensor",
      col: 0,
      row: 0,
      detail: { what: "Output of ln_2.", output: ["hidden_states", x], refs: ref },
    },
    {
      id: "m-fc",
      width: w,
      label: "c_fc",
      sub: `${n(d)} → ${n(w)}`,
      lane: "low",
      col: 1,
      row: 0,
      detail: {
        what: `Widens each position ${n(f.mlpRatio.value)} times. The config gives no inner width, so the block uses ${n(f.mlpRatio.value)} × the stream width.`,
        formula: "h = x @ W + b",
        inputs: [["x", x]],
        output: ["h", wide],
        params: [
          ["weight", shape(d, w)],
          ["bias", shape(w)],
        ],
        refs: [...ref, { source: "reference", symbol: "GPT2Block" }],
      },
    },
    {
      id: "m-act",
      width: w,
      label: "GELU (tanh form)",
      sub: n(w),
      lane: "low",
      col: 2,
      row: 0,
      detail: {
        what: `The activation named "${f.activation.value}": a smooth gate that passes large positive values and suppresses negative ones.`,
        formula: f.geluFormula.value,
        inputs: [["h", wide]],
        output: ["h", wide],
        refs: [{ source: "activations", symbol: "NewGELUActivation" }, ...ref],
      },
    },
    {
      id: "m-proj",
      width: d,
      label: "c_proj",
      sub: `${n(w)} → ${n(d)}`,
      lane: "low",
      col: 3,
      row: 0,
      detail: {
        what: "Projects back to the stream width.",
        formula: "y = h @ W + b",
        inputs: [["h", wide]],
        output: ["y", x],
        params: [
          ["weight", shape(w, d)],
          ["bias", shape(d)],
        ],
        refs: ref,
        note: "Dropout follows in training only.",
      },
    },
    {
      id: "m-out",
      width: d,
      label: "to residual add",
      sub: x,
      kind: "tensor",
      col: 4,
      row: 0,
      detail: { what: "Added to the stream by the block.", output: ["feed_forward_hidden_states", x], refs: ref },
    },
  ];
  const edges: Edge[] = [
    { from: "m-in", to: "m-fc" },
    { from: "m-fc", to: "m-act" },
    { from: "m-act", to: "m-proj" },
    { from: "m-proj", to: "m-out" },
  ];
  return (
    <Section
      id="mlp"
      lede="Applied to every position separately and identically. Two thirds of each block's weights are here."
      source={[{ source: "reference", symbol: "GPT2MLP" }]}
      title="The MLP"
    >
      <OpGraph edges={edges} id="mlp" ops={ops} title="GPT-2 MLP" />
      <Takeaways
        items={[
          "The MLP cannot see other positions. Whatever it computes about a token, it computes from that token's vector alone.",
          `It works in a space ${n(f.mlpRatio.value)} times wider than the stream and projects back, which is where most of a block's parameters are.`,
        ]}
      />
    </Section>
  );
}
