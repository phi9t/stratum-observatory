import { Check, OpGraph, Section, Takeaways, n, shape, type Op } from "../../../kit";
import * as f from "../facts";
export function Mlp() {
  const d = f.hidden.value,
    w = f.mlpWidth.value,
    x = shape("B", "S", d),
    wide = shape("B", "S", w);
  const refs = [
    { source: "training", symbol: "Qwen3DecoderLayer" },
    { source: "reference", symbol: "Qwen3MLP.forward" },
    { source: "serving-base", symbol: "Qwen2MLP.forward" },
  ];
  const ops: Op[] = [
    {
      id: "x",
      label: "normalized stream",
      col: 0,
      row: 0,
      kind: "tensor",
      width: d,
      sub: n(d),
      detail: {
        what: "The post-attention RMSNorm output feeds both projections.",
        formula: "x = RMSNorm(residual)",
        output: ["x", x],
        refs,
      },
    },
    {
      id: "gate",
      label: "gate projection",
      col: 1,
      row: 0,
      lane: "low",
      width: w,
      sub: n(w),
      detail: {
        what: "Build the branch whose activation modulates the other branch.",
        formula: "g=x Wgateᵀ",
        inputs: [["x", x]],
        output: ["g", wide],
        params: [["gate_proj.weight", shape(w, d)]],
        refs,
      },
    },
    {
      id: "up",
      label: "up projection",
      col: 1,
      row: 1,
      lane: "low",
      width: w,
      sub: n(w),
      detail: {
        what: "Build an independent content branch.",
        formula: "u=x Wupᵀ",
        inputs: [["x", x]],
        output: ["u", wide],
        params: [["up_proj.weight", shape(w, d)]],
        refs,
      },
    },
    {
      id: "silu",
      label: "SiLU",
      col: 2,
      row: 0,
      lane: "low",
      width: w,
      sub: n(w),
      detail: {
        what: "Apply a smooth gate; it is not a probability or router.",
        formula: "g / (1 + exp(-g))",
        inputs: [["g", wide]],
        output: ["activated gate", wide],
        refs,
      },
    },
    {
      id: "mul",
      label: "multiply",
      col: 3,
      row: 0,
      kind: "mul",
      width: w,
      detail: {
        what: "Combine content and gate coordinate by coordinate.",
        formula: "z=SiLU(g) ⊙ u",
        inputs: [
          ["g", wide],
          ["u", wide],
        ],
        output: ["z", wide],
        refs,
      },
    },
    {
      id: "down",
      label: "down projection",
      col: 4,
      row: 0,
      lane: "low",
      width: d,
      sub: n(d),
      detail: {
        what: "Return to the residual width for addition.",
        formula: "y=z Wdownᵀ",
        inputs: [["z", wide]],
        output: ["y", x],
        params: [["down_proj.weight", shape(d, w)]],
        refs,
      },
    },
  ];
  return (
    <Section
      id="mlp"
      title="The dense gated feed-forward path"
      lede="Every token runs both wide projections. A smooth gate multiplies the content branch before it returns to the stream; no experts are selected."
      source={[{ source: "reference", symbol: "Qwen3MLP" }]}
    >
      <OpGraph
        title="Dense SiLU-gated MLP"
        id="mlpops"
        ops={[
          ...ops.filter((o) => o.id === "x" || o.id === "gate"),
          ...ops.filter((o) => o.id === "silu"),
          ...ops.filter((o) => o.id === "up"),
          ...ops.filter((o) => o.id === "mul" || o.id === "down"),
        ]}
        edges={[
          { from: "x", to: "gate" },
          { from: "x", to: "up" },
          { from: "gate", to: "silu" },
          { from: "silu", to: "mul" },
          { from: "up", to: "mul" },
          { from: "mul", to: "down" },
        ]}
      />
      <Check question="Can the MLP directly read another token’s vector?">
        No. Each projection and multiplication acts on the final axis only. Cross-position information must arrive
        through attention.
      </Check>
      <Takeaways
        items={[
          `Both gate and up branches have width ${n(w)}.`,
          `Each layer spends ${n(f.mlpParams.value)} parameters on this path. There are no routing weights.`,
        ]}
      />
    </Section>
  );
}
