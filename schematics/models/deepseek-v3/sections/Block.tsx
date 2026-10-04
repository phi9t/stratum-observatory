import { Section, OpGraph, Check, Takeaways, Sentence, n, shape, type Op, type Edge } from "../../../kit";
import * as f from "../facts";
import { operation, blockRefs, stream, normFormula } from "../graph";
import { useLayer } from "./Stack";
const LAYOUT = { columns: [0, 1, 2, 3, 4, 5, 6, 7] };

export function Block() {
  const [layer] = useLayer();
  const d = f.hidden.value;
  const kind = f.kindOf(layer);
  const norm = (id: string, col: number, what: string) =>
    operation(
      id,
      "RMSNorm",
      col,
      d,
      {
        what,
        formula: normFormula,
        inputs: [["stream", stream]],
        output: ["normalized", stream],
        params: [["gain", shape(d)]],
        refs: blockRefs,
      },
      "merged",
    );
  const ops: Op[] = [
    operation(
      "x",
      "stream",
      0,
      d,
      {
        what: `Input to main layer ${n(layer)}.`,
        formula: "x = previous layer output",
        inputs: [["previous", stream]],
        output: ["x", stream],
        refs: blockRefs,
      },
      "merged",
      0,
      "tensor",
      n(d),
    ),
    norm("n1", 1, "Normalize each token before attention reads it."),
    operation("a", "MLA", 2, d, {
      what: "Gather from permitted earlier tokens, then project back to the residual width.",
      formula: "a = Attention(norm(x), positions, cache, mask)",
      inputs: [["normalized", stream]],
      output: ["a", stream],
      refs: blockRefs,
    }),
    operation(
      "s1",
      "add",
      LAYOUT.columns[3],
      d,
      {
        what: "Preserve the original stream and add attention's correction.",
        formula: "h = x + a",
        inputs: [
          ["x", stream],
          ["a", stream],
        ],
        output: ["h", stream],
        refs: blockRefs,
      },
      "merged",
      0,
      "sum",
      "",
    ),
    norm("n2", LAYOUT.columns[4], "Normalize the attention-updated stream before feed-forward computation."),
    operation(
      "f",
      kind === "dense" ? "dense FFN" : "MoE + shared",
      LAYOUT.columns[5],
      d,
      {
        what:
          kind === "dense"
            ? "Apply the dense gated transform independently to each position."
            : "Sum the routed experts and the always-on shared transform.",
        formula: kind === "dense" ? "f = down(SiLU(gate(h)) · up(h))" : "f = Σ w_e Expert_e(h) + Shared(h)",
        inputs: [["normalized", stream]],
        output: ["f", stream],
        refs: blockRefs,
      },
      "low",
      0,
      "op",
      n(kind === "dense" ? f.denseWidth.value : f.expertWidth.value),
    ),
    operation(
      "s2",
      "add",
      LAYOUT.columns[6],
      d,
      {
        what: "Add the feed-forward correction to the stream.",
        formula: "y = h + f",
        inputs: [
          ["h", stream],
          ["f", stream],
        ],
        output: ["y", stream],
        refs: blockRefs,
      },
      "merged",
      0,
      "sum",
      "",
    ),
    operation(
      "y",
      "stream",
      LAYOUT.columns[7],
      d,
      {
        what: "Pass the complete updated stream onward.",
        formula: "next = y",
        inputs: [["y", stream]],
        output: ["next", stream],
        refs: blockRefs,
      },
      "merged",
      0,
      "tensor",
      n(d),
    ),
  ];
  const edges: Edge[] = [
    { from: "x", to: "n1" },
    { from: "n1", to: "a" },
    { from: "a", to: "s1" },
    { from: "x", to: "s1", kind: "skip" },
    { from: "s1", to: "n2" },
    { from: "n2", to: "f" },
    { from: "f", to: "s2" },
    { from: "s1", to: "s2", kind: "skip" },
    { from: "s2", to: "y" },
  ];
  return (
    <Section
      id="block"
      title="Anatomy of a block"
      source={[{ source: "reference", symbol: "DeepseekV3DecoderLayer.forward" }]}
      lede="Normalization sits on each branch. Neither sublayer overwrites the stream: both return a correction of the same width."
    >
      <Sentence>
        Layer {n(layer)} follows the {kind === "dense" ? "dense" : "expert"} branch.
      </Sentence>
      <OpGraph id="block-op" title={`Main layer ${n(layer)}`} ops={ops} edges={edges} />
      <Check question="Would removing a branch norm change the residual addition's shape?">
        No. RMSNorm keeps every axis; it changes the values each sublayer reads. Width matching at the residual add does
        not prove normalization is unnecessary.
      </Check>
      <Takeaways
        items={[
          "RMSNorm uses a float32 mean square and learned gain, with no mean subtraction.",
          "vLLM can carry the residual separately and fuse addition with the next norm; this drawing shows the reference's explicit additions.",
        ]}
      />
    </Section>
  );
}
