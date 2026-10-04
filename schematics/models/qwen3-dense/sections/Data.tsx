import { Check, Figure, TensorScale, Section, Table, Takeaways, n, shape } from "../../../kit";
import * as f from "../facts";
export function Data() {
  const tokens = f.DISPLAY.tokens;
  return (
    <Section
      id="data"
      title="Data representation"
      lede="Queries retain the stream width, while each key or value tensor is narrower. Sharing KV heads saves state without reducing the query-head count."
      source={[{ source: "reference", symbol: "Qwen3Attention" }]}
    >
      <Figure
        label="Widths to scale"
        caption={`Each rectangle shows ${n(tokens)} positions. Height represents scalar width; queries use ${n(f.groups.value)} times the width of keys or values.`}
      >
        <TensorScale
          footnote="Height is scalar width; each column represents one position."
          title="Query and cache widths"
          pxPerToken={5}
          pxPerWidth={0.018}
          stages={[
            {
              label: "stream",
              tensors: [{ key: "x", name: "residual", tokens, width: f.hidden.value, lane: "merged" }],
            },
            { label: "queries", tensors: [{ key: "q", name: "Q", tokens, width: f.qWidth.value, lane: "high" }] },
            {
              label: "keys",
              tensors: [{ key: "k", name: "cached K", tokens, width: f.kvWidth.value, lane: "instance" }],
            },
            {
              label: "values",
              tensors: [{ key: "v", name: "cached V", tokens, width: f.kvWidth.value, lane: "instance" }],
            },
            {
              label: "MLP branch",
              tensors: [{ key: "m", name: "gate or up", tokens, width: f.mlpWidth.value, lane: "low" }],
            },
          ]}
        />
      </Figure>
      <Table
        caption="Inputs, intermediates and outputs"
        head={["Tensor", "Shape", "Meaning"]}
        rows={[
          ["Token ids", shape("B", "S"), "Integer vocabulary indices."],
          ["Position ids", shape(1, "S"), "Absolute positions; shared rotary frequencies."],
          ["Q", shape("B", f.heads.value, "S", f.headDim.value), "Normalized then rotated."],
          ["K / V", shape("B", f.kvHeads.value, "S", f.headDim.value), "Rotated K and unrotated V are kept."],
          ["Logits", shape("B", "S", f.vocab.value), "Scores before sampling."],
        ]}
      />
      <Check question="If the query-head count stayed fixed but every query had its own KV head, what would grow?">
        The key and value projections and cache would grow by {n(f.groups.value)} times. Query width and residual width
        would stay unchanged.
      </Check>
      <Takeaways
        items={[
          `Each KV head serves ${n(f.groups.value)} query heads.`,
          `The configured position limit is ${n(f.positions.value)}; this is not a measured quality claim.`,
        ]}
      />
    </Section>
  );
}
