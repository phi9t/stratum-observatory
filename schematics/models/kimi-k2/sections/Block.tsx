import { Section, OpGraph, Sentence, Prose, Check, Takeaways, shape, n, type Edge } from "../../../kit";
import * as f from "../facts";
import { useLayer } from "./Stack";
import { op, chain, stream, normFormula } from "./graph";
export function Block() {
  const [layer] = useLayer();
  const kind = f.kindOf(layer);
  const symbol = "DeepseekV3DecoderLayer";
  const ops = [
    op(
      "x",
      "hidden",
      f.hidden.value,
      { col: 0, row: 0 },
      symbol,
      "The stream entering the selected layer.",
      "x",
      stream,
    ),
    op(
      "norm1",
      "RMSNorm",
      f.hidden.value,
      { col: 1, row: 0 },
      symbol,
      "Normalize before attention reads the stream.",
      normFormula,
      stream,
      stream,
      [["input norm gain", shape(f.hidden.value)]],
      "merged",
    ),
    op(
      "attn",
      "latent attention",
      f.hidden.value,
      { col: 2, row: 0 },
      symbol,
      "Exchange information with visible earlier tokens.",
      "Attention(norm(x))",
      stream,
      stream,
      [],
      "high",
      "Open the attention section for its projections and cache boundary.",
    ),
    op(
      "add1",
      "add",
      f.hidden.value,
      { col: 3, row: 0 },
      symbol,
      "Preserve the incoming stream and add the attention update.",
      "a = x + attention",
      stream,
      stream,
      [],
      "skip",
    ),
    op(
      "norm2",
      "RMSNorm",
      f.hidden.value,
      { col: 4, row: 0 },
      symbol,
      "Normalize the updated stream before feed-forward computation.",
      normFormula,
      stream,
      stream,
      [["FFN norm gain", shape(f.hidden.value)]],
      "merged",
    ),
    op(
      "ffn",
      kind === "dense" ? "dense FFN" : "routed + shared",
      f.hidden.value,
      { col: 5, row: 0 },
      symbol,
      kind === "dense"
        ? "Apply the dense gated feed-forward transform."
        : "Choose routed experts and add the always-on shared transform.",
      kind === "dense" ? "down(SiLU(gate(x)) × up(x))" : "Σ selected_weight × expert(x) + shared(x)",
      stream,
      stream,
      [],
      "low",
    ),
    op(
      "add2",
      "add",
      f.hidden.value,
      { col: 6, row: 0 },
      symbol,
      "Add the feed-forward update without replacing the stream.",
      "y = a + ffn",
      stream,
      stream,
      [],
      "skip",
    ),
  ];
  const edges: Edge[] = [
    ...chain(ops),
    { from: "x", to: "add1", kind: "skip" },
    { from: "add1", to: "add2", kind: "skip" },
  ];
  return (
    <Section
      id="block"
      title="Anatomy of a block"
      source={[{ source: "reference", symbol }]}
      lede="Each sublayer reads a normalized copy, computes an update, and adds it back. The bypass keeps the full residual stream."
    >
      <Sentence>
        Selected layer {n(layer)}: {kind === "dense" ? "dense feed-forward" : "mixture of experts"}; input and output
        are {stream}.
      </Sentence>
      <OpGraph id="block-ops" title={`Layer ${n(layer)}: ${kind}`} ops={ops} edges={edges} />
      <Prose>
        The bar beneath each box is its output width relative to the widest output in that figure. Teal dashed edges
        preserve the stream around the sublayers. vLLM carries the residual separately and fuses additions into the
        following norms.
      </Prose>
      <Check question="Why can the dense and expert branches occupy the same residual slot?">
        Both return the original hidden width. Their internal parameter counts and computation differ, but addition
        requires the same stream shape.
      </Check>
      <Takeaways
        items={[
          "Normalization sits before each sublayer.",
          "Expert selection changes the update computation, not the stream's interface.",
        ]}
      />
    </Section>
  );
}
