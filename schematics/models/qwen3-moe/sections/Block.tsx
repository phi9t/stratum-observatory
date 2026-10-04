import { Section, OpGraph, Sentence, Check, Takeaways, n, shape, type Op, type Edge } from "../../../kit";
import * as f from "../facts";
const LAYOUT = { cols: [0, 1, 2, 3, 4, 5, 6, 7] };
export function Block({ layer }: { layer: number }) {
  const s = shape("B", "S", f.hidden.value),
    name = `model.layers.${n(layer)}`;
  const make = (
    id: string,
    label: string,
    col: number,
    what: string,
    formula: string,
    params: readonly (readonly [string, string])[] = [],
    lane: Op["lane"] = "merged",
    kind: Op["kind"] = "op",
  ): Op => ({
    id,
    label,
    col,
    row: 0,
    lane,
    kind,
    width: f.hidden.value,
    sub: n(f.hidden.value),
    detail: {
      what,
      formula,
      inputs: [["stream", s]],
      output: [id, s],
      params,
      refs: [
        { source: "reference", symbol: "Qwen3MoeDecoderLayer.forward" },
        { source: "serving", symbol: "Qwen3MoeDecoderLayer.forward" },
      ],
      note: `Parameter namespace: ${name}. vLLM carries a separate residual and fuses its add into the next norm.`,
    },
  });
  const ops: Op[] = [
    make(
      "x",
      `layer ${n(layer)} input`,
      0,
      "Residual stream entering this selected layer.",
      "x = previous layer output",
      [],
      "merged",
      "tensor",
    ),
    make(
      "input_norm",
      "input RMSNorm",
      1,
      "Normalize a copy before attention reads it.",
      `x / sqrt(mean(x²) + ${String(f.normEps.value)}) · gain`,
      [["input_layernorm.weight", shape(f.hidden.value)]],
    ),
    make(
      "attn",
      "GQA attention",
      2,
      "Mix earlier positions using separate query heads and shared KV heads.",
      "o_proj(concat(softmax(mask(qkᵀ · scale))v))",
      [],
      "high",
    ),
    make(
      "add_attn",
      "attention add",
      LAYOUT.cols[3],
      "Add attention's correction to the unnormalized residual.",
      "a = x + attention(norm(x))",
      [],
      "skip",
      "sum",
    ),
    make(
      "post_norm",
      "post RMSNorm",
      LAYOUT.cols[4],
      "Normalize the updated stream before routing and experts.",
      `a / sqrt(mean(a²) + ${String(f.normEps.value)}) · gain`,
      [["post_attention_layernorm.weight", shape(f.hidden.value)]],
    ),
    make(
      "moe",
      "routed experts",
      LAYOUT.cols[5],
      "Choose experts independently for every position and combine their weighted outputs.",
      "m = Σ selected p_e · expert_e(norm(a))",
      [],
      "low",
    ),
    make(
      "out",
      "expert add",
      LAYOUT.cols[6],
      "Restore the residual path after the sparse expert update.",
      "y = a + m",
      [],
      "skip",
      "sum",
    ),
    make(
      "y",
      `layer ${n(layer)} output`,
      LAYOUT.cols[7],
      "Pass the entire stream to the next decoder layer.",
      "next = y",
      [],
      "merged",
      "tensor",
    ),
  ];
  const edges: Edge[] = ops.slice(1).map((o, i) => ({ from: ops[i].id, to: o.id }));
  edges.push({ from: "x", to: "add_attn", kind: "skip" }, { from: "add_attn", to: "out", kind: "skip" });
  return (
    <Section
      id="block"
      title="Anatomy of a block"
      source={[{ source: "reference", symbol: "Qwen3MoeDecoderLayer.forward" }]}
      lede="Each sublayer normalizes a copy, computes an update, and adds it to the stream. The expert router reads the stream after attention, not the block input."
    >
      <Sentence>
        Inspecting <code>{name}</code>; all operation weights below belong to layer {n(layer)}.
      </Sentence>
      <OpGraph
        id="blockOp"
        title={`Block ${n(layer)} forward`}
        ops={ops}
        edges={edges}
        caption="The dashed bypasses carry the unnormalized residual. Width stays constant across both additions."
      />
      <Check question="Could moving the expert router before the attention addition preserve the computation?">
        No. The router currently reads the normalized post-attention stream; moving it would change both its scores and
        the expert inputs.
      </Check>
      <Takeaways
        items={[
          "RMSNorm rescales a vector without subtracting its mean; it changes values, not shape.",
          "A layer always executes attention. Only expert execution is selected per token.",
        ]}
      />
    </Section>
  );
}
