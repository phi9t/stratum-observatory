import { Figure, StackOverview, Section, Table, Takeaways, Prose, n, shape } from "../../../kit";
import * as f from "../facts";
import { KINDS, useLayer } from "./Stack";
export function Overview() {
  const [layer] = useLayer();
  const x = shape("B", "S", f.hidden.value);
  return (
    <Section
      id="overview"
      title="The whole circuit"
      lede="Token ids enter a dense decoder. Positions rotate queries and keys inside attention; they are never added to the embedding stream."
      source={[
        { source: "reference", symbol: "Qwen3Model" },
        { source: "reference", symbol: "Qwen3ForCausalLM" },
      ]}
    >
      <Figure
        label="Whole Qwen3 circuit"
        caption="The embedding and output head are independent matrices. The highlighted layer follows the stack selector."
      >
        <StackOverview
          title="Qwen3-8B forward pass"
          before={[
            { title: "token ids", line: shape("B", "S"), lane: "merged", kind: "tensor" },
            { title: "embedding", line: `${n(f.vocab.value)} × ${n(f.hidden.value)}`, lane: "merged" },
          ]}
          after={[
            { title: "RMSNorm", line: n(f.hidden.value), lane: "merged" },
            { title: "LM head", line: `${n(f.hidden.value)} → ${n(f.vocab.value)}`, lane: "merged" },
            { title: "logits", line: shape("B", "S", f.vocab.value), lane: "merged", kind: "tensor" },
          ]}
          count={f.layers.value}
          kinds={KINDS}
          kindOf={() => "dense"}
          selected={layer}
          stackTitle={`${n(f.layers.value)} pre-norm layers`}
          stackLine={x}
          side={{ title: "positions + mask", line: "shared across the stack", lane: "instance" }}
        />
      </Figure>
      <Prose>
        Ink is the residual stream, blue is attention, orange is the feed-forward path, violet is position and cache
        state, and dashed teal is a residual connection. The bar under each operation measures its output width relative
        to the widest in that figure.
      </Prose>
      <Table
        caption="Stages of the forward pass"
        head={["Stage", "Action", "Output"]}
        rows={[
          ["Embedding", "Select a vocabulary row.", x],
          ["Decoder stack", "Normalize, attend, add; normalize, gate, add.", x],
          ["Final norm", "Rescale the completed stream.", x],
          ["Output head", "Score each vocabulary entry with untied weights.", shape("B", "S", f.vocab.value)],
        ]}
      />
      <Takeaways
        items={[
          "Position information enters the attention branches, not an additive position embedding.",
          "The output head has its own weights.",
        ]}
      />
    </Section>
  );
}
