import { Circuit } from "./Circuit";
import { Figure, Section, Table, Takeaways, Prose, n, shape } from "../../../kit";
import * as f from "../facts";
import { useLayer } from "./Stack";
export function Overview() {
  const [layer] = useLayer();
  const x = shape("B", "S", f.hidden.value);
  return (
    <Section
      id="overview"
      title="The whole circuit"
      lede="Token ids enter a dense decoder. Positions rotate queries and keys inside attention; they are never added to the embedding stream."
      source={[
        { source: "reference", symbol: "LlamaModel" },
        { source: "reference", symbol: "LlamaForCausalLM" },
      ]}
    >
      <Figure
        label="Whole Llama circuit"
        caption="The output head reuses the token embedding matrix. The highlighted layer follows the stack selector."
      >
        <Circuit layer={layer} />
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
          ["Output head", "Score each vocabulary entry with the embedding weights.", shape("B", "S", f.vocab.value)],
        ]}
      />
      <Takeaways
        items={[
          "Position information enters the attention branches, not an additive position embedding.",
          "The input lookup and output head share the same learned matrix.",
        ]}
      />
    </Section>
  );
}
