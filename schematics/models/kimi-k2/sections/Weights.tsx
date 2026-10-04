import { Section, Ledger, Prose, Takeaways, n } from "../../../kit";
import * as f from "../facts";
export function Weights() {
  return (
    <Section
      id="weights"
      title="Where the weights live"
      source={[
        { source: "reference", symbol: "DeepseekV3MoE" },
        { source: "reference", symbol: "DeepseekV3Attention" },
      ]}
      lede="The routed bank dominates stored learned weights. A token executes only its selected experts, while the rest of the decoder remains active."
    >
      <Ledger
        caption="Exact learned-parameter ledger"
        unit="Parameters"
        rows={f.ledger}
        footnote="Embedding and vocabulary head are untied; shared experts counted separately."
      />
      <Prose>{f.parameterNote}</Prose>
      <Prose>
        The active-weight arithmetic is {n(f.activeParams.value)} parameters per token: all non-routed parameters plus
        the selected routed matrices. It includes the full embedding table as a parameter-count convention even though a
        lookup reads a row. It is neither a FLOP estimate nor a measured hardware footprint.
      </Prose>
      <Takeaways
        items={[
          "Sparse execution reduces the routed matrices touched per token, not the number of learned matrices stored.",
          "FP8 scale tensors and saved rotary buffers must not be confused with trainable parameters.",
        ]}
      />
    </Section>
  );
}
