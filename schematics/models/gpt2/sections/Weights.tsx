import { Ledger, Prose, Section, n } from "../../../kit";
import * as f from "../facts";

export function Weights() {
  const L = f.layers.value;
  return (
    <Section
      id="weights"
      lede="Counted from the shapes the constructors allocate, then reconciled with what the checkpoint file holds."
      source={[
        { source: "reference", symbol: "GPT2PreTrainedModel" },
        { source: "reference", symbol: "GPT2LMHeadModel" },
      ]}
      title="Where the weights live"
    >
      <Ledger
        caption="Parameters of GPT2LMHeadModel"
        footnote="the output head is tied to the token embedding and adds nothing"
        rows={[
          {
            label: "Token embedding",
            detail: `${n(f.vocab.value)} × ${n(f.hidden.value)}, shared with the head`,
            value: f.tokenEmbedParams.value,
            lane: "high",
          },
          {
            label: "Position embedding",
            detail: `${n(f.positions.value)} × ${n(f.hidden.value)}`,
            value: f.positionEmbedParams.value,
            lane: "instance",
          },
          {
            label: "Attention",
            detail: `${n(L)} × (c_attn + c_proj)`,
            value: L * f.attentionParams.value,
            lane: "high",
          },
          { label: "MLP", detail: `${n(L)} × (c_fc + c_proj)`, value: L * f.mlpParams.value, lane: "low" },
          {
            label: "LayerNorms",
            detail: `${n(L)} × 2, plus the final one`,
            value: L * f.blockNormParams.value + f.finalNormParams.value,
            lane: "merged",
          },
        ]}
        unit="Parameters"
      />
      <Prose>{f.parameterNote}</Prose>
    </Section>
  );
}
