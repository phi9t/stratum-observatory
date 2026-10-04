import { Ledger, Section, Takeaways, n } from "../../../kit";
import * as f from "../facts";
export function Weights() {
  return (
    <Section
      id="weights"
      title="Where the weights live"
      lede="The dense MLP dominates the ledger. The embedding also supplies the output head, so that matrix is counted once."
      source={[
        { source: "reference", symbol: "LlamaForCausalLM" },
        { source: "reference", symbol: "LlamaMLP" },
      ]}
    >
      <Ledger
        caption="Constructor parameter ledger"
        unit="parameters"
        rows={[
          {
            label: "Token embedding",
            detail: `${n(f.vocab.value)} × ${n(f.hidden.value)}`,
            value: f.tokenEmbedParams.value,
            lane: "merged",
          },
          {
            label: "Attention projections",
            detail: `${n(f.layers.value)} layers; bias-free Q/K/V and output`,
            value: f.layers.value * f.attentionParams.value,
            lane: "high",
          },
          {
            label: "Gated MLP",
            detail: "Gate, up and down projections.",
            value: f.layers.value * f.mlpParams.value,
            lane: "low",
          },
          {
            label: "Residual RMSNorm gains",
            detail: "Pre-attention, pre-MLP and final norms.",
            value: f.layers.value * f.blockNormParams.value + f.finalNormParams.value,
            lane: "merged",
          },
        ]}
      />
      <Takeaways
        items={[
          `The exact total is ${n(f.parameterTotal.value)}, equal to the hub's safetensors total.`,
          `Tying the output head avoids an additional ${n(f.tokenEmbedParams.value)} learned scalars.`,
          "RoPE frequencies are nonpersistent buffers, not learned parameters.",
        ]}
      />
    </Section>
  );
}
