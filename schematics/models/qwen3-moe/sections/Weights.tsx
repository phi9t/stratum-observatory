import { Section, Ledger, Prose, Takeaways, Check, n } from "../../../kit";
import * as f from "../facts";
export function Weights() {
  return (
    <Section
      id="weights"
      title="Where the weights live"
      source={[
        { source: "reference", symbol: "Qwen3MoeSparseMoeBlock" },
        { source: "reference", symbol: "Qwen3MoeForCausalLM" },
      ]}
      lede="Count the live parameter shapes, including every expert. Sparse execution reduces the matrices read per token, not the number of stored expert matrices."
    >
      <Ledger
        caption="Exact checkpoint parameter ledger"
        unit="Parameters"
        rows={[
          {
            label: "Token embedding",
            detail: `${n(f.vocab.value)} × ${n(f.hidden.value)}`,
            value: f.embeddingParams.value,
            lane: "merged",
          },
          {
            label: "Untied output head",
            detail: "Separate vocabulary projection; same dimensions as the embedding",
            value: f.embeddingParams.value,
            lane: "merged",
          },
          {
            label: "Attention + QK gains",
            detail: `${n(f.layers.value)} × (Q + K + V + O + shared head-width Q/K gains)`,
            value: f.attentionTotal.value,
            lane: "high",
          },
          {
            label: "Expert banks",
            detail: `${n(f.layers.value)} × ${n(f.experts.value)} × ${n(f.matricesPerExpert.value)} × ${n(f.hidden.value)} × ${n(f.expertWidth.value)}`,
            value: f.expertTotal.value,
            lane: "low",
          },
          {
            label: "Routers",
            detail: `${n(f.layers.value)} × ${n(f.experts.value)} × ${n(f.hidden.value)}`,
            value: f.routerTotal.value,
            lane: "instance",
          },
          {
            label: "Residual norms",
            detail: `Input and post-attention gains per layer, plus final gain (${((f.normTotal.value / f.parameterTotal.value) * f.PERCENT).toPrecision(f.DECIMALS)}% of parameters)`,
            value: f.normTotal.value,
            lane: "merged",
          },
        ]}
        footnote="Equals the pinned hub safetensors.total exactly; nonpersistent RoPE frequencies add no stored parameters."
      />
      <Prose>
        The expert bank accounts for {((f.expertTotal.value / f.parameterTotal.value) * f.PERCENT).toFixed(f.DECIMALS)}%
        of stored parameters. An explicitly defined active matrix budget—selected expert matrices, all
        attention/router/norm weights, the output head and final norm—is {n(f.activeMatrixParams.value)} per token. It
        excludes the embedding lookup table and is an accounting convention, not a measured runtime or proof of the
        checkpoint name's rounded active count.
      </Prose>
      <Check question="If top-k doubled while expert count stayed fixed, would this ledger double?">
        No. Stored expert shapes would stay the same. More expert matrices would execute per token, while attention,
        router and norm execution would remain unchanged.
      </Check>
      <Takeaways
        items={[
          "All experts belong in the stored parameter ledger, even if a particular token does not visit them.",
          "The separate output head is counted once in addition to the input embedding; there is no weight tying.",
        ]}
      />
    </Section>
  );
}
