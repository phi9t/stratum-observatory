import { Section, Ledger, Prose, Table, Takeaways, n, SourceRef } from "../../../kit";
import * as f from "../facts";
export function Weights() {
  return (
    <Section
      id="weights"
      title="Where the weights live"
      source={[
        { source: "reference", symbol: "DeepseekV3ForCausalLM" },
        { source: "reference", symbol: "DeepseekV3MoE" },
      ]}
      lede="The routed expert banks dominate storage. Their parameter count is distinct from the subset of matrices used by a token, and from the hub's count of all stored scalars."
    >
      <Ledger
        caption="Main forward parameters, counted from constructor shapes"
        unit="Parameters"
        rows={[
          {
            label: "Embedding",
            detail: `${n(f.vocab.value)} × ${n(f.hidden.value)}`,
            value: f.embeddingParams.value,
            lane: "high",
          },
          { label: "Independent head", detail: "No weight tying; no bias", value: f.headParams.value, lane: "high" },
          {
            label: "Attention",
            detail: `${n(f.layers.value)} sets of latent projections and latent norms`,
            value: f.allAttentionParams.value,
            lane: "high",
          },
          {
            label: "Dense FFNs",
            detail: `${n(f.denseCount.value)} gated transforms`,
            value: f.allDenseParams.value,
            lane: "low",
          },
          {
            label: "Routed experts",
            detail: `${n(f.moeCount.value)} × ${n(f.experts.value)} expert FFNs`,
            value: f.routedParams.value,
            lane: "low",
          },
          {
            label: "Shared experts",
            detail: `${n(f.moeCount.value)} × ${n(f.shared.value)} always-on FFN`,
            value: f.sharedParams.value,
            lane: "low",
          },
          {
            label: "Routers",
            detail: "Router matrices plus correction biases",
            value: f.allRouterParams.value,
            lane: "instance",
          },
          {
            label: "Block + final norms",
            detail: "Latent norms are counted with attention",
            value: f.allNormParams.value,
            lane: "merged",
          },
        ]}
        footnote="Inference modules constructed by the pinned reference"
      />
      <Prose>{f.parameterNote}</Prose>
      <Table
        caption="Reconciliation with stored checkpoint scalars"
        head={["Added to main forward", "Stored scalars", "Evidence and scope"]}
        rows={[
          [
            "Unique MTP parameters",
            n(f.mtpUnique.value),
            <>
              Auxiliary MoE block, fusion projection and norms; <SourceRef source="mtp" symbol="MTPTransformerBlock" />.
            </>,
          ],
          [
            "MTP embedding + head entries",
            n(f.mtpDuplicates.value),
            "Separate names in the pinned index; README_WEIGHTS describes conceptual sharing.",
          ],
          [
            "FP8 scale entries",
            n(f.scaleScalars.value),
            `${n(f.scaleTensorCount.value)} scale tensors; ceil(out/block) × ceil(in/block).`,
          ],
          [
            "Full stored total",
            n(f.storedTotal.value),
            "Matches hub safetensors.total exactly; scalar count is not bytes.",
          ],
        ]}
      />
      <Takeaways
        items={[
          "Changing the selected token's experts changes active computation, not the number of model parameters.",
          "A quantization scale is stored metadata; it is not another learned matrix in the main forward architecture.",
        ]}
      />
    </Section>
  );
}
