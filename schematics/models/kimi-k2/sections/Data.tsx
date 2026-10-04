import { Section, Figure, TensorScale, Table, Takeaways, Check, shape, n } from "../../../kit";
import * as f from "../facts";
export function Data() {
  return (
    <Section
      id="data"
      title="Data representation"
      source={[{ source: "reference", symbol: "DeepseekV3Attention.forward" }]}
      lede="Attention briefly compresses the stream, then rebuilds head-specific keys and values. The positional key is shared across heads; values contain no rotary channels."
    >
      <Figure
        label="Latent and reconstructed key value widths"
        caption={`For the same ${n(f.demo.scaleTokens)} tokens, latent serving storage is ${f.cacheRatio.value.toFixed(f.demo.radDigits)} times narrower than the reference's expanded key and value cache.`}
      >
        <TensorScale
          title="Latent cache versus expanded cache"
          pxPerToken={8}
          pxPerWidth={0.008}
          footnote="Equal token counts; rectangle height encodes width. Expanded keys include the shared positional slice."
          stages={[
            {
              label: "Residual",
              tensors: [
                { key: "x", name: "hidden stream", tokens: f.demo.scaleTokens, width: f.hidden.value, lane: "merged" },
              ],
            },
            {
              label: "Latent serving cache",
              tensors: [
                {
                  key: "c",
                  from: "x",
                  name: "latent + rotary key",
                  tokens: f.demo.scaleTokens,
                  width: f.latentWidth.value,
                  lane: "instance",
                },
              ],
            },
            {
              label: "Reference expanded cache",
              tensors: [
                {
                  key: "kv",
                  from: "c",
                  name: "all heads: key + value",
                  tokens: f.demo.scaleTokens,
                  width: f.expandedCache.value,
                  lane: "high",
                },
              ],
            },
          ]}
        />
      </Figure>
      <Table
        caption="Tensor interfaces"
        head={["Tensor", "Shape", "Meaning"]}
        rows={[
          ["Token ids", shape("B", "S"), `Indices into ${n(f.vocab.value)} vocabulary entries.`],
          ["Positions", shape(1, "S"), "Absolute offsets; decoding starts after the cached prefix."],
          ["Query", shape("B", f.heads.value, "S", f.qHead.value), "Content and rotated positional channels per head."],
          ["Content latent", shape("B", "S", f.kvRank.value), "Normalized compressed keys and values."],
          ["Positional key", shape("B", 1, "S", f.rope.value), "One rotated key shared by all heads."],
          ["Logits", shape("B", "S", f.vocab.value), "Unnormalized vocabulary scores; sampling is outside this page."],
        ]}
      />
      <Check question="Does sharing the positional key make all attention heads identical?">
        No. Their content keys, queries and values are different slices of the expanded projections. Only the positional
        key is shared.
      </Check>
      <Takeaways
        items={[
          "The latent and the reconstructed heads represent the same tokens in different spaces.",
          "A cache size depends on which implementation and cache representation actually runs.",
        ]}
      />
    </Section>
  );
}
