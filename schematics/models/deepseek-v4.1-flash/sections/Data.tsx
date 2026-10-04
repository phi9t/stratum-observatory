import { Section, Figure, TensorScale, FactList, Takeaways, Check, n, shape } from "../../../kit";
import * as f from "../facts";
export function Data() {
  return (
    <Section
      id="data"
      title="Width is not the head count"
      source={[{ source: "reference", symbol: "Attention" }]}
      lede="Queries expand per head, but the same narrow KV vector serves as both key and value for every head. The output projection is grouped."
    >
      <Figure
        label="Architecture figure"
        caption="Equal token counts; tensor heights show total scalar width. The shared KV is much narrower than the expanded queries."
      >
        <TensorScale
          title="Attention widths to scale"
          pxPerToken={18}
          pxPerWidth={0.004}
          footnote="Height is total channels per token; symbolic head axes are flattened only for this drawing."
          stages={[
            {
              label: "Collapsed input",
              tensors: [{ key: "x", name: "x", tokens: f.toyTokens.value, width: f.hidden.value, lane: "merged" }],
            },
            {
              label: "Query latent",
              tensors: [
                { key: "qr", from: "x", name: "qr", tokens: f.toyTokens.value, width: f.qRank.value, lane: "high" },
              ],
            },
            {
              label: "Queries / shared KV",
              tensors: [
                {
                  key: "q",
                  from: "qr",
                  name: "q heads",
                  tokens: f.toyTokens.value,
                  width: f.qWidth.value,
                  lane: "high",
                },
                { key: "kv", name: "shared kv", tokens: f.toyTokens.value, width: f.headDim.value, lane: "instance" },
              ],
            },
            {
              label: "Output latent",
              tensors: [
                {
                  key: "o",
                  from: "q",
                  name: "grouped output",
                  tokens: f.toyTokens.value,
                  width: f.outputLatent.value,
                  lane: "high",
                },
              ],
            },
          ]}
        />
      </Figure>
      <FactList
        items={[
          ["Residual", <code>{shape("B", "S", f.copies.value, f.hidden.value)}</code>],
          ["Query", <code>{shape("B", "S", f.heads.value, f.headDim.value)}</code>],
          ["Shared KV", <code>{shape("B", "T", f.headDim.value)}</code>],
          ["Logits", <code>{shape("B", "S", f.vocab.value)}</code>],
        ]}
      />
      <Check question="Does doubling query heads necessarily double the number of KV cache scalars?">
        No. Every head reads the same shared KV vector. Query width and grouped output weights change, while the KV
        vector width is independently configured.
      </Check>
      <Takeaways
        items={[
          `Shared KV uses ${n(f.qWidth.value / f.headDim.value)} times fewer scalars than expanded queries per token.`,
          "There are no separate expanded value heads in this reference.",
          "Cache payload, activation precision and physical packing are different quantities.",
        ]}
      />
    </Section>
  );
}
