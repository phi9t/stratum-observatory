import { LayerMap, Section, Sentence, Table, Takeaways, useUrlNumber, n } from "../../../kit";
import * as f from "../facts";
export const KINDS = [
  { id: "dense", label: "latent attention + dense FFN", lane: "merged" },
  { id: "moe", label: "latent attention + experts", lane: "low" },
] as const;
export function useLayer() {
  return useUrlNumber("layer", 0, 0, f.layers.value - 1);
}
export function Stack() {
  const [layer, setLayer] = useLayer();
  return (
    <Section
      id="stack"
      title="The stack"
      source={[{ source: "reference", symbol: "DeepseekV3DecoderLayer" }]}
      lede="Every layer has the same attention. The feed-forward branch changes after the first layer; pick a tile to follow it into the block."
    >
      <LayerMap
        count={f.layers.value}
        selected={layer}
        onSelect={setLayer}
        kindOf={f.kindOf}
        kinds={KINDS}
        label="Kimi K2 layers"
      />
      <Sentence>
        Layer {n(layer)} uses{" "}
        {f.kindOf(layer) === "dense"
          ? `a dense FFN of width ${n(f.dense.value)}`
          : `${n(f.topK.value)} selected experts plus ${n(f.shared.value)} always-on shared expert`}
        . Its attention still has {n(f.heads.value)} heads.
      </Sentence>
      <Table
        caption="The exact zero-based layer condition"
        head={["Kind", "Condition", "Layers", "Inner width"]}
        rows={[
          ["Dense", "index < first_k_dense_replace", n(f.denseLayers.value), n(f.dense.value)],
          [
            "Experts",
            "index ≥ first_k_dense_replace and index % moe_layer_freq = 0",
            n(f.moeLayers.value),
            n(f.expert.value),
          ],
        ]}
      />
      <Takeaways
        items={[
          "The first layer has no router or expert selection.",
          "Selecting an expert layer changes its feed-forward branch, not its attention geometry.",
        ]}
      />
    </Section>
  );
}
