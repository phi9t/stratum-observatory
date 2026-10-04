import { LayerMap, Section, Sentence, Table, Takeaways, n, useUrlNumber } from "../../../kit";
import * as f from "../facts";
export const KINDS = [
  { id: "dense", label: "MLA + dense FFN", lane: "merged" },
  { id: "moe", label: "MLA + experts", lane: "low" },
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
      lede="Attention stays the same throughout the stack. The feed-forward path changes from a wide dense transform to a bank of selectively activated experts."
    >
      <LayerMap
        label="Main decoder layers, zero-based"
        kinds={KINDS}
        kindOf={f.kindOf}
        count={f.layers.value}
        selected={layer}
        onSelect={setLayer}
      />
      <Sentence>
        Layer {n(layer)} uses{" "}
        {f.kindOf(layer) === "dense"
          ? `a dense FFN of width ${n(f.denseWidth.value)}`
          : `${n(f.topK.value)} routed experts plus ${n(f.shared.value)} shared expert, each width ${n(f.expertWidth.value)}`}
        . The block and FFN drawings below follow this choice.
      </Sentence>
      <Table
        caption="The exact constructor condition"
        head={["Kind", "Indices", "Feed-forward parameters per layer"]}
        rows={[
          ["Dense", `0…${n(f.firstDense.value - 1)}`, n(f.denseParams.value)],
          ["MoE", `${n(f.firstDense.value)}…${n(f.layers.value - 1)}`, n(f.moeParams.value)],
        ]}
      />
      <p className="ms-prose">
        The code selects experts when <code>{f.layerRule.value}</code>. These are absolute, zero-based indices in the
        reference; the stored MTP layer is outside this main stack.
      </p>
      <Takeaways
        items={[
          `${n(f.denseCount.value)} dense layers precede ${n(f.moeCount.value)} MoE layers.`,
          "Every layer has its own weights and cache; selecting another layer does not change the attention architecture.",
        ]}
      />
    </Section>
  );
}
