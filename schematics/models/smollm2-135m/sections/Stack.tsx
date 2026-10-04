import { LayerMap, Section, Sentence, Table, Takeaways, n, useUrlNumber } from "../../../kit";
import * as f from "../facts";
export const KINDS = [{ id: "dense", label: "full causal attention + dense gated MLP", lane: "high" }] as const;
export function useLayer() {
  return useUrlNumber("layer", 0, 0, f.layers.value - 1);
}
export function Stack() {
  const [layer, setLayer] = useLayer();
  return (
    <Section
      id="stack"
      title="The stack"
      lede="Every layer has the same circuit and independent weights. Select a layer to follow its place in the residual stream."
      source={[{ source: "reference", symbol: "LlamaModel" }]}
    >
      <LayerMap
        label="Llama layers"
        count={f.layers.value}
        selected={layer}
        onSelect={setLayer}
        kinds={KINDS}
        kindOf={() => "dense"}
      />
      <Sentence>
        Layer {n(layer)} reads {layer === 0 ? "the token embedding" : `layer ${n(layer - 1)}`} and feeds{" "}
        {layer === f.layers.value - 1 ? "the final RMSNorm" : `layer ${n(layer + 1)}`}.
      </Sentence>
      <Table
        caption="Layer kind conditions"
        head={["Kind", "Condition", "Difference"]}
        rows={[
          [
            "Full causal + dense",
            "LlamaModel constructs the same decoder class at every index.",
            "Only weights and cache ownership vary by layer index.",
          ],
        ]}
      />
      <Takeaways
        items={[
          "No layer is an expert router or a sliding-window layer.",
          "The selected layer uses its own projections and cache.",
        ]}
      />
    </Section>
  );
}
