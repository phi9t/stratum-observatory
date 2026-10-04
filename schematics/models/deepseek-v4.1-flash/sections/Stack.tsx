import { LayerMap, Section, Sentence, Table, Takeaways, useUrlNumber, n } from "../../../kit";
import * as f from "../facts";
export const KINDS = [
  { id: "window", label: "Window only", lane: "merged" },
  { id: "kv", label: "KV + index source", lane: "high" },
  { id: "index", label: "Index source / reused KV", lane: "instance" },
  { id: "reuse", label: "Reused KV + indices", lane: "skip" },
] as const;
export const useLayer = () => useUrlNumber("layer", 2, 0, f.layers.value - 1);
export function Stack() {
  const [layer, setLayer] = useLayer();
  return (
    <Section
      id="stack"
      title="Who builds state, who reuses it"
      source={[{ source: "reference", symbol: "Attention.__init__" }]}
      lede="Select any backbone layer. The block and attention sections follow it; each source writes before its consumers read."
    >
      <LayerMap
        label="Backbone layers · publisher zero-based IDs"
        count={f.layers.value}
        kinds={KINDS}
        kindOf={f.kindOf}
        selected={layer}
        onSelect={setLayer}
      />
      <Sentence>
        Layer {n(layer)}: {f.layerDescription(layer)}{" "}
        {f.engramLayers.value.includes(layer) ? "An Engram lookup precedes this block." : ""}
      </Sentence>
      <Table
        caption="Ownership at the selected layer"
        head={["Property", "Current value"]}
        rows={[
          ["Compression ratio", n(f.ratios.value[layer])],
          ["Compressed KV owner", f.ratios.value[layer] ? n(f.owner(layer, f.kvSources.value)!) : "none"],
          ["Index owner", f.ratios.value[layer] ? n(f.owner(layer, f.indexSources.value)!) : "none"],
          ["Candidate source", layer >= f.candidateSource.value ? n(f.candidateSource.value) : "not yet"],
          ["Engram", f.engramLayers.value.includes(layer) ? "before attention" : "absent"],
        ]}
      />
      <Takeaways
        items={[
          "Every backbone layer has MoE; there is no initial dense-FFN branch.",
          "Compression ratio describes what a layer reads. Source membership determines what it constructs.",
          "Layer IDs here match the zero-based publisher code.",
        ]}
      />
    </Section>
  );
}
