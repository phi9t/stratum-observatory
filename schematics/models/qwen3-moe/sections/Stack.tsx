import { Section, LayerMap, Sentence, Table, Takeaways, n } from "../../../kit";
import * as f from "../facts";
import { KINDS } from "./Overview";
export function Stack({ layer, onLayer }: { layer: number; onLayer: (v: number) => void }) {
  return (
    <Section
      id="stack"
      title="The stack"
      source={[{ source: "reference", symbol: "Qwen3MoeDecoderLayer" }]}
      lede="Select any layer to inspect its own parameter namespace. This checkpoint has one layer kind throughout; there is no dense warm-up block or alternate attention pattern."
    >
      <LayerMap
        label="Checkpoint decoder layers"
        count={f.layers.value}
        selected={layer}
        onSelect={onLayer}
        kinds={KINDS}
        kindOf={f.kindOf}
      />
      <Sentence>
        Layer {n(layer)} is a {f.kindOf(layer)} layer: <code>model.layers.{n(layer)}</code>. Its block below reads and
        writes a {n(f.hidden.value)}-channel stream.
      </Sentence>
      <Table
        caption="The branch this checkpoint takes"
        head={["Condition", "Pinned value", "Consequence"]}
        rows={[
          ["mlp_only_layers", "[]", "No layer forced dense"],
          ["decoder_sparse_step", n(f.sparseStep.value), "Every layer eligible for routed experts"],
          ["use_sliding_window", String(f.sliding.value), "Full causal attention throughout"],
          [
            "Routed parameters per layer",
            n(f.expertsPerLayer.value),
            `${n(f.topK.value)} of ${n(f.experts.value)} experts execute for a token`,
          ],
        ]}
      />
      <Takeaways
        items={[
          "Layers share an architecture, not their weights. Selecting another tile changes which parameter namespace you inspect.",
          "The configured dense intermediate width belongs to an unused branch and must not enter the expert ledger.",
        ]}
      />
    </Section>
  );
}
