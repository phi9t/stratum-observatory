import { Schematic, SourceRef, n, useUrlNumber } from "../../kit";
import * as f from "./facts";
import {
  Overview,
  Data,
  Layers,
  Attention,
  Indexer,
  Experts,
  Run,
  Weights,
  Serving,
  Sources,
} from "./sections/Circuit";
import sources from "./sources.json";
const sections = [
  { id: "overview", label: "Circuit" },
  { id: "data", label: "Data" },
  { id: "layers", label: "Layers" },
  { id: "attention", label: "Attention" },
  { id: "indexer", label: "Sparse search" },
  { id: "experts", label: "Experts" },
  { id: "run", label: "Run it" },
  { id: "weights", label: "Weights" },
  { id: "serving", label: "Serving" },
  { id: "sources", label: "Sources" },
];
export const sectionIds = sections.map((s) => s.id);
export function Page() {
  const [layer, onLayer] = useUrlNumber("layer", 0, 0, f.layers.value - 1);
  return (
    <Schematic
      name="GLM-5.3"
      eyebrow="Model schematic · zai-org/GLM-5.3"
      tagline="Compressed attention, reused sparse selections, and routed experts — from the whole decoder to executable arithmetic."
      sections={sections}
      authority={
        <>
          Checkpoint {sources.checkpoint.revision} · drawn from{" "}
          <SourceRef source="reference" symbol="GlmMoeDsaForCausalLM" />
        </>
      }
      spec={[
        { value: n(f.layers.value), label: "layers" },
        { value: n(f.hidden.value), label: "stream width" },
        { value: n(f.heads.value), label: "attention heads" },
        { value: `${n(f.topK.value)} / ${n(f.experts.value)}`, label: "routed experts" },
        { value: n(f.indexTopK.value), label: "selected keys" },
        { value: n(f.maxContext.value), label: "position limit" },
      ]}
    >
      <Overview layer={layer} />
      <Data />
      <Layers layer={layer} onLayer={onLayer} />
      <Attention />
      <Indexer layer={layer} />
      <Experts layer={layer} />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
