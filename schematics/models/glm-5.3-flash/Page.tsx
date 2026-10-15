import { Schematic, SourceRef, approx, n, useUrlNumber, type SectionSpec } from "../../kit";
import * as f from "./facts";
import {
  Block,
  Data,
  Experts,
  Kda,
  Overview,
  Run,
  Serving,
  Sources,
  Sparse,
  Stack,
  Training,
  Weights,
} from "./sections/Circuit";
import sources from "./sources.json";

const sections: readonly SectionSpec[] = [
  { id: "overview", label: "Circuit" },
  { id: "data", label: "Data" },
  { id: "stack", label: "Layers" },
  { id: "block", label: "Block" },
  { id: "kda", label: "KDA" },
  { id: "sparse", label: "Sparse" },
  { id: "experts", label: "Experts" },
  { id: "training", label: "Training" },
  { id: "run", label: "Run" },
  { id: "weights", label: "Weights" },
  { id: "serving", label: "Serving" },
  { id: "sources", label: "Sources" },
];

/** Section ids, in page order. coverage.json maps reference classes onto these. */
export const sectionIds: string[] = sections.map((section) => section.id);

export function Page() {
  const [layer, setLayer] = useUrlNumber("layer", f.demo.layerDefault, 0, f.layers.value - 1);
  return (
    <Schematic
      authority={
        <>
          Checkpoint {sources.checkpoint.revision.slice(0, f.topK.value + f.sharedExperts.value)} · drawn from{" "}
          <SourceRef source="reference" symbol="Glm5NextTextModel" /> and vLLM serving · shape-only reference, numerical
          parity unverified
        </>
      }
      eyebrow="Model schematic · zai-org/GLM-5.3-Flash-BF16"
      name="GLM-5.3-Flash"
      sections={sections}
      spec={[
        { value: n(f.layers.value), label: "base layers" },
        { value: `${n(f.kdaCount.value)} / ${n(f.sparseAttentionCount.value)}`, label: "KDA / sparse" },
        { value: n(f.hcMult.value), label: "mHC streams" },
        { value: `${n(f.topK.value)} / ${n(f.experts.value)}`, label: "selected / routed experts" },
        { value: approx(f.parameterTotal.value), label: "stored tensors" },
        { value: n(f.maxContext.value), label: "config context" },
      ]}
      tagline="KDA three times, sparse MLA plus IndexPool once, four residual streams throughout, and expert-parallel training evidence for the released BF16 checkpoint."
    >
      <Overview layer={layer} />
      <Data />
      <Stack layer={layer} onLayer={setLayer} />
      <Block />
      <Kda />
      <Sparse />
      <Experts />
      <Training />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
