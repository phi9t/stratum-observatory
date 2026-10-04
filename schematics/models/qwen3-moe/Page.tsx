import { Schematic, SourceRef, n, useUrlNumber } from "../../kit";
import * as f from "./facts";
import sources from "./sources.json";
import { Overview } from "./sections/Overview";
import { Data } from "./sections/Data";
import { Stack } from "./sections/Stack";
import { Block } from "./sections/Block";
import { Attention } from "./sections/Attention";
import { Experts } from "./sections/Experts";
import { Run } from "./sections/Run";
import { Weights } from "./sections/Weights";
import { Serving } from "./sections/Serving";
import { Sources } from "./sections/Sources";
export const sections = [
  { id: "overview", label: "Overview" },
  { id: "data", label: "Data" },
  { id: "stack", label: "Layers" },
  { id: "block", label: "Block" },
  { id: "attention", label: "Attention" },
  { id: "experts", label: "Experts" },
  { id: "run", label: "Run it" },
  { id: "weights", label: "Weights" },
  { id: "serving", label: "Serving" },
  { id: "sources", label: "Sources" },
] as const;
export const sectionIds: string[] = sections.map((s) => s.id);
export function Page() {
  const [layer, setLayer] = useUrlNumber("layer", 0, 0, f.layers.value - 1);
  return (
    <Schematic
      name="Qwen3-235B-A22B"
      eyebrow={`Model schematic · ${sources.checkpoint.repo}`}
      tagline="Wide queries, compact memory, and a sparse expert bank: the exact decoder from token ids to vocabulary logits."
      sections={sections}
      authority={
        <>
          Drawn from <SourceRef source="reference" symbol="Qwen3MoeForCausalLM" /> · Checkpoint{" "}
          <code>{sources.checkpoint.revision}</code>
        </>
      }
      spec={[
        { value: n(f.layers.value), label: "layers" },
        { value: n(f.hidden.value), label: "stream width" },
        { value: `${n(f.heads.value)} / ${n(f.kvHeads.value)}`, label: "query / KV heads" },
        { value: `${n(f.topK.value)} / ${n(f.experts.value)}`, label: "experts per token" },
        { value: n(f.positions.value), label: "configured positions" },
        { value: n(f.parameterTotal.value), label: "stored parameters" },
      ]}
    >
      <Overview layer={layer} />
      <Data />
      <Stack layer={layer} onLayer={setLayer} />
      <Block layer={layer} />
      <Attention layer={layer} />
      <Experts layer={layer} />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
