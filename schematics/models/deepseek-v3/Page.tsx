import { Schematic, SourceRef, n, approx, useSources } from "../../kit";
import * as f from "./facts";
import { Overview } from "./sections/Overview";
import { Data } from "./sections/Data";
import { Stack } from "./sections/Stack";
import { Block } from "./sections/Block";
import { Attention } from "./sections/Attention";
import { Positions } from "./sections/Positions";
import { Experts } from "./sections/Experts";
import { Weights } from "./sections/Weights";
import { Serving } from "./sections/Serving";
import { Sources } from "./sections/Sources";
import { Run } from "./sections/Run";
export const sections = [
  { id: "overview", label: "Overview" },
  { id: "data", label: "Data" },
  { id: "stack", label: "Stack" },
  { id: "block", label: "Block" },
  { id: "attention", label: "Attention" },
  { id: "positions", label: "Positions" },
  { id: "experts", label: "Experts" },
  { id: "run", label: "Run it" },
  { id: "weights", label: "Weights" },
  { id: "serving", label: "Serving" },
  { id: "sources", label: "Sources" },
] as const;
export const sectionIds: string[] = sections.map((s) => s.id);
export function Page() {
  const { file } = useSources();
  return (
    <Schematic
      name="DeepSeek-V3"
      eyebrow={`Model schematic · ${file.checkpoint.repo} @ ${file.checkpoint.revision}`}
      tagline="A compressed attention path and selectively activated experts, from the whole circuit down to each operation in the main forward pass."
      sections={sections}
      authority={
        <>
          Drawn from <SourceRef source="reference" symbol="DeepseekV3ForCausalLM" />
        </>
      }
      spec={[
        { value: n(f.layers.value), label: "main layers" },
        { value: n(f.hidden.value), label: "residual width" },
        { value: n(f.heads.value), label: "attention heads" },
        { value: `${n(f.topK.value)} / ${n(f.experts.value)}`, label: "routed experts per token" },
        { value: n(f.kvAWidth.value), label: "compressed cache scalars" },
        { value: approx(f.parameterTotal.value), label: "main parameters" },
      ]}
    >
      <Overview />
      <Data />
      <Stack />
      <Block />
      <Attention />
      <Positions />
      <Experts />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
