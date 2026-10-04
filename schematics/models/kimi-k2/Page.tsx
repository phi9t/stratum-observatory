import { Schematic, SourceRef, n, approx } from "../../kit";
import * as f from "./facts";
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
  { id: "stack", label: "Stack" },
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
  return (
    <Schematic
      name="Kimi K2"
      eyebrow="Model schematic · moonshotai/Kimi-K2-Instruct"
      tagline="A vast expert bank behind a narrow latent attention path. Follow the original Instruct checkpoint from token ids to logits, then run its arithmetic at toy width."
      sections={sections}
      authority={
        <>
          Drawn from the checkpoint's <SourceRef source="reference" symbol="DeepseekV3ForCausalLM" />
        </>
      }
      spec={[
        { value: n(f.layers.value), label: "layers" },
        { value: n(f.hidden.value), label: "stream width" },
        { value: `${n(f.topK.value)} / ${n(f.experts.value)}`, label: "routed experts per token" },
        { value: n(f.heads.value), label: "attention heads" },
        { value: n(f.context.value), label: "configured context" },
        { value: approx(f.parameterTotal.value), label: "learned parameters" },
      ]}
    >
      <Overview />
      <Data />
      <Stack />
      <Block />
      <Attention />
      <Experts />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
