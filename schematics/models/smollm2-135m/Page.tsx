import { Schematic, SourceRef, n, approx } from "../../kit";
import * as f from "./facts";
import { Attention } from "./sections/Attention";
import { Block } from "./sections/Block";
import { Data } from "./sections/Data";
import { Mlp } from "./sections/Mlp";
import { Overview } from "./sections/Overview";
import { Run } from "./sections/Run";
import { Serving } from "./sections/Serving";
import { Sources } from "./sections/Sources";
import { Stack } from "./sections/Stack";
import { Weights } from "./sections/Weights";

export const sections = [
  { id: "overview", label: "Overview" },
  { id: "data", label: "Data" },
  { id: "stack", label: "Stack" },
  { id: "block", label: "Block" },
  { id: "attention", label: "Attention" },
  { id: "mlp", label: "MLP" },
  { id: "run", label: "Run it" },
  { id: "weights", label: "Weights" },
  { id: "serving", label: "Serving" },
  { id: "sources", label: "Sources" },
] as const;

/** Section ids, in page order. coverage.json maps reference classes onto these. */
export const sectionIds: string[] = sections.map((section) => section.id);

export function Page() {
  return (
    <Schematic
      authority={
        <>
          Drawn from <SourceRef source="reference" symbol="LlamaForCausalLM" />
        </>
      }
      eyebrow="Model schematic · HuggingFaceTB/SmolLM2-135M"
      name="SmolLM2-135M"
      sections={sections}
      spec={[
        { value: n(f.layers.value), label: "blocks" },
        { value: n(f.hidden.value), label: "residual width" },
        { value: `${n(f.heads.value)} × ${n(f.headDim.value)}`, label: "attention heads" },
        { value: n(f.positions.value), label: "positions" },
        { value: n(f.vocab.value), label: "tokens" },
        { value: approx(f.parameterTotal.value), label: "parameters" },
      ]}
      tagline="A compact decoder with shared keys, rotary positions and a tied vocabulary matrix."
    >
      <Overview />
      <Data />
      <Stack />
      <Block />
      <Attention />
      <Mlp />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
