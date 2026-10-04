import { Schematic, SourceRef, n } from "../../kit";
import * as f from "./facts";
import sources from "./sources.json";
import { Overview } from "./sections/Overview";
import { Data } from "./sections/Data";
import { Stack } from "./sections/Stack";
import { Block } from "./sections/Block";
import { Attention } from "./sections/Attention";
import { Compression } from "./sections/Compression";
import { Experts } from "./sections/Experts";
import { Engram } from "./sections/Engram";
import { Auxiliary } from "./sections/Auxiliary";
import { Run } from "./sections/Run";
import { Weights } from "./sections/Weights";
import { Serving } from "./sections/Serving";
import { Sources } from "./sections/Sources";
export const sections = [
  { id: "overview", label: "Circuit" },
  { id: "data", label: "Tensors" },
  { id: "stack", label: "Layers" },
  { id: "block", label: "Block" },
  { id: "attention", label: "Attention" },
  { id: "compression", label: "Compression" },
  { id: "experts", label: "Experts" },
  { id: "engram", label: "Engram" },
  { id: "auxiliary", label: "Image + draft" },
  { id: "run", label: "Run it" },
  { id: "weights", label: "Weights" },
  { id: "serving", label: "Serving" },
  { id: "sources", label: "Sources" },
] as const;
export const sectionIds: string[] = sections.map((s) => s.id);
export function Page() {
  return (
    <Schematic
      name="DeepSeek V4.1 Flash"
      eyebrow="Model schematic · deepseek-ai/DeepSeek-V4.1-Flash"
      tagline="A shared attention memory across depth, a mixed residual across copies, and sparse experts at every layer."
      sections={sections}
      authority={
        <>
          Publisher <SourceRef source="reference" symbol="Transformer" /> · checkpoint{" "}
          {sources.checkpoint.revision.slice(0, 12)} · numerical parity unverified · logical parameter inventory
          reconciled
        </>
      }
      spec={[
        { value: n(f.layers.value), label: "backbone layers" },
        { value: n(f.hidden.value), label: "language width" },
        { value: n(f.copies.value), label: "residual copies" },
        { value: `${n(f.topK.value)} / ${n(f.experts.value)}`, label: "routed experts" },
        { value: n(f.context.value), label: "configured context" },
        { value: n(f.headDim.value), label: "shared KV width" },
      ]}
    >
      <Overview />
      <Data />
      <Stack />
      <Block />
      <Attention />
      <Compression />
      <Experts />
      <Engram />
      <Auxiliary />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
