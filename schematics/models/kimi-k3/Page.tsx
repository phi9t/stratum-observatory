import { Schematic, SourceRef, n, approx } from "../../kit";
import * as f from "./facts";
import { Overview } from "./sections/Overview";
import { Data } from "./sections/Data";
import { Stack } from "./sections/Stack";
import { Residual } from "./sections/Residual";
import { Attention } from "./sections/Attention";
import { Kda } from "./sections/Kda";
import { Experts } from "./sections/Experts";
import { Run } from "./sections/Run";
import { Serving, Weights } from "./sections/Serving";
import { Sources } from "./sections/Sources";
export const sections = [
  { id: "overview", label: "Circuit" },
  { id: "data", label: "Data" },
  { id: "stack", label: "Layers + block" },
  { id: "residual", label: "Depth residuals" },
  { id: "attention", label: "Full attention" },
  { id: "kda", label: "KDA" },
  { id: "experts", label: "Experts" },
  { id: "run", label: "Run it" },
  { id: "weights", label: "Weights" },
  { id: "serving", label: "Serving" },
  { id: "sources", label: "Sources" },
] as const;
export const sectionIds = sections.map((s) => s.id);
export function Page() {
  return (
    <Schematic
      name="Kimi-K3"
      eyebrow="Model schematic · moonshotai/Kimi-K3"
      tagline="A hybrid token memory, compressed routed experts, and attention across depth — traced through the publisher's exact checkpoint."
      sections={sections}
      authority={
        <>
          Publisher reference · <SourceRef source="reference" symbol="KimiLinearModel" /> · numerical whole-model parity
          unverified
        </>
      }
      spec={[
        { value: n(f.layers.value), label: "decoder layers" },
        { value: n(f.hidden.value), label: "stream width" },
        { value: `${n(f.kdaCount.value)} / ${n(f.fullCount.value)}`, label: "KDA / full" },
        { value: `${n(f.topK.value)} / ${n(f.experts.value)}`, label: "selected / routed experts" },
        { value: approx(f.parameterTotal.value), label: "logical text parameters" },
        { value: n(f.maxContext.value), label: "config context limit" },
      ]}
    >
      <Overview />
      <Data />
      <Stack />
      <Residual />
      <Attention />
      <Kda />
      <Experts />
      <Run />
      <Weights />
      <Serving />
      <Sources />
    </Schematic>
  );
}
