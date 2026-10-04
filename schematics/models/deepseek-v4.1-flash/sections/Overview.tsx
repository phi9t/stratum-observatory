import { Section, Figure, StackOverview, Table, Takeaways, shape, n, Prose } from "../../../kit";
import * as f from "../facts";
import { KINDS, useLayer } from "./Stack";
export function Overview() {
  const [layer] = useLayer();
  return (
    <Section
      id="overview"
      title="The whole circuit"
      source={[{ source: "reference", symbol: "Transformer.forward" }]}
      lede="Token vectors expand into parallel residual copies. Attention and experts read a collapsed stream; learned mixing returns their outputs to the copies."
    >
      <Figure
        label="Architecture figure"
        caption="Attention is blue; experts orange; state violet; residual mixing teal. Layer tiles describe state ownership, not a different FFN architecture."
      >
        <StackOverview
          title="DeepSeek V4.1 Flash text backbone"
          before={[
            { title: "token ids", line: shape("B", "S"), lane: "merged", kind: "tensor" },
            { title: "embed + expand", line: shape("B", "S", f.copies.value, f.hidden.value), lane: "merged" },
          ]}
          after={[
            { title: "collapse + norm", line: shape("B", "S", f.hidden.value), lane: "merged" },
            { title: "untied head", line: shape("B", "S", f.vocab.value), lane: "merged" },
          ]}
          count={f.layers.value}
          kinds={KINDS}
          kindOf={f.kindOf}
          stackTitle={`${n(f.layers.value)} hyper-connected MoE layers`}
          stackLine={shape("B", "S", f.copies.value, f.hidden.value)}
          selected={layer}
          side={{ title: "positions + owned state", line: "window / compressed KV / indices", lane: "instance" }}
        />
      </Figure>
      <Prose>On narrow screens, scroll diagrams and wide tables horizontally to inspect the whole circuit.</Prose>
      <Table
        caption="Stages and dependencies"
        head={["Stage", "What happens", "Output"]}
        rows={[
          [
            "Input",
            "Text ids; optional image features replace their designated spans.",
            shape("B", "S", f.hidden.value),
          ],
          [
            "Expand",
            "Repeat the embedding; initial collapse selects the first copy.",
            shape("B", "S", f.copies.value, f.hidden.value),
          ],
          [
            "Backbone",
            "Engram precedes its selected blocks; each block performs attention then MoE.",
            shape("B", "S", f.copies.value, f.hidden.value),
          ],
          [
            "Collapse",
            "Final FFN hands its collapse coefficients to the output stage.",
            shape("B", "S", f.hidden.value),
          ],
          [
            "Norm + head",
            "RMSNorm and untied vocabulary projection. Publisher generation takes the last position.",
            shape("B", f.vocab.value),
          ],
        ]}
      />
      <Prose>
        Vision input and dSpark draft computation are shown in their own section. They are separate paths; neither
        changes the backbone checkpoint identity.
      </Prose>
      <Takeaways
        items={[
          `The stream carries ${n(f.copies.value)} copies, each ${n(f.hidden.value)} channels wide.`,
          `All ${n(f.layers.value)} backbone layers include routed and shared experts.`,
          "The output collapse uses handed-off coefficients, not a learned standalone HC head.",
        ]}
      />
    </Section>
  );
}
