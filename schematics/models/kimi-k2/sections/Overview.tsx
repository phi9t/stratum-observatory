import { Figure, StackOverview, Section, Table, Takeaways, shape, n } from "../../../kit";
import * as f from "../facts";
import { KINDS, useLayer } from "./Stack";
export function Overview() {
  const [layer] = useLayer();
  return (
    <Section
      id="overview"
      title="The whole circuit"
      source={[
        { source: "reference", symbol: "DeepseekV3Model" },
        { source: "reference", symbol: "DeepseekV3ForCausalLM" },
      ]}
      lede="Token ids enter a decoder. Attention reads earlier positions; a dense or expert branch rewrites each position. The final head scores the next token without sharing the embedding weights."
    >
      <Figure
        label="Kimi K2 whole circuit"
        caption="Ink is the residual stream; blue is attention, orange is feed-forward, violet is position or routing state, and teal is a residual bypass. Yellow marks selection."
      >
        <StackOverview
          title="Kimi K2 decoder from ids to logits"
          before={[
            { title: "ids", line: shape("B", "S"), lane: "merged", kind: "tensor" },
            { title: "embedding", line: `${n(f.vocab.value)} × ${n(f.hidden.value)}`, lane: "merged" },
          ]}
          after={[
            { title: "RMSNorm", line: n(f.hidden.value), lane: "merged" },
            { title: "head", line: `${n(f.hidden.value)} → ${n(f.vocab.value)}`, lane: "merged" },
            { title: "logits", line: shape("B", "S", f.vocab.value), lane: "merged", kind: "tensor" },
          ]}
          count={f.layers.value}
          kinds={KINDS}
          kindOf={f.kindOf}
          selected={layer}
          stackTitle={`${n(f.layers.value)} pre-norm layers`}
          stackLine={shape("B", "S", f.hidden.value)}
          side={{ title: "positions + causal mask", line: "enter every attention", lane: "instance" }}
        />
      </Figure>
      <Table
        caption="Stages of the forward pass"
        head={["Stage", "What happens", "Output"]}
        rows={[
          [
            "Embedding",
            "Look up a vector per token id; no position embedding is added.",
            shape("B", "S", f.hidden.value),
          ],
          [
            "Decoder stack",
            `${n(f.denseLayers.value)} dense layer, then ${n(f.moeLayers.value)} expert layers; attention throughout.`,
            shape("B", "S", f.hidden.value),
          ],
          ["Final norm", "Normalize the stream before the vocabulary projection.", shape("B", "S", f.hidden.value)],
          ["Head", "A separate biasless matrix produces float32 logits.", shape("B", "S", f.vocab.value)],
        ]}
      />
      <Takeaways
        items={[
          "The stream preserves its width across every residual addition.",
          "Rotary positions affect attention comparisons, not the initial embedding.",
        ]}
      />
    </Section>
  );
}
