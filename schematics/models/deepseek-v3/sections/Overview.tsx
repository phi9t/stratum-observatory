import { Figure, Section, StackOverview, Table, Tensor, Takeaways, n, shape, Prose } from "../../../kit";
import * as f from "../facts";
import { KINDS, useLayer } from "./Stack";
export function Overview() {
  const [selected] = useLayer();
  const x = shape("B", "S", f.hidden.value);
  return (
    <Section
      id="overview"
      title="The whole circuit"
      source={[
        { source: "reference", symbol: "DeepseekV3Model.forward" },
        { source: "reference", symbol: "DeepseekV3ForCausalLM.forward" },
      ]}
      lede="Token ids become a residual stream. Each layer adds an attention correction and a feed-forward correction; a final norm and an independent output matrix turn the stream into next-token scores."
    >
      <Figure
        label="Main model circuit"
        caption="Tiles represent individual main layers, coloured by their feed-forward kind. Positions and a causal mask enter attention, rather than being added to the embeddings."
      >
        <StackOverview
          title="DeepSeek-V3 main inference circuit"
          before={[
            { title: "ids", line: shape("B", "S"), kind: "tensor", lane: "merged" },
            { title: "embedding", line: `${n(f.vocab.value)} × ${n(f.hidden.value)}`, lane: "high" },
          ]}
          after={[
            { title: "RMSNorm", line: n(f.hidden.value), lane: "merged" },
            { title: "head", line: `${n(f.hidden.value)} → ${n(f.vocab.value)}`, lane: "high" },
            { title: "logits", line: shape("B", "S", f.vocab.value), kind: "tensor", lane: "merged" },
          ]}
          count={f.layers.value}
          kinds={KINDS}
          kindOf={f.kindOf}
          selected={selected}
          stackTitle={`${n(f.layers.value)} pre-norm layers`}
          stackLine={`${x} in and out`}
          side={{ title: "positions + mask", line: "enter each attention", lane: "instance" }}
        />
      </Figure>
      <Prose>
        Blue identifies attention and vocabulary projections; orange identifies feed-forward/expert computation; violet
        identifies positions, routing and cache state. Teal dashed wires carry residual skips. Ink is the residual
        stream; yellow marks the selected operation or layer.
      </Prose>
      <Table
        caption="Stages of the main forward pass"
        head={["Stage", "What happens", "Output"]}
        rows={[
          ["Embedding", "Select a learned vector for each token id.", <Tensor name="x" shape={x} />],
          [
            "Decoder stack",
            `${n(f.layers.value)} layers rewrite the stream through residual additions.`,
            <Tensor name="hidden" shape={x} />,
          ],
          ["Final norm", "Normalize each final vector with a learned gain.", <Tensor name="normalized" shape={x} />],
          [
            "Head",
            "Apply the untied, bias-free output projection and cast logits to float32.",
            <Tensor name="logits" shape={shape("B", "S", f.vocab.value)} />,
          ],
        ]}
      />
      <Takeaways
        items={[
          "The head and embedding are separate matrices, even though they have matching shapes.",
          "The main forward pass does not call the MTP tensors stored alongside it.",
        ]}
      />
    </Section>
  );
}
