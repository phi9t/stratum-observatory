import {
  Figure,
  StackOverview,
  Section,
  Table,
  Tensor,
  Takeaways,
  Prose,
  n,
  shape,
  type LayerKind,
} from "../../../kit";
import * as f from "../facts";
export const KINDS: readonly LayerKind[] = [{ id: "moe", label: "GQA + routed experts", lane: "low" }];
export function Overview({ layer }: { layer: number }) {
  const stream = shape("B", "S", f.hidden.value);
  return (
    <Section
      id="overview"
      title="The whole circuit"
      source={[
        { source: "reference", symbol: "Qwen3MoeModel.forward" },
        { source: "reference", symbol: "Qwen3MoeForCausalLM.forward" },
      ]}
      lede="Token ids enter a constant-width residual stream. Every layer reads earlier positions through attention, then sends each token to its selected experts. A separate output matrix scores the vocabulary."
    >
      <Figure
        label="Whole Qwen3 MoE decoder"
        caption="The highlighted tile follows your layer selection. Position rotations and the causal mask enter attention at every layer; no position vector is added to the embedding."
      >
        <StackOverview
          title="Qwen3-235B-A22B forward circuit"
          before={[
            { title: "token ids", line: shape("B", "S"), kind: "tensor", lane: "merged" },
            { title: "embedding", line: `${n(f.vocab.value)} × ${n(f.hidden.value)}`, lane: "merged" },
          ]}
          after={[
            { title: "RMSNorm", line: n(f.hidden.value), lane: "merged" },
            { title: "untied head", line: `${n(f.hidden.value)} → ${n(f.vocab.value)}`, lane: "merged" },
            { title: "logits", line: shape("B", "S", f.vocab.value), lane: "merged", kind: "tensor" },
          ]}
          count={f.layers.value}
          kinds={KINDS}
          kindOf={f.kindOf}
          selected={layer}
          stackTitle={`${n(f.layers.value)} pre-norm MoE layers`}
          stackLine={`${stream} in and out`}
          side={{ title: "positions + mask", line: "enter every attention", lane: "instance" }}
        />
      </Figure>
      <Prose>
        Ink is the residual stream; blue is attention and queries; orange is keys, values and expert computation; violet
        is positions, caches and routing; dashed teal is a residual bypass. Yellow marks selection. Bars beneath
        operation boxes compare output widths within that figure.
      </Prose>
      <Table
        caption="Stages of the forward pass"
        head={["Stage", "What happens", "Output"]}
        rows={[
          [
            "Embedding",
            "Select a learned vocabulary row for each id.",
            <Tensor key="e" name="hidden" shape={stream} />,
          ],
          [
            "Positions",
            "Compute cosine and sine once; every layer rotates its own queries and keys.",
            shape("B", "S", f.headDim.value),
          ],
          [
            "Decoder stack",
            `${n(f.layers.value)} separate parameter sets; attention then MoE, with an add after each.`,
            stream,
          ],
          ["Final norm", "Normalize the last residual stream before scoring tokens.", stream],
          [
            "Output head",
            "Project using its own biasless matrix; embedding weights are not reused.",
            shape("B", "S", f.vocab.value),
          ],
        ]}
      />
      <Takeaways
        items={[
          "Sparse expert execution does not make the residual stream sparse: every position still has a full vector.",
          "The head and the input embedding are distinct learned matrices.",
        ]}
      />
    </Section>
  );
}
