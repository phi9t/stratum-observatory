import {
  Button,
  Figure,
  Heatmap,
  Prose,
  Readout,
  ref,
  Section,
  Segments,
  Sentence,
  shape,
  Slider,
  Takeaways,
  n,
  useUrlChoice,
  useUrlNumber,
} from "../../../kit";
import * as f from "../facts";
import { forward, type QwenConfig } from "../reference/model";

const TOY: QwenConfig = {
  vocab: f.TOY.vocab,
  hidden: f.TOY.hidden,
  layers: f.TOY.layers,
  heads: f.TOY.heads,
  kvHeads: f.TOY.kvHeads,
  headDim: f.TOY.headDim,
  inner: f.TOY.inner,
  theta: f.theta.value,
  normEps: f.normEps.value,
};
const LAYERS = Array.from({ length: f.TOY.layers }, (_unused, index) => String(index));
const HEADS = Array.from({ length: f.TOY.heads }, (_unused, index) => String(index));

export function Run() {
  const [seed, setSeed] = useUrlNumber("runSeed", 1, 0, f.DISPLAY.seedMax);
  const [layer, setLayer] = useUrlChoice("runLayer", "0", LAYERS);
  const [head, setHead] = useUrlChoice("runHead", "0", HEADS);
  const [query, setQuery] = useUrlNumber("runQ", f.TOY.tokens - 1, 0, f.TOY.tokens - 1);

  // The clean reference, run here in the browser: seeded weights, seeded token ids.
  const ctx = new ref.Ctx("value", null, seed);
  const ids = [
    Array.from({ length: f.TOY.tokens }, (_unused, index) => (index * f.DISPLAY.idStride + seed) % f.TOY.vocab),
  ];
  const logits = forward(ctx, TOY, ids);
  const weights = ctx.trace.get(`h${layer}.attn.weights`);
  if (weights?.data == null) {
    throw new Error("reference run produced no attention weights");
  }
  const seq = f.TOY.tokens;
  const offset = Number(head) * seq * seq;
  const grid = Array.from({ length: seq }, (_r, row) =>
    Array.from({ length: seq }, (_c, column) => weights.data![offset + row * seq + column]),
  );
  const rowSum = grid[query].reduce((sum, value) => sum + value, 0);
  const strongest = grid[query].indexOf(Math.max(...grid[query]));

  return (
    <Section
      id="run"
      lede="The same forward pass, written cleanly and run in your browser at toy width. The weights are random, so this shows how the mechanism behaves, not what the trained model has learned."
      source={[
        { source: "reference", symbol: "Qwen3Attention.forward" },
        { source: "reference", symbol: "eager_attention_forward" },
      ]}
      title="Run it"
    >
      <Segments
        legend="Layer"
        onChange={setLayer}
        options={LAYERS.map((id) => ({ id, label: `layer ${id}` }))}
        value={layer}
      />
      <Segments
        legend="Head"
        onChange={setHead}
        options={HEADS.map((id) => ({ id, label: `head ${id}` }))}
        value={head}
      />
      <Slider label="Query position" max={f.TOY.tokens - 1} min={0} onChange={setQuery} value={query} />
      <div className="ms-control">
        <Button onClick={() => setSeed((current) => current + 1)}>Draw new weights</Button>
      </div>
      <Sentence>
        With weights from seed {n(seed)}, position {n(query)} in layer {layer}, head {head} puts its largest weight,{" "}
        {grid[query][strongest].toFixed(f.DISPLAY.digits)}, on position {n(strongest)}. Its weights sum to{" "}
        {rowSum.toFixed(f.DISPLAY.digits)}.
      </Sentence>
      <Figure
        caption="Computed, not drawn: these are the softmax outputs of the reference implementation. The upper triangle is exactly zero because the causal mask set those scores to minus infinity before the softmax."
        label="Computed attention weights"
      >
        <Heatmap
          columnLabel="key position"
          marked={query}
          rowLabel="query"
          title={`Attention weights of layer ${layer}, head ${head}, from a toy-width run`}
          values={grid}
        />
      </Figure>
      <Readout
        items={[
          {
            term: "Toy size",
            value: `${n(f.TOY.layers)} × ${n(f.TOY.hidden)}`,
            note: `layers × width, ${n(f.TOY.heads)} heads`,
          },
          { term: "Parameters", value: n(ctx.parameterCount()), note: "in this toy instance" },
          { term: "Weights tensor", value: shape(...weights.shape) },
          { term: "Logits", value: shape(...logits.shape) },
        ]}
        label="Toy run readout"
      />
      <Prose>
        The code that produced this is <code>models/qwen3-dense/reference/model.ts</code>: one file, the checkpoint’s
        own parameter names, no framework. A test loads weights exported from the upstream implementation at this toy
        width and requires a maximum absolute logits error below {n(f.DISPLAY.parityTolerance)}; the same code run
        without data at full width yields every shape on this page and the parameter total.
      </Prose>
      <Takeaways
        items={[
          "Each row of attention weights is a probability distribution over the positions that row may read.",
          "Position 0 can only read itself, so its weight on itself is always one, whatever the weights are.",
        ]}
      />
    </Section>
  );
}
