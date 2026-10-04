import {
  Button,
  Check,
  Sub,
  Table,
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
  const ctx = new ref.Ctx("value", null, seed, f.DISPLAY.runGain);
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
  const groups = TOY.heads / TOY.kvHeads;
  const kvHead = Math.floor(Number(head) / groups);
  const groupHeads = Array.from({ length: groups }, (_, i) => kvHead * groups + i);
  const trace = (name: string) => ctx.trace.get(`h${layer}.attn.${name}`)!;
  const vector = (name: string, h: number, pos: number) =>
    Array.from(trace(name).data!.slice((h * seq + pos) * TOY.headDim, (h * seq + pos + 1) * TOY.headDim));
  const format = (value: number) => value.toFixed(f.DISPLAY.digits);
  const groupWeights = groupHeads.map((h) =>
    Array.from({ length: seq }, (_, k) => weights.data![(h * seq + query) * seq + k]),
  );
  const scoreAt = (name: string, k: number) => trace(name).data![(Number(head) * seq + query) * seq + k];
  const [key, setKey] = useUrlNumber("runKey", 1, 0, seq - 1);
  const pair = TOY.headDim / 2;

  return (
    <Section
      id="run"
      lede="Different queries can read identical keys and values. Run Qwen3’s projections, head normalization and rotary arithmetic with random toy weights to see why their attention still differs."
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
      <Sub>Shared keys, different queries</Sub>
      <Sentence>
        In toy layer {layer}, query heads {groupHeads.map(n).join(" and ")} read KV head {n(kvHead)} at position{" "}
        {n(query)}. Seed {n(seed)} uses weight gain {n(f.DISPLAY.runGain)}. The checkpoint shares each KV head across{" "}
        {n(f.groups.value)} queries; this smaller toy shares it across {n(groups)}.
      </Sentence>
      <Figure
        label="Queries sharing a KV head"
        caption={`Rows are query heads ${groupHeads.map(n).join(" and ")}, in that order. Every column is the same key position and the same value vector for both queries; shading is the computed attention probability. Future keys remain masked.`}
      >
        <Heatmap
          columnLabel="shared key position"
          rowLabel="group row"
          title="Different query heads attending to the same KV head"
          values={groupWeights}
          marked={groupHeads.indexOf(Number(head))}
        />
      </Figure>
      <Table
        caption="Identical KV vectors, distinct query vectors at the selected position"
        head={["Query head", "KV head", "Rotated query", "Shared rotated key", "Shared value"]}
        rows={groupHeads.map((h) => [
          n(h),
          n(kvHead),
          vector("q", h, query).map(format).join(" · "),
          vector("k_repeated", h, query).map(format).join(" · "),
          vector("v_repeated", h, query).map(format).join(" · "),
        ])}
      />
      <Sub>What head normalization changes</Sub>
      <Prose>
        Follow the selected query head’s scaled dot products against every key. The first comparison omits head
        normalization and rotation; the next applies the learned Q/K RMSNorm gains; the last rotates those normalized
        vectors. These diagnostic comparisons reuse the same projections, not separate model runs. Only the last scores
        enter the actual forward pass.
      </Prose>
      <Sentence>
        Head {head} at query position {n(query)} scores key {n(key)} as {format(scoreAt("scores_raw", key))} before head
        normalization, {format(scoreAt("scores_norm", key))} after normalization, and {format(scoreAt("scores", key))}{" "}
        after rotation.{" "}
        {key > query
          ? "This key is in the future and is masked before softmax."
          : "This key is allowed by the causal mask."}
      </Sentence>
      <Table
        caption="Scaled QK scores before the causal mask"
        head={["Key position", "Projected only", "Head-normalized", "Normalized + RoPE", "Final probability"]}
        current={key}
        rows={Array.from({ length: seq }, (_, k) => [
          n(k),
          format(scoreAt("scores_raw", k)),
          format(scoreAt("scores_norm", k)),
          format(scoreAt("scores", k)),
          format(grid[query][k]),
        ])}
      />
      <Sub>Watch a half-split pair rotate</Sub>
      <Slider label="Key position" min={0} max={seq - 1} value={key} onChange={setKey} />
      <Sentence>
        The first rotary pair joins coordinates {n(0)} and {n(pair)}. Query position {n(query)} rotates by {n(query)}{" "}
        radians; key position {n(key)} rotates by {n(key)} radians. Rotation preserves each pair’s length while their
        relative angle changes the dot product.
      </Sentence>
      <Table
        caption="Actual normalized vectors before and after RoPE"
        head={["Vector", "Before pair", "After pair", "Pair length before → after"]}
        rows={[
          ["Query", "q_norm", "q", Number(head), query],
          ["Shared key", "k_norm", "k", kvHead, key],
        ].map(([label, before, after, h, pos]) => {
          const a = vector(String(before), Number(h), Number(pos));
          const b = vector(String(after), Number(h), Number(pos));
          return [
            String(label),
            `${format(a[0])} · ${format(a[pair])}`,
            `${format(b[0])} · ${format(b[pair])}`,
            `${format(Math.hypot(a[0], a[pair]))} → ${format(Math.hypot(b[0], b[pair]))}`,
          ];
        })}
      />
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
      <Check question="If query heads read identical keys and values, must their attention outputs match?">
        No. Their query projections differ, so their dot products and softmax probabilities can differ. Each mixes the
        same values with its own probabilities.
      </Check>
      <Takeaways
        items={[
          "Sharing a KV head saves stored keys and values without forcing its query heads to assign the same probabilities.",
          "Qwen3 normalizes each query and key head before rotation; learned coordinate gains still affect their dot products.",
          "RoPE preserves pair lengths but changes cross-position scores through relative rotation; values are never rotated.",
        ]}
      />
    </Section>
  );
}
