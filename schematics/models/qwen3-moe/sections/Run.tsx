import {
  Section,
  Slider,
  Button,
  Segments,
  Figure,
  Heatmap,
  Sentence,
  Readout,
  Sub,
  Table,
  Check,
  Takeaways,
  Prose,
  n,
  shape,
  useUrlChoice,
  useUrlNumber,
  type Routing,
} from "../../../kit";
import * as f from "../facts";
import { toyRun } from "../experiments";
const LAYERS = Array.from({ length: f.TOY.layers }, (_, i) => String(i));
const HEADS = Array.from({ length: f.TOY.heads }, (_, i) => String(i));
const LAYOUT = { left: 32, pitch: 104, barWidth: 38, base: 164, barHeight: 112, width: 480, height: 224 };
function ToyRouter({ routing }: { routing: Routing }) {
  const peak = Math.max(...routing.scores);
  return (
    <svg
      className="ms-fig"
      role="img"
      aria-label="Computed toy expert scores and selected weights"
      width={LAYOUT.width}
      height={LAYOUT.height}
      viewBox={`0 0 ${LAYOUT.width} ${LAYOUT.height}`}
    >
      <title>Computed toy expert scores and selected mixture</title>
      <text className="ms-fig__column" x={LAYOUT.left} y={20}>
        Router scores · {n(f.TOY.experts)} toy experts
      </text>
      {routing.scores.map((score, e) => {
        const chosen = routing.chosen.indexOf(e),
          x = LAYOUT.left + e * LAYOUT.pitch;
        const height = Math.max(1, (score / peak) * LAYOUT.barHeight);
        return (
          <g key={e}>
            <rect
              className="ms-router__bar"
              data-chosen={chosen >= 0 ? "true" : undefined}
              x={x}
              y={LAYOUT.base - height}
              width={LAYOUT.barWidth}
              height={height}
            />
            <text className="ms-fig__note" x={x} y={LAYOUT.base + 20}>
              expert {n(e)}
            </text>
            <text className="ms-node__line" x={x} y={LAYOUT.base + 38}>
              {chosen >= 0 ? `w=${routing.weights[chosen].toFixed(f.DECIMALS)}` : "not chosen"}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function Run() {
  const [seed, setSeed] = useUrlNumber("runSeed", f.DEFAULT_SEED, 0, f.MAX_SEED);
  const [layer, setLayer] = useUrlChoice("runLayer", "0", LAYERS);
  const [head, setHead] = useUrlChoice("runHead", "0", HEADS);
  const [token, setToken] = useUrlNumber("runToken", f.TOY.tokens - 1, 0, f.TOY.tokens - 1);
  const { ctx, ids, logits } = toyRun(seed),
    prefix = `h${layer}`;
  const data = (id: string) => ctx.trace.get(id)!.data!;
  const scores = data(`${prefix}.moe.scores`),
    selected = data(`${prefix}.moe.selected`),
    weights = data(`${prefix}.moe.weights`);
  const routing: Routing = {
    scores: Array.from(scores.slice(token * f.TOY.experts, (token + 1) * f.TOY.experts)),
    chosen: Array.from(selected.slice(token * f.TOY.topK, (token + 1) * f.TOY.topK)),
    weights: Array.from(weights.slice(token * f.TOY.topK, (token + 1) * f.TOY.topK)),
    chosenGroups: [0],
  };
  const seq = f.TOY.tokens,
    a = data(`${prefix}.attn.weights`),
    offset = Number(head) * seq * seq;
  const grid = Array.from({ length: seq }, (_, q) => Array.from({ length: seq }, (_, k) => a[offset + q * seq + k]));
  return (
    <Section
      id="run"
      title="Run it: a token's computed route"
      source={[
        { source: "reference", symbol: "Qwen3MoeSparseMoeBlock.forward" },
        { source: "reference", symbol: "Qwen3MoeAttention.forward" },
      ]}
      lede="The clean decoder runs here at toy widths. All weights are seeded random values; these figures reveal the mechanism, not the trained checkpoint's behavior."
    >
      <Segments
        legend="Toy layer"
        value={layer}
        onChange={setLayer}
        options={LAYERS.map((id) => ({ id, label: `layer ${id}` }))}
      />
      <Slider label="Token position" min={0} max={seq - 1} value={token} onChange={setToken} />
      <div className="ms-control">
        <Button onClick={() => setSeed((s) => (s + 1) % (f.MAX_SEED + 1))}>Draw new weights</Button>
      </div>
      <Sentence>
        Token id {n(ids[0][token])} at position {n(token)} in toy layer {layer} executes experts{" "}
        {routing.chosen.map(n).join(", ")}, with weights{" "}
        {routing.weights.map((w) => w.toFixed(f.DECIMALS)).join(" and ")}.
      </Sentence>
      <Figure
        label="Actual toy decoder router output"
        caption="These scores come from the post-attention normalized hidden stream of the full toy forward pass. Each toy expert has its own learned matrix set."
      >
        <ToyRouter routing={routing} />
      </Figure>
      <Table
        caption="Every toy token's expert choices"
        head={["Position", "Token id", "Selected experts", "Normalized weights"]}
        current={token}
        rows={ids[0].map((id, t) => [
          n(t),
          n(id),
          Array.from(selected.slice(t * f.TOY.topK, (t + 1) * f.TOY.topK))
            .map(n)
            .join(", "),
          Array.from(weights.slice(t * f.TOY.topK, (t + 1) * f.TOY.topK))
            .map((w) => w.toFixed(f.DECIMALS))
            .join(" · "),
        ])}
      />
      <Sub>Different queries can share the same keys</Sub>
      <Segments
        legend="Toy query head"
        value={head}
        onChange={setHead}
        options={HEADS.map((id) => ({ id, label: `head ${id}` }))}
      />
      <Sentence>
        Query head {head} shares KV head {n(Math.floor(Number(head) / (f.TOY.heads / f.TOY.kvHeads)))}. At position{" "}
        {n(token)}, it may read positions through {n(token)} only.
      </Sentence>
      <Figure
        label="Computed toy causal attention"
        caption="The upper triangle is exactly zero after causal masking. Switching query heads changes the computed weights even within a shared KV group."
      >
        <Heatmap title="Toy attention probabilities" values={grid} rowLabel="query" columnLabel="key" marked={token} />
      </Figure>
      <Readout
        label="Toy run dimensions"
        items={[
          {
            term: "Toy architecture",
            value: `${n(f.TOY.layers)} × ${n(f.TOY.hidden)}`,
            note: `${n(f.TOY.experts)} experts, top-${n(f.TOY.topK)}`,
          },
          { term: "Query / KV heads", value: `${n(f.TOY.heads)} / ${n(f.TOY.kvHeads)}` },
          { term: "Toy parameters", value: n(ctx.parameterCount()) },
          { term: "Logits", value: shape(...logits.shape) },
        ]}
      />
      <Prose>
        The same model.ts runs at real checkpoint widths without allocating weights to verify the parameter count. Its
        CPU fixture uses pinned Transformers with perturbed toy weights and compares logits, attention, expert ids and
        mixture weights. Browser arithmetic uses float32 tensors; it does not test GPU kernels or bfloat16 precision.
      </Prose>
      <Check question="Why can the same token choose different experts in different layers?">
        Each layer has different router weights and a different post-attention hidden vector. Neither its expert ids nor
        its routing weights are reused from earlier layers.
      </Check>
      <Takeaways
        items={[
          "Expert selection is computed from the current hidden representation, not directly from the integer token id.",
          "Changing the random seed changes a real forward pass; the result is an architecture experiment, not generated language.",
        ]}
      />
    </Section>
  );
}
