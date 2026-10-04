import {
  Section,
  OpGraph,
  Figure,
  MoeRouter,
  Button,
  Sentence,
  Readout,
  Prose,
  Check,
  Takeaways,
  shape,
  n,
  useUrlNumber,
  ref,
  type Edge,
} from "../../../kit";
import * as f from "../facts";
import { op, chain, stream } from "./graph";
import { MoEGate } from "../reference/model";
import { useLayer } from "./Stack";
export function Experts() {
  const [layer] = useLayer();
  const inner = f.kindOf(layer) === "dense" ? f.dense.value : f.expert.value;
  const [seed, setSeed] = useUrlNumber("router-seed", 1, 0, f.demo.seedMax);
  const ctx = new ref.Ctx("value", null, seed, f.demo.gain);
  const x = ref.input(ctx, [1, 1, f.hidden.value], () =>
    Array.from({ length: f.hidden.value }, (_, i) => Math.sin(i + seed)),
  );
  const result = MoEGate(ctx, "router", "demo", x, f.real);
  const ordered = Array.from(result.indices.data!)
    .map((expert, slot) => ({ expert, weight: result.weights.data![slot] }))
    .sort((a, b) => a.expert - b.expert);
  const routing = {
    chosen: ordered.map((item) => item.expert),
    weights: ordered.map((item) => item.weight),
    scores: Array.from(ctx.trace.get("demo.scores")!.data!),
    chosenGroups: [0],
  };
  const topology = [
    op(
      "x",
      "normalized token",
      f.hidden.value,
      { col: 0, row: 0 },
      "DeepseekV3MoE",
      "The same normalized token enters routed and shared paths.",
      "x",
      stream,
      stream,
      [],
      "merged",
    ),
    op(
      "route",
      "router",
      f.topK.value,
      { col: 1, row: 0 },
      "MoEGate",
      "Choose logical routed experts and their weights.",
      "topk(sigmoid(x Wᵀ) + bias)",
      stream,
      shape("B×S", f.topK.value),
      [],
      "instance",
    ),
    op(
      "expert",
      "selected experts",
      f.hidden.value,
      { col: 2, row: 0 },
      "DeepseekV3MoE",
      "Dispatch the token to each selected expert, then combine their outputs.",
      "Σ weight × expert(x)",
      stream,
      stream,
      [],
      "low",
    ),
    op(
      "shared",
      "shared expert",
      f.hidden.value,
      { col: 2, row: 1 },
      "DeepseekV3MoE",
      "Run the shared transform on every token without a router logit.",
      "shared(x)",
      stream,
      stream,
      [
        ["gate weight", shape(f.expert.value * f.shared.value, f.hidden.value)],
        ["up weight", shape(f.expert.value * f.shared.value, f.hidden.value)],
        ["down weight", shape(f.hidden.value, f.expert.value * f.shared.value)],
      ],
      "low",
    ),
    op(
      "sum",
      "add",
      f.hidden.value,
      { col: 3, row: 0 },
      "DeepseekV3MoE",
      "Add the always-on shared contribution to the routed sum.",
      "routed + shared",
      stream,
      stream,
      [],
      "low",
    ),
  ];
  const router = [
    op(
      "logits",
      "router linear",
      f.experts.value,
      { col: 0, row: 0 },
      "MoEGate",
      "Flatten token axes and compute float32 selection logits.",
      "flatten(x) W_gateᵀ",
      stream,
      shape("B×S", f.experts.value),
      [["gate.weight", shape(f.experts.value, f.hidden.value)]],
      "instance",
    ),
    op(
      "scores",
      "sigmoid",
      f.experts.value,
      { col: 1, row: 0 },
      "MoEGate",
      "Bound each expert score independently; scores need not sum to unity.",
      "score = sigmoid(logit)",
      shape("B×S", f.experts.value),
      shape("B×S", f.experts.value),
      [],
      "instance",
    ),
    op(
      "bias",
      "selection bias",
      f.experts.value,
      { col: 2, row: 0 },
      "MoEGate",
      "Use learned correction only for ranking, never for mixture weights.",
      "choice = score + correction_bias",
      shape("B×S", f.experts.value),
      shape("B×S", f.experts.value),
      [["e_score_correction_bias", shape(f.experts.value)]],
      "instance",
    ),
    op(
      "groups",
      "group filter",
      f.experts.value,
      { col: 3, row: 0 },
      "MoEGate",
      "Score groups by their best pair and retain selected groups. Here the only group is retained.",
      `top-${n(f.groupScoreCount.value)} sum → group top-k → mask`,
      shape("B×S", f.experts.value),
      shape("B×S", f.experts.value),
      [],
      "instance",
      `${n(f.groups.value)} group, ${n(f.topGroups.value)} retained: the mask removes nothing.`,
    ),
    op(
      "topk",
      "expert top-k",
      f.topK.value,
      { col: 4, row: 0 },
      "MoEGate",
      "Select expert indices using corrected choice scores.",
      `indices = topk(choice, ${n(f.topK.value)})`,
      shape("B×S", f.experts.value),
      shape("B×S", f.topK.value),
      [],
      "instance",
    ),
    op(
      "gather-scores",
      "gather scores",
      f.topK.value,
      { col: 5, row: 0 },
      "MoEGate",
      "Read mixture scores from the original sigmoid tensor at selected indices, ignoring correction bias.",
      "weights = gather(score, indices)",
      shape("B×S", f.experts.value),
      shape("B×S", f.topK.value),
      [],
      "instance",
    ),
    op(
      "norm",
      "normalize",
      f.topK.value,
      { col: 6, row: 0 },
      "MoEGate",
      "Normalize the selected original sigmoid scores.",
      `w = score / (Σ selected_score + ${f.correctionDenominator.value})`,
      shape("B×S", f.topK.value),
      shape("B×S", f.topK.value),
      [],
      "instance",
    ),
    op(
      "scale",
      "scale weights",
      f.topK.value,
      { col: 7, row: 0 },
      "MoEGate",
      "Apply the routed multiplier; the shared result is not multiplied by it.",
      `w × ${f.scale.value}`,
      shape("B×S", f.topK.value),
      shape("B×S", f.topK.value),
      [],
      "instance",
    ),
  ];
  const expert = [
    op(
      "input",
      "FFN input",
      f.hidden.value,
      { col: 0, row: 0 },
      "DeepseekV3MLP",
      "Each position is transformed independently of other token positions.",
      "x",
      stream,
      stream,
      [],
      "merged",
    ),
    op(
      "gate",
      "gate_proj",
      inner,
      { col: 1, row: 0 },
      "DeepseekV3MLP",
      "Build the activation gate before multiplying it with the parallel up branch.",
      "x W_gateᵀ",
      stream,
      shape("B", "S", inner),
      [["gate_proj.weight", shape(inner, f.hidden.value)]],
      "low",
    ),
    op(
      "up",
      "up_proj",
      inner,
      { col: 1, row: 1 },
      "DeepseekV3MLP",
      "Build a separate set of features to be gated.",
      "x W_upᵀ",
      stream,
      shape("B", "S", inner),
      [["up_proj.weight", shape(inner, f.hidden.value)]],
      "low",
    ),
    op(
      "act",
      "SiLU",
      inner,
      { col: 2, row: 0 },
      "DeepseekV3MLP",
      "Smoothly gate the first projection.",
      "gate × sigmoid(gate)",
      shape("B", "S", inner),
      shape("B", "S", inner),
      [],
      "low",
    ),
    op(
      "mul",
      "multiply",
      inner,
      { col: 3, row: 0 },
      "DeepseekV3MLP",
      "Combine the activated gate with the up projection elementwise.",
      "SiLU(gate) × up",
      shape("B", "S", inner),
      shape("B", "S", inner),
      [],
      "low",
    ),
    op(
      "down",
      "down_proj",
      f.hidden.value,
      { col: 4, row: 0 },
      "DeepseekV3MLP",
      "Return the gated features to residual width.",
      "product W_downᵀ",
      shape("B", "S", inner),
      stream,
      [["down_proj.weight", shape(f.hidden.value, inner)]],
      "low",
    ),
  ];
  const expertEdges: Edge[] = [
    { from: "input", to: "gate" },
    { from: "input", to: "up" },
    { from: "gate", to: "act" },
    { from: "act", to: "mul" },
    { from: "up", to: "mul" },
    { from: "mul", to: "down" },
  ];
  const dispatch = [
    op(
      "assign",
      "sort assignments",
      f.topK.value,
      { col: 0, row: 0 },
      "DeepseekV3MoE",
      "Count tokens per expert and sort selected assignments by expert id.",
      "order = argsort(flatten(topk_ids))",
      shape("B×S", f.topK.value),
      shape("B×S×K"),
      [],
      "instance",
    ),
    op(
      "gather",
      "gather tokens",
      f.hidden.value,
      { col: 1, row: 0 },
      "DeepseekV3MoE",
      "Gather each input once for each selected expert slot, in sorted assignment order.",
      "sorted_tokens = x[order // K]",
      stream,
      shape("B×S×K", f.hidden.value),
      [],
      "low",
    ),
    op(
      "execute",
      "expert FFNs",
      f.hidden.value,
      { col: 2, row: 0 },
      "DeepseekV3MoE",
      "Run each expert on its assigned tokens; empty experts execute no arithmetic.",
      "expert(sorted_tokens[expert_slice])",
      shape("B×S×K", f.hidden.value),
      shape("B×S×K", f.hidden.value),
      [],
      "low",
    ),
    op(
      "restore",
      "restore slots",
      f.hidden.value,
      { col: 3, row: 0 },
      "DeepseekV3MoE",
      "Undo the sorting and restore token and selected-slot axes.",
      "unsorted[order] = expert_outputs; reshape(N,K,D)",
      shape("B×S×K", f.hidden.value),
      shape("B×S", f.topK.value, f.hidden.value),
      [],
      "low",
    ),
    op(
      "weight",
      "weight outputs",
      f.hidden.value,
      { col: 4, row: 0 },
      "DeepseekV3MoE",
      "Multiply each expert output by its original-score mixture weight.",
      "output × topk_weight[..., None]",
      shape("B×S", f.topK.value, f.hidden.value),
      shape("B×S", f.topK.value, f.hidden.value),
      [],
      "low",
    ),
    op(
      "reduce",
      "sum slots",
      f.hidden.value,
      { col: 5, row: 0 },
      "DeepseekV3MoE",
      "Reduce selected slots and restore the original batch/sequence axes, before the shared path is added.",
      "sum(selected_outputs, slot).reshape(B,S,D)",
      shape("B×S", f.topK.value, f.hidden.value),
      stream,
      [],
      "low",
    ),
  ];
  return (
    <Section
      id="experts"
      title="Choose experts, keep a shared path"
      source={[
        { source: "reference", symbol: "DeepseekV3MoE" },
        { source: "reference", symbol: "MoEGate" },
      ]}
      lede="The router selects a small set of token-specific transforms. A separate shared expert processes every token and joins their weighted sum."
    >
      <OpGraph
        id="moe-topology"
        title="Routed and shared topology"
        heading="Parallel paths meet at addition"
        ops={topology}
        edges={[
          { from: "x", to: "route" },
          { from: "x", to: "shared" },
          { from: "x", to: "expert" },
          { from: "route", to: "expert", kind: "state" },
          { from: "expert", to: "sum" },
          { from: "shared", to: "sum" },
        ]}
      />
      <Prose>
        The router makes a discrete choice with corrected scores. It uses the uncorrected scores to weight the chosen
        outputs, so selection bias cannot directly amplify a chosen expert.
      </Prose>
      <OpGraph
        id="router-ops"
        title="Checkpoint router rule"
        heading="Score, choose and weight"
        ops={router.slice(0, f.groupScoreCount.value * 2)}
        edges={chain(router.slice(0, f.groupScoreCount.value * 2))}
      />
      <Prose>
        Group filtering leaves all experts eligible in this checkpoint. The next graph chooses indices, then returns to
        the original score tensor for its weights.
      </Prose>
      <OpGraph
        id="router-weight-ops"
        title="Selected expert weights"
        heading="Select indices, gather original scores"
        ops={router.slice(f.groupScoreCount.value * 2).map((item, index) => ({ ...item, col: index }))}
        edges={chain(router.slice(f.groupScoreCount.value * 2))}
      />
      <Prose>
        Dispatch gathers the same input token for each selected expert. Each expert uses the gated computation below;
        its result is weighted, reduced across selected slots, then added to the shared output.
      </Prose>
      <OpGraph
        id="dispatch-ops"
        title="Expert dispatch and combination"
        heading="Gather, execute and restore"
        ops={dispatch}
        edges={chain(dispatch)}
      />
      <Prose>
        The routed reduction joins the separate shared FFN at the topology's addition. Sorting is an execution schedule;
        it does not alter token order at the module boundary.
      </Prose>
      <Sentence>
        Layer {n(layer)} uses{" "}
        {f.kindOf(layer) === "dense"
          ? `the dense FFN, inner width ${n(inner)}`
          : `routed expert FFNs, inner width ${n(inner)}`}
        . The shared FFN has inner width {n(f.expert.value * f.shared.value)}.
      </Sentence>
      <OpGraph
        id="expert-ops"
        title="The selected layer's feed-forward transform"
        heading="Inside the feed-forward transform"
        ops={expert}
        edges={expertEdges}
      />
      <Prose>
        Only {n(f.topK.value)} of {n(f.experts.value)} routed expert matrices execute for a token. Shared experts
        execute unconditionally. The dense first layer uses the same gate/up/down arithmetic at its own wider
        intermediate size.
      </Prose>
      <Button onClick={() => setSeed((seed + 1) % (f.demo.seedMax + 1))}>Route another token</Button>
      <Sentence>
        Random-weight router seed {n(seed)} chooses experts {routing.chosen.map(n).join(", ")}; their mixture weights
        sum to {routing.weights.reduce((a, b) => a + b, 0).toFixed(f.demo.radDigits)}. This is a mechanism experiment,
        not a trained-model prediction.
      </Sentence>
      <Figure
        label="Actual rule on a computed random-weight stream"
        caption="Computed by the clean reference router at checkpoint widths with random weights and a synthetic token vector. The bias influences the highlighted selection; bars show original sigmoid scores. Listed weights follow the displayed expert order."
      >
        <MoeRouter
          title="Kimi K2 corrected sigmoid routing"
          routing={routing}
          experts={f.experts.value}
          groups={f.groups.value}
          shared={f.shared.value}
        />
      </Figure>
      <Readout
        label="Computed values"
        items={[
          {
            term: "Routed selection",
            value: `${n(f.topK.value)} / ${n(f.experts.value)}`,
            note: `${((f.selectedFraction.value * f.experts.value) / f.topK.value).toFixed(0)} group; shared separate`,
          },
          {
            term: "Selected routed weights",
            value: String(f.scale.value),
            note: "sum after normalization and scaling",
          },
          { term: "Expert parameters", value: n(f.expertParams.value), note: "gate + up + down per routed expert" },
        ]}
      />
      <Check question="Could a correction bias change an expert's weight without changing the chosen set?">
        No. With the same chosen set, weights are gathered from the same original sigmoid scores, normalized and scaled.
        Correction bias only affects membership.
      </Check>
      <Takeaways
        items={[
          `Each token activates ${n(f.topK.value)} routed experts, a ${f.selectedFraction.value.toFixed(f.demo.radDigits)} fraction of the routed bank, plus the shared path.`,
          "Shared experts have no router logits.",
          "Selection scores and output mixture weights are different tensors.",
        ]}
      />
    </Section>
  );
}
