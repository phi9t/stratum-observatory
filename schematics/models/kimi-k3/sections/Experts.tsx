import {
  Section,
  Prose,
  Figure,
  MoeRouter,
  Button,
  Sentence,
  Readout,
  OpGraph,
  Takeaways,
  Check,
  useUrlNumber,
  ref,
  n,
  shape,
  type Op,
} from "../../../kit";
import * as f from "../facts";
import { Graph, type Step } from "./graph";
import { KimiMoEGate } from "../reference/model";
const refs = [
  { source: "reference", symbol: "KimiSparseMoeBlock.forward" },
  { source: "serving", symbol: "KimiK3MoE._forward_unfused" },
];
export function Experts() {
  const [seed, setSeed] = useUrlNumber("route-seed", 1, 0, f.DEMO.seedMax);
  const [layer] = useUrlNumber("layer", 0, 0, f.layers.value - 1);
  const ctx = new ref.Ctx("value", null, seed);
  const x = ref.input(ctx, [1, 1, f.hidden.value], () =>
    Array.from({ length: f.hidden.value }, (_, i) => Math.sin(i + seed)),
  );
  const routing = KimiMoEGate(
    ctx,
    { ...f.TOY, hidden: f.hidden.value, experts: f.experts.value, topK: f.topK.value },
    x,
    "demo.gate",
  );
  const selected = Array.from(routing.ids.data!);
  const scores = Array.from(routing.scores.data!);
  const router: Step[] = [
    {
      id: "logits",
      label: "router linear",
      width: f.experts.value,
      what: "Score every routed expert from the full-width stream in float32.",
      formula: "logits = x W_gateᵀ",
      params: [["weight", shape(f.experts.value, f.hidden.value)]],
    },
    {
      id: "scores",
      label: "sigmoid",
      width: f.experts.value,
      what: "Produce independent positive scores rather than an expert softmax.",
      formula: "scores = sigmoid(logits)",
    },
    {
      id: "choice",
      label: "correction bias",
      width: f.experts.value,
      what: "Adjust selection rankings only.",
      formula: "choice = scores + correction_bias",
      params: [["e_score_correction_bias", shape(f.experts.value)]],
      note: "The ungrouped branch is taken: group count and retained group count are both one.",
    },
    {
      id: "topk",
      label: "top-k indices",
      width: f.topK.value,
      what: "Select routed experts from corrected scores.",
      formula: `ids = topk(choice, ${n(f.topK.value)})`,
      output: shape("B", "S", f.topK.value),
    },
    {
      id: "gather",
      inputs: [
        ["scores", shape("B", "S", f.experts.value)],
        ["ids", shape("B", "S", f.topK.value)],
      ],
      label: "gather raw scores",
      width: f.topK.value,
      what: "Retrieve unbiased scores of selected experts.",
      formula: "weights = gather(scores,ids)",
    },
    {
      id: "renorm",
      label: "normalize + scale",
      width: f.topK.value,
      what: "Normalize selected weights and apply the checkpoint routing scale.",
      formula: `w = selected / (sum(selected) + ${n(f.routingEps.value)}); scale = unity`,
      note: "Correction bias is absent from the denominator and weights.",
    },
  ];
  const expert: Step[] = [
    {
      id: "gate",
      col: 0,
      row: 0,
      inputs: [["z", shape("B", "S", f.latent.value)]],
      label: "gate projection",
      width: f.intermediate.value,
      what: "Build the expert activation gate.",
      formula: "a = z W1ᵀ",
      params: [["w1.weight", shape(f.intermediate.value, f.latent.value)]],
      input: shape("B", "S", f.latent.value),
      lane: "low",
    },
    {
      id: "up",
      col: 0,
      row: 1,
      inputs: [["z", shape("B", "S", f.latent.value)]],
      label: "up projection",
      width: f.intermediate.value,
      what: "Independently project the same latent into value channels.",
      formula: "b = z W3ᵀ",
      params: [["w3.weight", shape(f.intermediate.value, f.latent.value)]],
      input: shape("B", "S", f.latent.value),
      lane: "low",
    },
    {
      id: "concat",
      col: 1,
      row: 0,
      inputs: [
        ["a", shape("B", "S", f.intermediate.value)],
        ["b", shape("B", "S", f.intermediate.value)],
      ],
      label: "concat gate / up",
      width: 2 * f.intermediate.value,
      what: "Join the gate and value slices for the fused activation.",
      formula: "gate_up = concat(a,b)",
      lane: "low",
    },
    {
      id: "situ",
      col: 2,
      label: "SiTU and multiply",
      width: f.intermediate.value,
      what: "Saturate both paths, sigmoid-gate the first, then multiply.",
      formula: `z = ${n(f.beta.value)} tanh(a/${n(f.beta.value)}) sigmoid(a) × ${n(f.linearBeta.value)} tanh(b/${n(f.linearBeta.value)})`,
      lane: "low",
    },
    {
      id: "down",
      col: 3,
      label: "down projection",
      width: f.latent.value,
      what: "Return each selected expert to latent width.",
      formula: "y_e = z W2ᵀ",
      params: [["w2.weight", shape(f.latent.value, f.intermediate.value)]],
      lane: "low",
    },
  ];
  const latent: Step[] = [
    {
      id: "compress",
      label: "latent down-proj",
      width: f.latent.value,
      what: "Compress the original FFN input after routing.",
      formula: "z = x W_downᵀ",
      params: [["weight", shape(f.latent.value, f.hidden.value)]],
      lane: "low",
    },
    {
      id: "dispatch",
      label: "gather / dispatch",
      width: f.latent.value,
      what: "Group token rows by selected expert id; unselected experts receive no rows.",
      formula: "expert inputs = gather(z, selected token rows)",
      lane: "low",
    },
    {
      id: "experts",
      label: "expert SiTU FFNs",
      width: f.latent.value,
      what: "Run each selected token through its assigned expert.",
      formula: "y_e = expert_e(z)",
      lane: "low",
    },
    {
      id: "weighted",
      label: "unpermute + sum",
      width: f.latent.value,
      what: "Restore token order and sum weighted expert outputs before normalization.",
      formula: "y = Σ_e w_e y_e",
      lane: "low",
    },
    {
      id: "norm",
      label: "latent RMSNorm",
      width: f.latent.value,
      what: "Normalize the combined expert output, not individual expert outputs.",
      formula: "y ← RMSNorm(y)",
      params: [["weight", shape(f.latent.value)]],
      lane: "low",
      note: "Distributed serving must reduce partial expert sums before this norm.",
    },
    {
      id: "expand",
      label: "latent up-proj",
      width: f.hidden.value,
      what: "Expand the normalized routed result back to the stream.",
      formula: "routed = y W_upᵀ",
      params: [["weight", shape(f.hidden.value, f.latent.value)]],
      lane: "low",
    },
  ];
  const fullWidth = f.isMoe(layer) ? f.sharedWidth.value : f.denseIntermediate.value;
  const shared: Step[] = [
    {
      id: "gate",
      label: "gate_proj",
      width: fullWidth,
      col: 0,
      row: 0,
      lane: "low",
      what: "Project the original full-width input into activation gates.",
      formula: "a = x W_gateᵀ",
      inputs: [["x", shape("B", "S", f.hidden.value)]],
      params: [["weight", shape(fullWidth, f.hidden.value)]],
    },
    {
      id: "up",
      label: "up_proj",
      width: fullWidth,
      col: 0,
      row: 1,
      lane: "low",
      what: "Independently project the same input into value channels.",
      formula: "b = x W_upᵀ",
      inputs: [["x", shape("B", "S", f.hidden.value)]],
      params: [["weight", shape(fullWidth, f.hidden.value)]],
    },
    {
      id: "concat",
      label: "concat gate / up",
      width: fullWidth * 2,
      col: 1,
      row: 0,
      lane: "low",
      what: "Join the parallel gate and value slices.",
      formula: "gate_up = concat(a,b)",
      inputs: [
        ["a", shape("B", "S", fullWidth)],
        ["b", shape("B", "S", fullWidth)],
      ],
    },
    {
      id: "situ",
      label: "SiTU × value",
      width: fullWidth,
      col: 2,
      row: 0,
      lane: "low",
      what: "Apply the same saturating activation as routed experts.",
      formula: `z = ${n(f.beta.value)} tanh(a/${n(f.beta.value)}) sigmoid(a) × ${n(f.linearBeta.value)} tanh(b/${n(f.linearBeta.value)})`,
    },
    {
      id: "down",
      label: "down_proj",
      width: f.hidden.value,
      col: 3,
      row: 0,
      lane: "low",
      what: "Return to stream width. Dense layers return this directly; MoE layers add it to the separately expanded routed result.",
      formula: "y = z W_downᵀ",
      params: [["weight", shape(f.hidden.value, fullWidth)]],
    },
  ];
  const topology: Op[] = [
    {
      id: "x",
      label: "normed stream",
      width: f.hidden.value,
      sub: n(f.hidden.value),
      kind: "tensor",
      col: 0,
      row: 0,
      detail: { what: "The router and shared FFN see the full stream.", refs },
    },
    {
      id: "route",
      label: "router",
      width: f.experts.value,
      sub: n(f.experts.value),
      lane: "instance",
      col: 1,
      row: 0,
      detail: { what: "Select experts and mixture weights.", refs },
    },
    {
      id: "latent",
      label: "compress + experts",
      width: f.latent.value,
      sub: n(f.latent.value),
      lane: "low",
      col: 2,
      row: 0,
      detail: { what: "Experts operate in compressed width.", refs },
    },
    {
      id: "expand",
      label: "sum, norm, expand",
      width: f.hidden.value,
      sub: n(f.hidden.value),
      lane: "low",
      col: 3,
      row: 0,
      detail: { what: "Normalization follows weighted aggregation.", refs },
    },
    {
      id: "shared",
      label: "shared FFN",
      width: f.hidden.value,
      sub: n(f.sharedWidth.value),
      lane: "low",
      col: 2,
      row: 1,
      detail: { what: "Always-on experts consume the original stream.", refs },
    },
    {
      id: "add",
      label: "add",
      width: f.hidden.value,
      kind: "sum",
      col: 4,
      row: 0,
      detail: { what: "Combine shared and routed outputs.", refs },
    },
  ];
  const summaries: Record<
    string,
    { formula: string; inputs: readonly (readonly [string, string])[]; output: readonly [string, string] }
  > = {
    x: {
      formula: "x = post_attention_layernorm(depth_mix)",
      inputs: [["mixture", shape("B", "S", f.hidden.value)]],
      output: ["x", shape("B", "S", f.hidden.value)],
    },
    route: {
      formula: "ids = topk(sigmoid(xWᵀ) + bias); w = normalize(gather(sigmoid(xWᵀ),ids))",
      inputs: [["x", shape("B", "S", f.hidden.value)]],
      output: ["ids,w", shape("B", "S", f.topK.value)],
    },
    latent: {
      formula: "z = compress(x); y_e = expert_e(z)",
      inputs: [
        ["x", shape("B", "S", f.hidden.value)],
        ["ids", shape("B", "S", f.topK.value)],
      ],
      output: ["y_e", shape("B", "S", f.topK.value, f.latent.value)],
    },
    expand: {
      formula: "routed = expand(RMSNorm(Σ_e w_e y_e))",
      inputs: [
        ["y_e", shape("B", "S", f.topK.value, f.latent.value)],
        ["w", shape("B", "S", f.topK.value)],
      ],
      output: ["routed", shape("B", "S", f.hidden.value)],
    },
    shared: {
      formula: "shared = down(SiTU(gate(x),up(x)))",
      inputs: [["x", shape("B", "S", f.hidden.value)]],
      output: ["shared", shape("B", "S", f.hidden.value)],
    },
    add: {
      formula: "y = shared + routed",
      inputs: [["shared,routed", shape("B", "S", f.hidden.value)]],
      output: ["y", shape("B", "S", f.hidden.value)],
    },
  };
  const detailedTopology = topology.map((op) => ({ ...op, detail: { ...op.detail, ...summaries[op.id] } }));
  return (
    <Section
      id="experts"
      title="Experts in a narrower space"
      source={refs}
      lede="The router sees the full stream, but routed experts work in a compressed latent. Shared experts stay at full width and run independently of router choice."
    >
      <Sentence>
        Selected layer {n(layer)} uses{" "}
        {f.isMoe(layer)
          ? "this latent expert block"
          : "the dense FFN instead; its gate/up width is " + n(f.denseIntermediate.value)}
        . The routed mechanism below describes the MoE layers.
      </Sentence>
      <OpGraph
        id="moe-topology"
        title="Parallel routed and shared paths"
        heading="The paths and where they meet"
        ops={detailedTopology}
        edges={[
          { from: "x", to: "route" },
          { from: "x", to: "latent" },
          { from: "route", to: "latent" },
          { from: "latent", to: "expand" },
          { from: "x", to: "shared" },
          { from: "expand", to: "add" },
          { from: "shared", to: "add" },
        ]}
      />
      <Prose>
        The correction bias chooses experts; original scores determine their weights. Group pruning is bypassed by this
        checkpoint.
      </Prose>
      <Graph
        id="router"
        title="Expert selection"
        heading="Choose routed experts"
        steps={router}
        edges={[
          ...router.slice(1).map((s, i) => ({ from: router[i].id, to: s.id })),
          { from: "scores", to: "gather", kind: "skip" },
        ]}
        source="KimiMoEGate.forward"
        serving="KimiK3MoE.__init__"
      />
      <Prose>
        The routed path compresses only after routing, then aggregates expert outputs before its learned norm.
      </Prose>
      <Graph
        id="latent-experts"
        title="Routed latent path"
        heading="Compress, dispatch and aggregate"
        steps={latent}
        source="KimiSparseMoeBlock.forward"
        serving="KimiK3MoE._forward_unfused"
      />
      <Prose>
        Within each assigned expert, gate and up projections consume the same latent in parallel; neither consumes the
        other projection.
      </Prose>
      <Graph
        id="expert-ffn"
        title="One routed expert"
        heading="Inside an expert"
        steps={expert}
        edges={[
          { from: "gate", to: "concat" },
          { from: "up", to: "concat" },
          { from: "concat", to: "situ" },
          { from: "situ", to: "down" },
        ]}
        source="KimiBlockSparseMLP.forward"
        serving="KimiK3MLP.forward"
      />
      <Prose>The always-on shared path uses the same activation recipe, with its own full-width weights.</Prose>
      <Graph
        id="shared-experts"
        title="Full-width feed-forward path"
        heading={f.isMoe(layer) ? "Inside the shared experts" : "Inside the selected dense FFN"}
        steps={shared}
        edges={[
          { from: "gate", to: "concat" },
          { from: "up", to: "concat" },
          { from: "concat", to: "situ" },
          { from: "situ", to: "down" },
        ]}
        source="KimiMLP.forward"
        serving="KimiK3MoE._forward_unfused"
      />
      <Button onClick={() => setSeed((seed + 1) % (f.DEMO.seedMax + 1))}>Route another input</Button>
      <Sentence>
        With random router weights and synthetic stream seed {n(seed)}, selected routed experts are{" "}
        {selected.map(n).join(", ")}. Shared experts are always on.
      </Sentence>
      <Figure
        label="Computed checkpoint routing rule"
        caption="Computed by the clean router from random weights, with the checkpoint's exact expert count and top-k. The heights show original sigmoid scores; bias-adjusted rankings choose the highlighted entries."
      >
        <MoeRouter
          title="Routed scores and selected experts"
          experts={f.experts.value}
          shared={f.shared.value}
          routing={{
            scores,

            chosen: selected,
            weights: Array.from(routing.weights.data!),
            chosenGroups: [0],
          }}
        />
      </Figure>
      <Readout
        label="Computed values"
        items={[
          { term: "Selected fraction", value: `${n(f.topK.value)} / ${n(f.experts.value)}` },
          { term: "Weight sum", value: n(Array.from(routing.weights.data!).reduce((a, b) => a + b, 0)) },
          { term: "Latent / stream", value: `${n(f.latent.value)} / ${n(f.hidden.value)}` },
        ]}
      />
      <Check question="Can increasing correction bias directly increase an expert's mixture weight while its selected set stays fixed?">
        No. Bias affects selection only. Weights gather original sigmoid scores and normalize those scores.
      </Check>
      <Takeaways
        items={[
          `The latent width ${n(f.latent.value)} is ${n(f.hidden.value / f.latent.value)} times narrower than the stream.`,
          `Only ${n(f.topK.value)} of ${n(f.experts.value)} routed experts process each token, but every expert's weights remain resident.`,
          `Shared experts consume the original stream and have no router logits.`,
        ]}
      />
    </Section>
  );
}
