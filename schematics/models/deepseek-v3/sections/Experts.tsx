import { ToyRouter } from "./ToyRouter";
import { useMemo } from "react";
import {
  Section,
  OpGraph,
  Sub,
  Prose,
  Figure,
  Button,
  Sentence,
  Readout,
  Check,
  Takeaways,
  n,
  shape,
  useUrlNumber,
  type Op,
  type Edge,
} from "../../../kit";
import * as f from "../facts";
import * as ref from "../../../tnsr/tensor";
import { forward, TOY } from "../reference/model";
import { operation as op, chain, routerRefs, ffnRefs, stream, linearDetail } from "../graph";
import { useLayer } from "./Stack";
const LAYOUT = { columns: [0, 1, 2, 3, 4, 5, 6, 7] };

export function Experts() {
  const [layer] = useLayer();
  const dense = f.kindOf(layer) === "dense";
  const d = f.hidden.value;
  const w = dense ? f.denseWidth.value : f.expertWidth.value;
  const [seed, setSeed] = useUrlNumber("seed", f.TOY.seed, 0, f.DEMO.maxSeed);
  const routing = useMemo(() => {
    const ctx = new ref.Ctx("value", null, seed, f.TOY.gain);
    forward(ctx, TOY, f.TOY.ids);
    const get = (id: string, width: number) =>
      Array.from(ctx.trace.get(`h${f.TOY.firstDense}.${id}`)!.data!.slice(0, width));
    return {
      scores: get("sigmoid", f.TOY.experts),
      chosen: get("topk", f.TOY.topK),
      chosenGroups: get("groups", f.TOY.topGroups),
      weights: get("routeweights", f.TOY.topK),
    };
  }, [seed]);
  const e = f.experts.value,
    scoreShape = shape("B·S", e),
    chosenShape = shape("B·S", f.topK.value);
  const router: Op[] = [
    op(
      "flatten",
      "flatten tokens",
      0,
      d,
      {
        what: "Treat each token independently for expert selection.",
        formula: "tokens = hidden.view(B·S, D)",
        inputs: [["hidden", stream]],
        output: ["tokens", shape("B·S", d)],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "logits",
      "router linear",
      1,
      e,
      {
        what: "Compute a float32 affinity for every routed expert, with no linear bias.",
        formula: "logits = float32(tokens) Wᵀ",
        inputs: [["tokens", shape("B·S", d)]],
        output: ["logits", scoreShape],
        params: [["weight", shape(e, d)]],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "sigmoid",
      "sigmoid",
      2,
      e,
      {
        what: "Map each affinity to an independent bounded score; this is not softmax over experts.",
        formula: "scores = sigmoid(logits)",
        inputs: [["logits", scoreShape]],
        output: ["scores", scoreShape],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "bias",
      "choice bias",
      LAYOUT.columns[3],
      e,
      {
        what: "Apply learned correction biases only to the values used for selection.",
        formula: "choice = scores + correction_bias",
        inputs: [["scores", scoreShape]],
        output: ["choice", scoreShape],
        params: [["correction_bias", shape(e)]],
        refs: routerRefs,
        note: "Keep original scores for combination weights. Demo biases are synthetic, not downloaded checkpoint values.",
      },
      "instance",
    ),
    op(
      "groups",
      "score groups",
      LAYOUT.columns[4],
      f.groups.value,
      {
        what: "Reshape expert scores into groups and sum each group's best corrected scores.",
        formula: `group_score = sum(top${n(f.groupPeakCount.value)}(choice.view(N,G,E/G)))`,
        inputs: [["choice", scoreShape]],
        output: ["group_scores", shape("B·S", f.groups.value)],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "mask",
      "keep groups",
      LAYOUT.columns[5],
      e,
      {
        what: "Keep the highest scoring groups and set every expert in other groups to negative infinity.",
        formula: `group_ids = topk(group_scores, ${n(f.topGroups.value)}); choice[other groups] = −∞`,
        inputs: [
          ["group_scores", shape("B·S", f.groups.value)],
          ["choice", scoreShape],
        ],
        output: ["masked choice", scoreShape],
        refs: routerRefs,
        note: "Publisher inference Gate multiplies excluded groups by zero instead. Negative corrected scores can make that differ from the reference's negative infinity.",
      },
      "instance",
    ),
    op(
      "select",
      "expert top-k",
      LAYOUT.columns[6],
      f.topK.value,
      {
        what: "Select expert ids only within the surviving groups.",
        formula: `ids = topk(masked_choice, ${n(f.topK.value)})`,
        inputs: [["masked choice", scoreShape]],
        output: ["ids", chosenShape],
        refs: routerRefs,
      },
      "instance",
    ),
  ];
  const weights: Op[] = [
    op(
      "gather",
      "gather original",
      0,
      f.topK.value,
      {
        what: "Read the uncorrected sigmoid score for each chosen expert.",
        formula: "weights = scores.gather(ids)",
        inputs: [
          ["scores", scoreShape],
          ["ids", chosenShape],
        ],
        output: ["weights", chosenShape],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "normalize",
      "renormalize",
      1,
      f.topK.value,
      {
        what: "Make the selected scores sum to unity before the routed output scale.",
        formula: `weights /= sum(weights) + ${f.routeEpsilon.value}`,
        inputs: [["weights", chosenShape]],
        output: ["normalized", chosenShape],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "scale",
      "route scale",
      2,
      f.topK.value,
      {
        what: "Scale the routed expert mixture; this does not scale the always-on shared output.",
        formula: `weights *= ${n(f.routeScale.value)}`,
        inputs: [["normalized", chosenShape]],
        output: ["route weights", chosenShape],
        refs: routerRefs,
      },
      "instance",
    ),
  ];
  const ffRaw: Op[] = [
    op(
      "x",
      "normalized token",
      0,
      d,
      {
        what: dense
          ? "Read a token from the selected dense layer."
          : "Read a token assigned to a routed expert; the shared expert uses the original token too.",
        formula: "x = normalized token",
        inputs: [["stream", stream]],
        output: ["x", stream],
        refs: ffnRefs,
      },
      "merged",
      0,
      "tensor",
    ),
    op(
      "gate",
      "gate projection",
      1,
      w,
      linearDetail("Build the branch that will control the elementwise product.", d, w, ffnRefs),
      "low",
    ),
    op(
      "up",
      "up projection",
      1,
      w,
      linearDetail("Build the separate value branch from the same input.", d, w, ffnRefs),
      "low",
      1,
    ),
    op(
      "silu",
      "SiLU",
      2,
      w,
      {
        what: "Activate only the gate branch.",
        formula: "gate = gate · sigmoid(gate)",
        inputs: [["gate", shape("B", "S", w)]],
        output: ["activated gate", shape("B", "S", w)],
        refs: ffnRefs,
      },
      "low",
    ),
    op(
      "mul",
      "multiply",
      LAYOUT.columns[3],
      w,
      {
        what: "Gate the up projection element by element.",
        formula: "product = SiLU(gate_proj(x)) · up_proj(x)",
        inputs: [
          ["activated gate", shape("B", "S", w)],
          ["up", shape("B", "S", w)],
        ],
        output: ["product", shape("B", "S", w)],
        refs: ffnRefs,
      },
      "low",
      0,
      "mul",
      "",
    ),
    op(
      "down",
      "down projection",
      LAYOUT.columns[4],
      d,
      linearDetail(
        "Return the gated transform to the residual width.",
        w,
        d,
        ffnRefs,
        "vLLM fuses gate and up projections, then uses SiluAndMul. The arithmetic represented here stays the same.",
      ),
      "low",
    ),
  ];
  // Routed experts receive packed token batches N_e rather than a rectangular B,S batch.
  const expertShape = (value: string) => value.replace("[B, S,", "[N_e,");
  const ff: Op[] = ffRaw.map((node) =>
    dense
      ? node
      : {
          ...node,
          detail: {
            ...node.detail,
            inputs: node.detail.inputs?.map(([name, dims]) => [name, expertShape(dims)] as const),
            output: node.detail.output
              ? ([node.detail.output[0], expertShape(node.detail.output[1])] as const)
              : undefined,
            note: [
              node.detail.note,
              "N_e is the packed count of tokens assigned to this expert; assignments are unpacked after its FFN.",
            ]
              .filter(Boolean)
              .join(" "),
          },
        },
  );
  const ffEdges: Edge[] = [
    { from: "x", to: "gate" },
    { from: "x", to: "up" },
    { from: "gate", to: "silu" },
    { from: "silu", to: "mul" },
    { from: "up", to: "mul" },
    { from: "mul", to: "down" },
  ];
  const moeRefs = [
    { source: "reference", symbol: "DeepseekV3MoE.moe_infer" },
    { source: "reference", symbol: "DeepseekV3MoE.forward" },
    { source: "serving", symbol: "DeepseekV2MoE" },
    { source: "publisher", symbol: "MoE.forward" },
  ];
  const combine: Op[] = [
    op(
      "dispatch",
      "sort + dispatch",
      0,
      d,
      {
        what: "Count expert assignments, sort flattened expert ids and gather the matching token for each route.",
        formula: "idxs = argsort(ids.flatten()); sorted = x[idxs // K]",
        inputs: [
          ["tokens", shape("B·S", d)],
          ["ids", chosenShape],
        ],
        output: ["dispatched tokens", shape("B·S·K", d)],
        refs: moeRefs,
        note: `ep_size=${n(f.epSize.value)} disables distributed all-to-all in the reference.`,
      },
      "low",
    ),
    op(
      "expert",
      "routed FFNs",
      1,
      d,
      {
        what: "Run each nonempty expert on the tokens assigned to it.",
        formula: "expert_out = down(SiLU(gate(x)) · up(x))",
        inputs: [["dispatched tokens", shape("B·S·K", d)]],
        output: ["expert outputs", shape("B·S·K", d)],
        params: [
          ["gate / up", shape(f.expertWidth.value, d)],
          ["down", shape(d, f.expertWidth.value)],
        ],
        refs: moeRefs,
      },
      "low",
    ),
    op(
      "undo",
      "undo sort",
      2,
      d,
      {
        what: "Put expert outputs back in token-route order, introducing the selected-expert axis.",
        formula: "new_x[idxs] = outs; new_x.view(N,K,D)",
        inputs: [["expert outputs", shape("B·S·K", d)]],
        output: ["route outputs", shape("B·S", f.topK.value, d)],
        refs: moeRefs,
      },
      "low",
    ),
    op(
      "weighted",
      "weighted sum",
      LAYOUT.columns[3],
      d,
      {
        what: "Cast to weight dtype, multiply by the original-score weights, sum routes and cast back.",
        formula: "routed = Σ_e route_weight_e · expert_output_e",
        inputs: [
          ["route outputs", shape("B·S", f.topK.value, d)],
          ["weights", chosenShape],
        ],
        output: ["routed", stream],
        refs: moeRefs,
      },
      "low",
    ),
    op(
      "shared",
      "shared FFN",
      LAYOUT.columns[3],
      d,
      {
        what: "Run the always-on shared expert directly on the original normalized tokens.",
        formula: "shared = down(SiLU(gate(x)) · up(x))",
        inputs: [["identity", stream]],
        output: ["shared", stream],
        params: [
          ["gate / up", shape(f.shared.value * f.expertWidth.value, d)],
          ["down", shape(d, f.shared.value * f.expertWidth.value)],
        ],
        refs: moeRefs,
        note: "This branch has no router score, no top-k membership and no route-scale multiplier.",
      },
      "low",
      1,
    ),
    op(
      "sum",
      "add shared",
      LAYOUT.columns[4],
      d,
      {
        what: "Combine the routed mixture with the separate shared output.",
        formula: "ffn = routed + shared",
        inputs: [
          ["routed", stream],
          ["shared", stream],
        ],
        output: ["ffn", stream],
        refs: moeRefs,
      },
      "merged",
      0,
      "sum",
      "",
    ),
  ];
  const overview: Op[] = [
    op(
      "moex",
      "normalized stream",
      0,
      d,
      {
        what: "All MoE paths read the same normalized token stream.",
        formula: "x = post_attention_layernorm(hidden)",
        inputs: [["hidden", stream]],
        output: ["x", stream],
        refs: moeRefs,
      },
      "merged",
      1,
      "tensor",
      stream,
    ),
    op(
      "choose",
      "Choose identities",
      1,
      e,
      {
        what: "The choice graph applies sigmoid, correction bias, group restriction and expert top-k.",
        formula: "ids = topk(group_mask(sigmoid(x Wᵀ) + bias))",
        inputs: [["x", stream]],
        output: ["ids", chosenShape],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "mixture",
      "Compute weights",
      2,
      f.topK.value,
      {
        what: "The weight graph gathers original scores and renormalizes before scaling.",
        formula: "w = scale · gather(scores,ids) / sum(gather(scores,ids))",
        inputs: [
          ["ids", chosenShape],
          ["scores", scoreShape],
        ],
        output: ["weights", chosenShape],
        refs: routerRefs,
      },
      "instance",
    ),
    op(
      "routedff",
      "Dispatch + FFNs",
      LAYOUT.columns[3],
      d,
      {
        what: "The combination graph dispatches tokens to selected FFNs and sums their weighted outputs.",
        formula: "routed = Σ_e w_e · FFN_e(x)",
        inputs: [
          ["x", stream],
          ["ids,weights", chosenShape],
        ],
        output: ["routed", stream],
        refs: moeRefs,
      },
      "low",
      1,
    ),
    op(
      "always",
      "Shared FFN",
      1,
      d,
      {
        what: "The shared FFN reads tokens without participating in selection.",
        formula: "shared = FFN_shared(x)",
        inputs: [["x", stream]],
        output: ["shared", stream],
        refs: moeRefs,
      },
      "low",
      2,
    ),
    op(
      "moesum",
      "Add shared",
      LAYOUT.columns[4],
      d,
      {
        what: "Combine the routed mixture with the always-on shared transform.",
        formula: "out = routed + shared",
        inputs: [
          ["routed", stream],
          ["shared", stream],
        ],
        output: ["out", stream],
        refs: moeRefs,
      },
      "merged",
      1,
      "sum",
      "",
    ),
  ];
  return (
    <Section
      id="experts"
      title="Dense transforms and selective experts"
      source={[
        { source: "reference", symbol: "DeepseekV3MLP" },
        { source: "reference", symbol: "MoEGate" },
        { source: "reference", symbol: "DeepseekV3MoE" },
      ]}
      lede="Every feed-forward transform uses a gated product. Later layers keep many small transforms, select a few for each token, and add a shared transform that never competes for routing."
    >
      <OpGraph
        heading="How the MoE paths connect"
        id="moe-topology"
        title="MoE topology"
        ops={overview}
        edges={[
          { from: "moex", to: "choose" },
          { from: "moex", to: "always" },
          { from: "moex", to: "routedff" },
          { from: "choose", to: "mixture" },
          { from: "mixture", to: "routedff" },
          { from: "routedff", to: "moesum" },
          { from: "always", to: "moesum" },
        ]}
        caption="Each box names a detailed computation below. The shared FFN bypasses the router and joins after the weighted routed output."
      />
      <Prose>
        The FFN graph below is reused by dense layers, routed experts and the shared path, with the corresponding
        intermediate width.
      </Prose>
      <Sentence>
        Layer {n(layer)} uses {dense ? `the dense width ${n(w)}` : `expert width ${n(w)}`}. Its FFN below changes with
        the layer selection.
      </Sentence>
      <OpGraph
        heading={dense ? "Selected dense FFN" : "One selected expert's FFN"}
        id="ffn-op"
        title="Gated feed-forward transform"
        ops={ff}
        edges={ffEdges}
      />
      <Prose>
        The router below applies to the MoE layers. It remains visible when a dense layer is selected so the default
        page still explains the complete architecture.
      </Prose>
      <OpGraph
        heading="Choose expert identities"
        id="router-op"
        title="Correction-aware group routing"
        ops={router}
        edges={chain(router)}
      />
      <Prose>
        Expert identities are now fixed. Combination weights come from the saved original scores, without correction
        bias.
      </Prose>
      <OpGraph
        heading="Compute combination weights"
        id="weight-op"
        title="Unbiased routing weights"
        ops={weights}
        edges={chain(weights)}
      />
      <Prose>
        The selected tokens run through the expert FFNs. A separate shared FFN consumes the original tokens; it joins
        only after the routed weighted sum.
      </Prose>
      <OpGraph
        heading="Execute and combine"
        id="combine-op"
        title="Routed output plus always-on shared output"
        ops={combine}
        edges={[
          { from: "dispatch", to: "expert" },
          { from: "expert", to: "undo" },
          { from: "undo", to: "weighted" },
          { from: "weighted", to: "sum" },
          { from: "shared", to: "sum" },
        ]}
      />
      <Sub>Route a computed toy token</Sub>
      <Prose>
        The diagrams above retain checkpoint widths. This widget runs the clean reference at toy widths with random
        weights on the first token of the toy sequence. The Run it section shows the complete sequence and reconstructed
        KV.
      </Prose>
      <Button onClick={() => setSeed((current) => (current + 1) % (f.DEMO.maxSeed + 1))}>Route another token</Button>
      <Sentence>
        Toy seed {n(seed)} selects experts {routing.chosen.map(n).join(", ")} in groups{" "}
        {routing.chosenGroups.map(n).join(", ")}; the shared path stays on.
      </Sentence>
      <Figure
        label="Computed grouped routing"
        caption="Random toy weights generate the normalized stream, router logits and correction biases through the reference. Bar heights are original sigmoid scores; highlights use corrected scores. This shows the mechanism, not a trained checkpoint prediction."
      >
        <ToyRouter routing={routing} />
      </Figure>
      <Readout
        label="Expert selection"
        items={[
          { term: "Toy selected", value: `${n(f.TOY.topK)} / ${n(f.TOY.experts)}` },
          { term: "Toy groups kept", value: `${n(f.TOY.topGroups)} / ${n(f.TOY.groups)}` },
          { term: "Weight sum", value: n(routing.weights.reduce((a, b) => a + b, 0)) },
          {
            term: "Checkpoint FFN matrices used",
            value: n(f.activeFfnParams.value),
            note: `of ${n(f.storedFfnParams.value)} stored parameters per MoE layer`,
          },
        ]}
      />
      <Check question="If correction bias changes a selected expert's rank, does its mixture weight directly include that bias?">
        No. Bias changes which ids survive. The weights gather the original sigmoid scores for those ids, then
        renormalize and scale them. The shared output is added independently.
      </Check>
      <Takeaways
        items={[
          "Group selection sums the strongest corrected scores; it is not a top-k over individual groups' maximum score.",
          `${n(f.topK.value)} routed experts and ${n(f.shared.value)} shared expert use ${n(f.activeFfnParams.value)} FFN parameters per token, while the whole bank stores ${n(f.storedFfnParams.value)}.`,
          "Sparse activation saves expert computation; all expert matrices still belong to the stored model.",
        ]}
      />
    </Section>
  );
}
