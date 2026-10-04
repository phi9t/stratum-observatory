import {
  Section,
  OpGraph,
  Prose,
  Figure,
  MoeRouter,
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
import { routerExperiment } from "../experiments";
export function Experts({ layer }: { layer: number }) {
  const [seed, setSeed] = useUrlNumber("routerSeed", f.DEFAULT_SEED, 0, f.MAX_SEED);
  const routing = routerExperiment(seed);
  const d = f.hidden.value,
    i = f.expertWidth.value,
    e = f.experts.value,
    k = f.topK.value;
  const s = shape("N", d),
    scores = shape("N", e),
    choices = shape("N", k),
    inner = shape("Ne", i);
  const refs = [
    { source: "reference", symbol: "Qwen3MoeSparseMoeBlock.forward" },
    { source: "serving", symbol: "Qwen3MoeSparseMoeBlock.forward" },
  ];
  const make = (
    id: string,
    label: string,
    width: number,
    out: string,
    what: string,
    formula: string,
    inputs: readonly (readonly [string, string])[],
    params: readonly (readonly [string, string])[] = [],
    lane: Op["lane"] = "instance",
  ): Omit<Op, "col" | "row"> => ({
    id,
    label,
    width,
    sub: n(width),
    lane,
    detail: {
      what,
      formula,
      inputs,
      output: [id, out],
      params,
      refs,
      note: `Parameter namespace model.layers.${n(layer)}.mlp; expert matrices live under experts.e. N = B × S; Ne is the number of tokens selected for this expert. No shared-expert branch exists. Ne depends on actual routes; shape mode uses an all-token upper bound, not a predicted expert load.`,
    },
  });
  const topology: Op[] = [
    {
      ...make(
        "x",
        "normalized tokens",
        d,
        s,
        "Post-attention stream entering the MoE sublayer.",
        "x = post_attention_layernorm(a)",
        [["a", shape("B", "S", d)]],
        [],
        "merged",
      ),
      col: 0,
      row: 1,
    },
    {
      ...make(
        "router",
        "routing rule",
        k,
        choices,
        "Choose experts and weights for each token.",
        "p = normalize(topk(softmax(xW_gateᵀ)))",
        [["x", s]],
      ),
      col: 1,
      row: 0,
    },
    {
      ...make(
        "dispatch",
        "dispatch tokens",
        d,
        s,
        "Gather only the tokens assigned to each expert.",
        "x_e = x[tokens selected for e]",
        [
          ["x", s],
          ["expert ids", choices],
        ],
        [],
        "low",
      ),
      col: 2,
      row: 1,
    },
    {
      ...make(
        "experts",
        "expert banks",
        d,
        s,
        "Run the same gated MLP architecture with each selected expert's own weights.",
        "z_e = W_down,e(SiLU(W_gate,e x_e) ⊙ W_up,e x_e)",
        [["x_e", s]],
        [],
        "low",
      ),
      col: 3,
      row: 1,
    },
    {
      ...make(
        "combine",
        "weighted scatter",
        d,
        shape("B", "S", d),
        "Weight each contribution and accumulate into its original token slot.",
        "y[t] = Σ_e p[t,e] z_e[t]",
        [
          ["z_e", s],
          ["p", choices],
        ],
        [],
        "low",
      ),
      col: 4,
      row: 1,
    },
  ];
  const topEdges: Edge[] = [
    { from: "x", to: "router" },
    { from: "x", to: "dispatch" },
    { from: "router", to: "dispatch" },
    { from: "dispatch", to: "experts" },
    { from: "experts", to: "combine" },
    { from: "router", to: "combine", kind: "state" },
  ];
  const router: Op[] = [
    {
      ...make(
        "flat",
        "flatten tokens",
        d,
        s,
        "Combine batch and sequence axes; no tokens are mixed.",
        "view(B×S,d)",
        [["hidden", shape("B", "S", d)]],
        [],
        "merged",
      ),
      col: 0,
      row: 0,
    },
    {
      ...make(
        "logits",
        "router linear",
        e,
        scores,
        "Produce a score for every expert from each token vector.",
        "logits = x W_gateᵀ",
        [["x", s]],
        [["gate.weight", shape(e, d)]],
      ),
      col: 1,
      row: 0,
    },
    {
      ...make(
        "scores",
        "FP32 softmax",
        e,
        scores,
        "Turn all expert logits into probabilities before selection.",
        "p_e = exp(l_e) / Σ_j exp(l_j)",
        [["logits", scores]],
      ),
      col: 2,
      row: 0,
    },
    {
      ...make(
        "selected",
        "top-k",
        k,
        choices,
        "Keep the largest probabilities and their expert ids.",
        `values, ids = topk(p, ${n(k)})`,
        [["p", scores]],
      ),
      col: 3,
      row: 0,
    },
    {
      ...make(
        "weights",
        "renormalize",
        k,
        choices,
        "Redistribute the retained mass over selected experts only.",
        "w_e = p_e / Σ_selected p_j",
        [["selected probabilities", choices]],
      ),
      col: 4,
      row: 0,
    },
    {
      ...make(
        "cast",
        "cast weights",
        k,
        choices,
        "Cast routing weights back to hidden-state dtype before expert multiplication.",
        `w = w.to(${f.dtype.value})`,
        [["w_fp32", choices]],
      ),
      col: 5,
      row: 0,
    },
  ];
  const dispatch: Op[] = [
    {
      ...make(
        "mask",
        "one-hot dispatch",
        e,
        shape(e, k, "N"),
        "Transpose a one-hot selection map to find tokens for each expert.",
        "one_hot(ids,E).permute(expert,slot,token)",
        [["ids", choices]],
      ),
      col: 0,
      row: 0,
    },
    {
      ...make(
        "gather",
        "gather tokens",
        d,
        shape("Ne", d),
        "Find token/slot indices for this expert and gather their input vectors.",
        "slot,token = where(mask[e]); x_e = x[token]",
        [
          ["mask", shape(e, k, "N")],
          ["x", s],
        ],
        [],
        "low",
      ),
      col: 1,
      row: 0,
    },
    {
      ...make(
        "evaluate",
        "expert MLP",
        d,
        shape("Ne", d),
        "Run the expert's own gated projections shown in the next graph.",
        "z_e = expert_e(x_e)",
        [["x_e", shape("Ne", d)]],
        [],
        "low",
      ),
      col: 2,
      row: 0,
    },
    {
      ...make(
        "weight",
        "multiply weight",
        d,
        shape("Ne", d),
        "Multiply the expert output by the token's normalized selected score.",
        "z_e *= w[token,slot]",
        [
          ["z_e", shape("Ne", d)],
          ["w", choices],
        ],
        [],
        "low",
      ),
      col: 3,
      row: 0,
    },
    {
      ...make(
        "scatter",
        "index add",
        d,
        s,
        "Accumulate contributions into a zero-initialized full token output.",
        "zeros(N,d).index_add(token,z_e)",
        [["weighted z_e", shape("Ne", d)]],
        [],
        "low",
      ),
      col: 4,
      row: 0,
    },
    {
      ...make(
        "restore",
        "restore B / S",
        d,
        shape("B", "S", d),
        "Separate batch and sequence axes again before the block's residual addition.",
        "view(B,S,d)",
        [["accumulated", s]],
        [],
        "merged",
      ),
      col: 5,
      row: 0,
    },
  ];
  const er = [
    { source: "reference", symbol: "Qwen3MoeMLP.forward" },
    { source: "serving", symbol: "Qwen3MoeMLP.forward" },
    { source: "serving", symbol: "Qwen3MoeSparseMoeBlock" },
  ];
  const expert: Op[] = [
    {
      ...make(
        "gate",
        "gate projection",
        i,
        inner,
        "Build the activation branch with this expert's gate matrix.",
        "g = x_e W_gate,eᵀ",
        [["x_e", shape("Ne", d)]],
        [["gate_proj.weight", shape(i, d)]],
        "low",
      ),
      col: 0,
      row: 0,
    },
    {
      ...make(
        "silu",
        "SiLU",
        i,
        inner,
        "Apply a smooth elementwise gate; it does not choose experts.",
        "a = g · sigmoid(g)",
        [["g", inner]],
        [],
        "low",
      ),
      col: 1,
      row: 0,
    },
    {
      ...make(
        "up",
        "up projection",
        i,
        inner,
        "Build the independent content branch from the same expert input.",
        "u = x_e W_up,eᵀ",
        [["x_e", shape("Ne", d)]],
        [["up_proj.weight", shape(i, d)]],
        "low",
      ),
      col: 0,
      row: 1,
    },
    {
      ...make(
        "mul",
        "gate × up",
        i,
        inner,
        "Multiply activation and content elementwise.",
        "z = a ⊙ u",
        [
          ["a", inner],
          ["u", inner],
        ],
        [],
        "low",
      ),
      kind: "mul",
      col: 2,
      row: 0,
    },
    {
      ...make(
        "down",
        "down projection",
        d,
        shape("Ne", d),
        "Return the selected expert's update to residual width.",
        "z_e = z W_down,eᵀ",
        [["z", inner]],
        [["down_proj.weight", shape(d, i)]],
        "low",
      ),
      col: 3,
      row: 0,
    },
  ];
  expert.forEach((o, index) => {
    expert[index] = { ...o, detail: { ...o.detail, refs: er } };
  });
  return (
    <Section
      id="experts"
      title="Experts: choose, execute, combine"
      source={refs}
      lede={
        <>
          Every token scores {n(e)} experts but executes only {n(k)}. The selected scores are normalized again; the
          other {n(f.inactive.value)} expert outputs never enter that token's sum.
        </>
      }
    >
      <OpGraph
        id="moeTopology"
        title="MoE topology"
        heading="Routing and data are separate paths"
        ops={topology}
        edges={topEdges}
      />
      <Prose>
        The router produces expert ids and mixture weights. Its linear gate is distinct from the SiLU gate inside each
        expert.
      </Prose>
      <OpGraph
        id="routerOp"
        title="Top-k routing arithmetic"
        heading="Choose the mixture"
        ops={router}
        edges={router.slice(1).map((o, j) => ({ from: router[j].id, to: o.id }))}
      />
      <Prose>
        The reference dispatches by expert. vLLM delegates this gather, gated computation and weighted accumulation to
        FusedMoE rather than running this Python loop.
      </Prose>
      <OpGraph
        id="dispatchOp"
        title="Expert token dispatch"
        heading="Dispatch and accumulate"
        ops={dispatch}
        edges={dispatch.slice(1).map((o, j) => ({ from: dispatch[j].id, to: o.id }))}
      />
      <Prose>
        Inside each expert, the input is read by parallel gate and up projections. Its intermediate width is the expert
        width, not the unused dense-MLP config field.
      </Prose>
      <OpGraph
        id="expertOp"
        title="One selected expert"
        heading="The gated expert MLP"
        ops={expert}
        edges={[
          { from: "gate", to: "silu" },
          { from: "silu", to: "mul" },
          { from: "up", to: "mul" },
          { from: "mul", to: "down" },
        ]}
      />
      <div className="ms-control">
        <Button onClick={() => setSeed((v) => (v + 1) % (f.MAX_SEED + 1))}>Route another token</Button>
      </div>
      <Sentence>
        Random router weights from seed {n(seed)} send a toy-width token to experts {routing.chosen.map(n).join(", ")}.
        Its selected weights sum to {routing.weights.reduce((a, b) => a + b, 0).toFixed(f.DECIMALS)}.
      </Sentence>
      <Figure
        label="Computed checkpoint-size expert selection"
        caption={`Computed using the clean reference router: random weights and a synthetic ${n(f.TOY.hidden)}-channel input, with the checkpoint's ${n(e)} experts and top-${n(k)} rule. Weights below follow descending score order; the selected-id list is sorted by id. This is not a trained-model route.`}
      >
        <MoeRouter
          title="Normalized softmax top-k routing"
          routing={routing}
          experts={e}
          shared={f.sharedExperts.value}
        />
      </Figure>
      <Readout
        label="Sparsity readout"
        items={[
          { term: "Executed expert fraction", value: `${(f.expertFraction.value * f.PERCENT).toFixed(f.DECIMALS)}%` },
          { term: "Per expert parameters", value: n(f.expertParams.value) },
          { term: "Intermediate width", value: n(i), note: `residual ${n(d)}` },
        ]}
      />
      <Check question="Does renormalizing selected probabilities change which experts were selected?">
        No. Selection has already happened. Dividing every retained score by the same positive sum changes only their
        mixture weights, which now sum to one.
      </Check>
      <Takeaways
        items={[
          "The router evaluates every expert score; sparsity applies to expert execution, not score computation.",
          <>
            A token executes {(f.expertFraction.value * f.PERCENT).toFixed(f.DECIMALS)}% of the expert bank while the
            full bank remains learned parameters.
          </>,
          "This checkpoint has no shared experts, group filter, router bias correction or extra mixture scaling.",
        ]}
      />
    </Section>
  );
}
