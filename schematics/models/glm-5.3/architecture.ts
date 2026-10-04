import { shape, n, type Op, type Edge } from "../../kit";
import * as f from "./facts";
type Spec = {
  id: string;
  label: string;
  width: number;
  what: string;
  formula: string;
  output?: string;
  input?: string;
  inputs?: [string, string][];
  params?: [string, string][];
  note?: string;
  kind?: Op["kind"];
  lane?: Op["lane"];
};
export function graph(
  specs: Spec[],
  symbol: string,
  servingSymbol = "DeepseekV32Attention.forward",
): { ops: Op[]; edges: Edge[] } {
  const ops = specs.map((s, col): Op => ({
    id: s.id,
    label: s.label,
    sub: n(s.width),
    width: s.width,
    col,
    row: 0,
    kind: s.kind,
    lane: s.lane ?? (symbol.includes("MLP") ? "low" : symbol.includes("TopkRouter") ? "instance" : "high"),
    detail: {
      what: s.what,
      formula: s.formula,
      inputs: s.inputs ?? [
        [
          "input",
          s.input ??
            (col === 0
              ? shape("B", "S", f.hidden.value)
              : (specs[col - 1].output ?? shape("B", "S", specs[col - 1].width))),
        ],
      ],
      output: [s.id, s.output ?? shape("B", "S", s.width)],
      params: s.params ?? [],
      refs: [
        { source: "reference", symbol },
        ...(symbol.includes("Attention") || symbol.includes("Indexer")
          ? [{ source: "serving-attention", symbol: servingSymbol }]
          : []),
      ],
      note: s.note,
    },
  }));
  let edges: Edge[] = ops.slice(1).map((o, i) => ({ from: ops[i].id, to: o.id }));
  const wire = (from: string, to: string): Edge => ({ from, to });
  const positions = (places: Record<string, [number, number]>) =>
    ops.forEach((o, i) => {
      const at = places[o.id];
      if (at) ops[i] = { ...o, col: at[0], row: at[1] };
    });
  if (ops.some((o) => o.id === "gather")) {
    edges.push(wire("sigmoid", "gather"));
    // Original scores bypass choice correction; the inspector spells out both inputs.
    const gather = ops.find((o) => o.id === "gather")!;
    ops[ops.indexOf(gather)] = {
      ...gather,
      detail: {
        ...gather.detail,
        inputs: [
          ["scores", shape("B", "S", f.experts.value)],
          ["indices", shape("B", "S", f.topK.value)],
        ],
      },
    };
  }
  if (ops.some((o) => o.id === "kvsplit")) {
    positions({ kv_a_proj_with_mqa: [0, 0], kvsplit: [1, 0], kvnorm: [2, 0], krope: [2, 1], cache: [3, 0] });
    edges = [
      wire("kv_a_proj_with_mqa", "kvsplit"),
      wire("kvsplit", "kvnorm"),
      wire("kvsplit", "krope"),
      wire("kvnorm", "cache"),
      wire("krope", "cache"),
    ];
  }
  if (ops.some((o) => o.id === "indexqsplit")) {
    positions({ wq_b: [0, 0], indexqsplit: [1, 0], wk: [0, 1], knorm: [1, 1], indexkrot: [2, 1], indexcache: [3, 1] });
    edges = [
      wire("wq_b", "indexqsplit"),
      wire("wk", "knorm"),
      wire("knorm", "indexkrot"),
      wire("indexkrot", "indexcache"),
    ];
  }
  if (ops.some((o) => o.id === "sumheads")) {
    positions({
      indexdot: [0, 0],
      relu: [1, 0],
      weights_proj: [1, 1],
      sumheads: [2, 0],
      indexcausal: [3, 0],
      topindices: [4, 0],
    });
    edges = [
      wire("indexdot", "relu"),
      wire("relu", "sumheads"),
      wire("weights_proj", "sumheads"),
      wire("sumheads", "indexcausal"),
      wire("indexcausal", "topindices"),
    ];
  }
  if (ops.some((o) => o.id === "multiply")) {
    positions({ gate_proj: [0, 0], silu: [1, 0], up_proj: [0, 1], multiply: [2, 0], down_proj: [3, 0] });
    edges = [
      wire("gate_proj", "silu"),
      wire("silu", "multiply"),
      wire("up_proj", "multiply"),
      wire("multiply", "down_proj"),
    ];
  }
  const inputs: Record<string, [string, string][]> = {
    qrope: [
      ["q_rot", shape("B", heads, "S", r)],
      ["cos/sin", shape("B", "S", r)],
    ],
    qconcat: [
      ["q_content", shape("B", heads, "S", nd)],
      ["q_rotated", shape("B", heads, "S", r)],
    ],
    krope: [
      ["k_rope", shape("B", 1, "S", r)],
      ["cos/sin", shape("B", "S", r)],
    ],
    cache: [
      ["normalized latent", shape("B", 1, "S", kv)],
      ["rotated key", shape("B", 1, "S", r)],
      ["prior cache", shape("B", 1, "T−S", kv + r)],
    ],
    key: [
      ["content key", shape("B", heads, "T", nd)],
      ["rotated key", shape("B", 1, "T", r)],
    ],
    scores: [
      ["query", shape("B", heads, "S", nd + r)],
      ["key", shape("B", heads, "T", nd + r)],
    ],
    mask: [
      ["scores", shape("B", heads, "S", "T")],
      ["indices", shape("B", "S", "min(K,T)")],
      ["causal mask", shape("B", 1, "S", "T")],
    ],
    weighted: [
      ["attention", shape("B", heads, "S", "T")],
      ["values", shape("B", heads, "T", vd)],
    ],
    indexdot: [
      ["index query", shape("B", "S", ih, ix)],
      ["index keys", shape("B", "T", ix)],
    ],
    indexqsplit: [
      ["q", shape("B", "S", ih * ix)],
      ["cos/sin", shape("B", "S", r)],
    ],
    sumheads: [
      ["ReLU similarities", shape("B", "S", ih, "T")],
      ["head weights", shape("B", "S", ih)],
    ],
    indexcausal: [
      ["index scores", shape("B", "S", "T")],
      ["causal mask", shape("B", "S", "T")],
    ],
    correction: [
      ["sigmoid scores", shape("B", "S", f.experts.value)],
      ["correction buffer", shape(f.experts.value)],
    ],
    multiply: [
      ["SiLU gate", shape("B", "S", specs.find((o) => o.id === "silu")?.width ?? h)],
      ["up channels", shape("B", "S", specs.find((o) => o.id === "up_proj")?.width ?? h)],
    ],
  };
  ops.forEach((o, i) => {
    if (inputs[o.id]) ops[i] = { ...o, detail: { ...o.detail, inputs: inputs[o.id] } };
  });
  return { ops, edges };
}
const h = f.hidden.value,
  q = f.qRank.value,
  kv = f.kvRank.value,
  r = f.rope.value,
  heads = f.heads.value,
  nd = f.nope.value,
  vd = f.valueDim.value,
  ih = f.indexHeads.value,
  ix = f.indexDim.value;
const projection = (id: string, label: string, input: number, width: number, what: string): Spec => ({
  id,
  label,
  width,
  what,
  formula: `${id} = input @ ${id}.weightᵀ`,
  params: [[id + ".weight", shape(width, input)]],
  input: shape("B", "S", input),
});
export function blockGraph(layer: number) {
  const sh = shape("B", "S", h),
    ops: Op[] = [
      {
        id: "stream",
        label: "hidden",
        kind: "tensor",
        width: h,
        sub: n(h),
        col: 0,
        row: 0,
        detail: {
          what: `Residual stream entering layer ${layer}.`,
          formula: "x = previous layer output",
          output: ["x", sh],
          params: [],
          refs: [{ source: "reference", symbol: "GlmMoeDsaDecoderLayer.forward" }],
        },
      },
      ...[
        "input RMSNorm",
        "latent attention",
        "add attention",
        "post RMSNorm",
        f.mlpTypes.value[layer] === "dense" ? "dense SiLU MLP" : "routed + shared",
        "add MLP",
      ].map((label, i): Op => ({
        id: `block${i}`,
        label,
        sub: n(h),
        width: h,
        col: i + 1,
        row: 0,
        kind: label.startsWith("add") ? "sum" : "op",
        lane: i === 4 ? "low" : "merged",
        detail: {
          what: label.startsWith("add")
            ? "Adds the branch correction to its saved residual."
            : `Transforms a copy of the stream: ${label}.`,
          formula: label.startsWith("add")
            ? "y = residual + branch"
            : label.includes("RMS")
              ? "y = x / sqrt(mean(x²)+eps) · gain"
              : "y = sublayer(normalized stream)",
          inputs: [["x", sh]],
          output: ["y", sh],
          params: label.includes("RMS") ? [["gain", shape(h)]] : [],
          refs: [
            { source: "reference", symbol: "GlmMoeDsaDecoderLayer.forward" },
            { source: "serving", symbol: "DeepseekV32DecoderLayer.forward" },
          ],
          note: `Attention selection belongs to layer ${f.selectionOwner(layer)}; this layer has fresh attention projections and cache.`,
        },
      })),
    ];
  const edges: Edge[] = ops.slice(1).map((o, i) => ({ from: ops[i].id, to: o.id }));
  edges.push({ from: "stream", to: "block2", kind: "skip" }, { from: "block2", to: "block5", kind: "skip" });
  return { ops, edges };
}
export const queryGraph = () =>
  graph(
    [
      projection("q_a_proj", "q compress", h, q, "Compresses the normalized stream into a query latent."),
      {
        id: "qnorm",
        label: "q RMSNorm",
        width: q,
        what: "Normalizes the query latent before expansion.",
        formula: "q̂ = q / sqrt(mean(q²)+eps) · gain",
        params: [["q_a_layernorm.weight", shape(q)]],
        note: `Reference epsilon ${f.latentEps.value}; serving epsilon ${f.eps.value}.`,
      },
      projection(
        "q_b_proj",
        "q expand",
        q,
        f.qWidth.value,
        "Creates independent content and positional query channels per head.",
      ),
      {
        id: "qsplit",
        label: "split heads",
        width: nd + r,
        what: "Introduces the head axis and separates content from positional channels.",
        formula: "q → [content, rotary]",
        output: shape("B", heads, "S", nd + r),
      },
      {
        id: "qrope",
        label: "rotate q slice",
        width: r,
        what: "Rotates interleaved positional pairs; content channels pass untouched.",
        formula: "(u,v) → (u cosθ − v sinθ, v cosθ + u sinθ)",
        output: shape("B", heads, "S", r),
        note: "Reference stores rotated even and odd channels in separate halves.",
      },
      {
        id: "qconcat",
        label: "join query",
        width: nd + r,
        what: "Concatenates unchanged content and rotated positional query.",
        formula: "q = concat(q_content,q_rotated)",
        output: shape("B", heads, "S", nd + r),
      },
    ],
    "GlmMoeDsaAttention.forward",
  );
export const kvGraph = () =>
  graph(
    [
      projection("kv_a_proj_with_mqa", "KV compress", h, kv + r, "Produces a shared KV latent and a positional key."),
      {
        id: "kvsplit",
        label: "split latent / RoPE",
        width: kv,
        what: "Separates compressed content from the positional key.",
        formula: "compressed → latent, key_rope",
        output: shape("B", "S", kv),
        note: `A separate ${r}-wide positional key is retained.`,
      },
      {
        id: "kvnorm",
        label: "latent RMSNorm",
        width: kv,
        what: "Normalizes content only; the positional key bypasses this norm.",
        formula: "ĉ = c / sqrt(mean(c²)+eps) · gain",
        params: [["kv_a_layernorm.weight", shape(kv)]],
      },
      {
        id: "krope",
        label: "rotate key slice",
        width: r,
        what: "Rotates the shared positional key using the same angles as query.",
        formula: "k_rot = rotate(k_rope,position)",
        output: shape("B", 1, "S", r),
      },
      {
        id: "cache",
        label: "latent cache",
        kind: "state",
        lane: "instance",
        width: kv + r,
        what: "Hands normalized content and the rotated key to the cache as two separate tensors, then reads back all previous tokens.",
        formula: "ĉ_all, k_rot_all = cache.update(ĉ, k_rot, layer)",
        inputs: [
          ["ĉ", shape("B", 1, "S", kv)],
          ["k_rot", shape("B", 1, "S", r)],
        ],
        output: `${shape("B", 1, "T", kv)} and ${shape("B", 1, "T", r)}`,
        note: "Prefill reference experiment disables caching; decode keeps these tensors before expansion. The cache class is not pinned, so its storage layout is not shown.",
      },
    ],
    "GlmMoeDsaAttention.forward",
  );
export const expandGraph = () =>
  graph(
    [
      {
        ...projection(
          "kv_b_proj",
          "KV expand",
          kv,
          heads * (nd + vd),
          "Expands latent content to a key and value for each attention head.",
        ),
        input: shape("B", 1, "T", kv),
        output: shape("B", 1, "T", heads * (nd + vd)),
        note: "Runs after the cache read, so the token axis is T (all cached tokens) and the latent is still single-head.",
      },
      {
        id: "kvheads",
        label: "split K / V",
        width: nd + vd,
        what: "Splits each head into non-positional key and value channels.",
        formula: "expanded → k_content, v",
        output: shape("B", heads, "T", nd + vd),
      },
      {
        id: "key",
        label: "broadcast + join K",
        width: nd + r,
        what: "Broadcasts the cached positional key across heads and joins it to content key.",
        formula: "k = concat(k_content,broadcast(k_rot))",
        output: shape("B", heads, "T", nd + r),
        note: "vLLM latent decode absorbs this expansion into query and output weights.",
      },
    ],
    "GlmMoeDsaAttention.expand_kv",
  );
export const attendGraph = () =>
  graph(
    [
      {
        id: "scores",
        label: "q · kᵀ / √d",
        width: f.defaultContext.value,
        what: "Compares each query with each key within its head.",
        formula: `scores = q @ kᵀ / sqrt(${nd + r})`,
        output: shape("B", heads, "S", "T"),
      },
      {
        id: "mask",
        label: "causal ∩ selected",
        width: f.defaultContext.value,
        lane: "instance",
        what: "Masks future tokens and any key absent from the selected index set.",
        formula: "scores[future or unselected] = −∞",
        output: shape("B", heads, "S", "T"),
        note: "Eager path materializes an exclusion mask; sparse serving consumes indices directly.",
      },
      {
        id: "softmax",
        label: "softmax",
        width: f.defaultContext.value,
        what: "Normalizes remaining scores over key positions.",
        formula: "a = exp(scores − max) / sum exp(scores − max)",
        output: shape("B", heads, "S", "T"),
      },
      {
        id: "weighted",
        label: "weights · values",
        width: vd,
        what: "Forms a weighted value vector per head.",
        formula: "z = a @ v",
        output: shape("B", heads, "S", vd),
      },
      {
        id: "merge",
        label: "merge heads",
        width: f.vWidth.value,
        what: "Moves the head axis next to the value axis and flattens both.",
        formula: "z → [B,S,heads × value_dim]",
      },
      projection("o_proj", "output project", f.vWidth.value, h, "Returns the attention correction to stream width."),
    ],
    "GlmMoeDsaAttention.forward",
  );
export const indexQGraph = () =>
  graph(
    [
      projection("wq_b", "index q", q, ih * ix, "Reads the normalized query latent to make lightweight search heads."),
      {
        id: "indexqsplit",
        label: "split + rotate q",
        width: ix,
        what: "Rotates only the leading positional slice of each search head.",
        formula: "q_i = concat(rotate(q_rot),q_pass)",
        output: shape("B", "S", ih, ix),
      },
      projection("wk", "index key", h, ix, "Builds one search key per token, shared across search heads."),
      {
        id: "knorm",
        label: "key LayerNorm",
        width: ix,
        what: "Centers and normalizes each index key with learned gain and bias.",
        formula: "k̂ = (k−mean)/sqrt(var+eps)·g+b",
        params: [
          ["k_norm.weight", shape(ix)],
          ["k_norm.bias", shape(ix)],
        ],
      },
      {
        id: "indexkrot",
        label: "split + rotate k",
        width: ix,
        what: "Rotates the positional slice and rejoins unchanged channels.",
        formula: "k_i = concat(rotate(k_rot),k_pass)",
      },
      {
        id: "indexcache",
        label: "index key cache",
        kind: "state",
        lane: "instance",
        width: ix,
        what: "Keeps one index key per token only in full-indexer layers.",
        formula: "index_cache.append(k_i)",
        output: shape("B", "T", ix),
      },
    ],
    "GlmMoeDsaIndexer.forward",
  );
export const indexScoreGraph = () =>
  graph(
    [
      {
        id: "indexdot",
        label: "search dot product",
        width: f.defaultContext.value,
        what: "Computes scaled query-key similarity for each search head.",
        formula: `s_h = q_h · k / sqrt(${ix})`,
        output: shape("B", "S", ih, "T"),
      },
      {
        id: "relu",
        label: "ReLU",
        width: f.defaultContext.value,
        what: "Discards negative per-head similarities before head aggregation.",
        formula: "s_h = max(s_h,0)",
        output: shape("B", "S", ih, "T"),
      },
      projection("weights_proj", "head weights", h, ih, "Produces signed input-dependent weights for search heads."),
      {
        id: "sumheads",
        label: "weighted head sum",
        width: f.defaultContext.value,
        what: "Combines head similarities using scaled learned head weights.",
        formula: `score = Σ_h s_h · w_h / sqrt(${ih})`,
        output: shape("B", "S", "T"),
      },
      {
        id: "indexcausal",
        label: "causal mask",
        width: f.defaultContext.value,
        lane: "instance",
        what: "Excludes future keys before selecting positions.",
        formula: "score[future] = −∞",
        output: shape("B", "S", "T"),
      },
      {
        id: "topindices",
        label: "token top-k",
        width: f.indexTopK.value,
        lane: "instance",
        what: "Selects key positions; no attention probability is computed here.",
        formula: `indices = topk(score,min(${f.indexTopK.value},T))`,
        output: shape("B", "S", "min(K,T)"),
        note: "Shared layers reuse these indices but recompute their attention probabilities.",
      },
    ],
    "GlmMoeDsaIndexer.forward",
  );
export const routerGraph = () =>
  graph(
    [
      projection("gate", "router linear", h, f.experts.value, "Scores routed experts independently for each token."),
      {
        id: "sigmoid",
        label: "sigmoid",
        width: f.experts.value,
        what: "Maps logits to positive expert affinity scores.",
        formula: "score = 1 / (1 + exp(−logit))",
      },
      {
        id: "correction",
        label: "choice correction",
        width: f.experts.value,
        lane: "instance",
        what: "Adds the stored correction bias for selecting experts only.",
        formula: "choice = score + correction_bias",
        params: [],
        note: "The correction is a buffer, not a trainable parameter; it never modifies mixture weights.",
      },
      {
        id: "group",
        label: "group selection",
        width: f.experts.value,
        what: "Computes best-pair group scores and excludes unselected groups.",
        formula: "group_score = sum(top2(choice))",
        note: "The checkpoint has a single group; this step excludes no expert.",
      },
      {
        id: "expert_topk",
        label: "expert top-k",
        width: f.topK.value,
        what: "Chooses routed expert identities using corrected scores.",
        formula: `selected = topk(choice,${f.topK.value})`,
      },
      {
        id: "gather",
        label: "gather raw scores",
        width: f.topK.value,
        what: "Looks up uncorrected sigmoid values at the selected identities.",
        formula: "weight = gather(score,selected)",
      },
      {
        id: "normalize",
        label: "normalize + scale",
        width: f.topK.value,
        what: "Renormalizes selected affinities then scales the routed mixture.",
        formula: `weight = ${f.scale.value} · weight / (sum(weight)+ε)`,
        note: "The always-on shared expert bypasses all router operations.",
      },
    ],
    "GlmMoeDsaTopkRouter.forward",
    "DeepseekV32Attention.forward",
  );
export function expertGraph(layer: number) {
  const width = f.mlpTypes.value[layer] === "dense" ? f.denseWidth.value : f.expertWidth.value;
  const result = graph(
    [
      projection(
        "gate_proj",
        "gate projection",
        h,
        width,
        "Creates the gate channels independently of the up projection.",
      ),
      {
        id: "silu",
        label: "SiLU",
        width,
        what: "Smoothly gates each intermediate channel.",
        formula: "gate = x / (1 + exp(−x))",
      },
      projection("up_proj", "up projection", h, width, "Creates the value channels from the same sublayer input."),
      {
        id: "multiply",
        label: "gate × up",
        kind: "mul",
        width,
        what: "Multiplies gate and up channels elementwise.",
        formula: "intermediate = SiLU(gate_proj(x)) × up_proj(x)",
      },
      projection("down_proj", "down projection", width, h, "Returns an expert or dense MLP output to residual width."),
    ],
    f.mlpTypes.value[layer] === "dense" ? "GlmMoeDsaMLP.forward" : "GlmMoeDsaExperts.forward",
  );
  if (f.mlpTypes.value[layer] !== "dense") {
    result.ops = result.ops.map((op) => {
      if (op.id === "gate_proj" || op.id === "up_proj") {
        return {
          ...op,
          detail: {
            ...op.detail,
            params: [["experts.gate_up_proj", shape(f.experts.value, 2 * width, h)]],
            note: "The selected expert uses its gate or up slice of the packed parameter; these are independent projections of the same input.",
          },
        };
      }
      if (op.id === "down_proj") {
        return { ...op, detail: { ...op.detail, params: [["experts.down_proj", shape(f.experts.value, h, width)]] } };
      }
      return op;
    });
  }
  return result;
}

export function attentionTopology() {
  const g = graph(
    [
      {
        id: "x",
        label: "normalized stream",
        width: h,
        kind: "tensor",
        what: "The common input to independent projections.",
        formula: "x = input_RMSNorm(residual)",
      },
      {
        id: "qpath",
        label: "query path",
        width: heads * (nd + r),
        what: "Compress, normalize, expand and rotate queries.",
        formula: "q = query_path(x)",
      },
      {
        id: "kvpath",
        label: "cached KV path",
        width: kv + r,
        what: "Compress content and position; normalize content, rotate key, then cache.",
        formula: "cache = KV_path(x)",
        input: shape("B", "S", h),
      },
      {
        id: "selection",
        label: "search / reuse",
        width: f.indexTopK.value,
        lane: "instance",
        what: "Select key identities or read prior selection.",
        formula: "indices = indexer(x,q_latent) or previous_indices",
        inputs: [
          ["x", shape("B", "S", h)],
          ["q_latent", shape("B", "S", f.qRank.value)],
        ],
        output: shape("B", "S", `min(${f.indexTopK.value},T)`),
      },
      {
        id: "join",
        label: "sparse attention",
        width: heads * vd,
        what: "Read selected causal keys and form weighted values.",
        formula: "z = softmax(mask(qkᵀ))v",
        inputs: [
          ["q", shape("B", heads, "S", nd + r)],
          ["K", shape("B", heads, "T", nd + r)],
          ["V", shape("B", heads, "T", vd)],
          ["mask", shape("B", 1, "S", "T")],
        ],
      },
      {
        id: "project",
        label: "output project",
        width: h,
        what: "Project the merged heads back to stream width.",
        formula: "output = z @ o_proj.weightᵀ",
        params: [["o_proj.weight", shape(h, heads * vd)]],
      },
    ],
    "GlmMoeDsaAttention.forward",
  );
  const positions: Record<string, [number, number]> = {
    x: [0, 0],
    qpath: [1, 0],
    kvpath: [1, 1],
    selection: [2, 1],
    join: [3, 0],
    project: [4, 0],
  };
  g.ops = g.ops.map((o) => ({ ...o, col: positions[o.id][0], row: positions[o.id][1] }));
  g.edges = [
    { from: "x", to: "qpath" },
    { from: "x", to: "kvpath" },
    { from: "x", to: "selection" },
    { from: "qpath", to: "selection" },
    { from: "qpath", to: "join" },
    { from: "kvpath", to: "join" },
    { from: "selection", to: "join", kind: "state" },
    { from: "join", to: "project" },
  ];
  return g;
}

export function moeGraph() {
  const g = graph(
    [
      {
        id: "input",
        label: "normalized token",
        width: h,
        kind: "tensor",
        what: "Common input to routed and shared experts.",
        formula: "x = post_attention_RMSNorm(stream)",
      },
      {
        id: "dispatch",
        label: "dispatch chosen",
        width: h,
        what: "Gather token rows for each selected expert.",
        formula: "x_e = gather(x,token_indices_for_expert)",
        output: shape("selected tokens", h),
      },
      {
        id: "routedffn",
        label: "selected FFNs",
        width: h,
        lane: "low",
        what: "Each selected expert applies its own packed gate/up and down weights.",
        formula: "y_e = down_e(SiLU(gate_e(x_e)) × up_e(x_e))",
        output: shape("selected tokens", h),
        note: "Rows are the tokens routed to one expert, not the batch and sequence axes.",
        params: [
          ["experts.gate_up_proj", shape(f.experts.value, 2 * f.expertWidth.value, h)],
          ["experts.down_proj", shape(f.experts.value, h, f.expertWidth.value)],
        ],
      },
      {
        id: "mixture",
        label: "weight outputs",
        width: h,
        lane: "low",
        what: "Multiply each selected expert output by its router weight.",
        formula: "weighted_e = y_e × router_weight_e",
        output: shape("selected tokens", h),
        note: "Still one row per selected token; index_add_ restores token positions in the next step.",
      },
      {
        id: "scatter",
        label: "scatter-add",
        width: h,
        lane: "low",
        what: "Add weighted expert outputs back at original token positions.",
        formula: "routed[token] += weighted_e",
      },
      {
        id: "sharedffn",
        label: "shared FFN",
        width: h,
        lane: "low",
        what: "Run the always-on shared expert from the common input without routing.",
        formula: "shared = down(SiLU(gate(x)) × up(x))",
        input: shape("B", "S", h),
        params: [
          ["shared gate/up", shape(f.expertWidth.value * f.shared.value, h)],
          ["shared down", shape(h, f.expertWidth.value * f.shared.value)],
        ],
      },
      {
        id: "joinexperts",
        label: "add shared",
        width: h,
        kind: "sum",
        lane: "low",
        what: "Add shared and routed outputs before the outer residual addition.",
        formula: "mlp_output = routed + shared",
      },
    ],
    "GlmMoeDsaMoE.forward",
  );
  const positions: Record<string, [number, number]> = {
    input: [0, 0],
    dispatch: [1, 0],
    routedffn: [2, 0],
    mixture: [3, 0],
    scatter: [4, 0],
    sharedffn: [2, 1],
    joinexperts: [5, 0],
  };
  g.ops = g.ops.map((o) => ({
    ...o,
    col: positions[o.id][0],
    row: positions[o.id][1],
    detail: {
      ...o.detail,
      refs: [
        {
          source: "reference",
          symbol:
            o.id === "sharedffn"
              ? "GlmMoeDsaMLP.forward"
              : o.id === "joinexperts" || o.id === "input"
                ? "GlmMoeDsaMoE.forward"
                : "GlmMoeDsaExperts.forward",
        },
      ],
    },
  }));
  g.edges = [
    { from: "input", to: "dispatch" },
    { from: "dispatch", to: "routedffn" },
    { from: "routedffn", to: "mixture" },
    { from: "mixture", to: "scatter" },
    { from: "input", to: "sharedffn" },
    { from: "scatter", to: "joinexperts" },
    { from: "sharedffn", to: "joinexperts" },
  ];
  return g;
}
