import { shape, type Op, type Edge, type Lane } from "../../kit";
import * as f from "./facts";
const d = f.hidden.value,
  h = f.headDim.value,
  q = f.qRank.value,
  ih = f.indexHeads.value,
  id = f.indexDim.value;
export const stream = shape("B", "S", d),
  residual = shape("B", "S", f.copies.value, d),
  latent = shape("B", "S", h),
  query = shape("B", "S", f.heads.value, h);
type Spec = {
  id: string;
  label: string;
  width: number;
  output: string;
  what: string;
  formula: string;
  params?: readonly (readonly [string, string])[];
  note?: string;
  col?: number;
  row?: number;
  kind?: Op["kind"];
  lane?: Lane;
  input?: string;
};
export function graph(
  specs: readonly Spec[],
  symbol: string,
  serving = "MQALayer",
  source = "reference",
): { ops: Op[]; edges: Edge[] } {
  const ops: Op[] = specs.map((s, i) => ({
    id: s.id,
    label: s.label,
    sub: s.output,
    width: s.width,
    col: s.col ?? i,
    row: s.row ?? 0,
    kind: s.kind,
    lane:
      s.lane ??
      (/Gate|Expert|MoE/.test(symbol)
        ? "low"
        : /Engram/.test(symbol)
          ? "instance"
          : /hc_|mix/.test(symbol)
            ? "skip"
            : "high"),
    detail: {
      what: s.what,
      formula: s.formula,
      inputs: [[i ? "previous" : "input", s.input ?? (i ? specs[i - 1].output : stream)]],
      output: [s.id, s.output],
      params: s.params ?? [["parameters", "none"]],
      refs: [{ source, symbol }, ...(source === "reference" ? [{ source: "serving", symbol: serving }] : [])],
      note: s.note,
    },
  }));
  return { ops, edges: ops.slice(1).map((op, i) => ({ from: ops[i].id, to: op.id })) };
}
const projection = (name: string, out: number, input: number) => [[name, shape(out, input)]] as const;
const norm = (name: string, width: number) => [[name, shape(width)]] as const;
export function queryGraph() {
  return graph(
    [
      {
        id: "wqa",
        label: "wq_a",
        width: q,
        output: shape("B", "S", q),
        what: "Compress the attention input into the query latent.",
        formula: "qr = x Wq_aᵀ",
        params: projection("wq_a.weight", q, d),
      },
      {
        id: "qnorm",
        label: "RMSNorm",
        width: q,
        output: shape("B", "S", q),
        what: "Normalize the latent before both attention and indexer read it.",
        formula: "qr = qr · rsqrt(mean(qr²)+eps) · gain",
        params: norm("q_norm.weight", q),
      },
      {
        id: "wqb",
        label: "wq_b + heads",
        width: f.qWidth.value,
        output: query,
        what: "Expand the normalized query latent and introduce the head axis.",
        formula: "q = reshape(qr Wq_bᵀ)",
        params: projection("wq_b.weight", f.qWidth.value, q),
      },
      {
        id: "qrope",
        label: "tail RoPE",
        width: f.qWidth.value,
        output: query,
        what: "Rotate only the tail channels; compressed layers use YaRN and compressed base even for the window path.",
        formula: "q_tail = q_tail · exp(i position frequency)",
        note: `Rotary tail ${f.ropeDim.value}; no per-head query RMSNorm in the publisher reference.`,
      },
    ],
    "Attention.forward",
  );
}
export function windowGraph() {
  const g = graph(
    [
      {
        id: "wkv",
        label: "wkv",
        width: h,
        output: latent,
        what: "Build a single KV vector shared by every query head.",
        formula: "kv = x Wkvᵀ",
        params: projection("wkv.weight", h, d),
      },
      {
        id: "kvnorm",
        label: "RMSNorm",
        width: h,
        output: latent,
        what: "Normalize each shared KV vector before positional rotation.",
        formula: "kv = kv · rsqrt(mean(kv²)+eps) · gain",
        params: norm("kv_norm.weight", h),
      },
      {
        id: "kvrope",
        label: "tail RoPE",
        width: h,
        output: latent,
        what: "Apply the same layer-specific frequencies as the query.",
        formula: "kv_tail = kv_tail · exp(i position frequency)",
      },
      {
        id: "kvquant",
        label: "FP8 round",
        width: h,
        output: latent,
        what: "Quantize and dequantize the entire rotated vector, including its tail.",
        formula: "kv = dequant(quant_fp8(kv))",
        note: "Publisher retains fake-quantized values in an activation tensor; this is not a byte-packed cache-size claim.",
      },
      {
        id: "ring",
        label: "window ring",
        width: h,
        output: shape("B", f.window.value, h),
        kind: "state",
        lane: "instance",
        what: "Write the current token into the ring; prefill seeds its trailing window.",
        formula: "slot = position mod window",
        note: "During prefill attention reads the full current chunk with causal window indices.",
      },
    ],
    "Attention._window_kv",
  );
  g.ops.push(
    ...graph(
      [
        {
          id: "windowidx",
          label: "causal window indices",
          width: f.window.value,
          output: shape("B", "S", "min(S, W)"),
          input: shape("S"),
          what: "Generate positions within the causal window; use ring slots during decode, and mark unavailable slots negative.",
          formula: "max(0,position−window+1) ≤ key_position ≤ position",
          col: 3,
          row: 1,
          kind: "state",
          lane: "instance",
        },
      ],
      "get_window_topk_idxs",
    ).ops,
  );
  return g;
}
export function compressionGraph(i: number) {
  const r = f.ratios.value[i] || 2;
  const g = graph(
    [
      {
        id: "compresskv",
        label: "KV projection",
        width: h,
        output: shape("B", "S", h),
        what: "Project the source layer input before pooling.",
        formula: "v = x Wkvᵀ",
        params: projection("compressor.wkv.weight", h, d),
      },
      ...(r > 1
        ? [
            {
              id: "compressgate",
              label: "gate projection",
              width: h,
              output: shape("B", "S", h),
              what: "Compute per-channel token scores, independently of the KV values.",
              formula: "g = x Wgateᵀ",
              params: projection("compressor.wgate.weight", h, d),
              input: stream,
            },
            {
              id: "group",
              label: "group + softmax",
              width: h,
              output: shape("B", `⌊S/${r}⌋`, r, h),
              what: "Group consecutive tokens and normalize gate scores over the token axis for each channel.",
              formula: "a = softmax(g, group-token axis)",
              note: "Incomplete groups wait in KV/score state; they cannot be attended to yet.",
            },
            {
              id: "pool",
              label: "weighted pool",
              width: h,
              output: shape("B", `⌊S/${r}⌋`, h),
              what: "Sum gated KV values to produce one vector per complete group.",
              formula: "latent = Σgroup a · v",
            },
          ]
        : []),
      {
        id: "cnorm",
        label: "RMSNorm",
        width: h,
        output: shape("B", `⌊S/${r}⌋`, h),
        what: "Normalize the compressed latent before index-key projection and RoPE.",
        formula: "latent = RMSNorm(latent)",
        params: norm("compressor.norm.weight", h),
      },
    ],
    "Compressor",
    "MQALayer",
  );
  if (r > 1) {
    const columns = { compresskv: 0, compressgate: 0, group: 1, pool: 2, cnorm: 3 };
    g.ops = g.ops.map((op) => ({
      ...op,
      col: columns[op.id as keyof typeof columns],
      row: ["compressgate", "group"].includes(op.id) ? 1 : 0,
    }));
    g.edges = [
      { from: "compressgate", to: "group" },
      { from: "group", to: "pool" },
      { from: "compresskv", to: "pool" },
      { from: "pool", to: "cnorm" },
    ];
  }
  return g;
}
export function indexQueryGraph() {
  return graph(
    [
      {
        id: "iq",
        label: "index wq_b",
        width: ih * id,
        output: shape("B", "S", ih, id),
        what: "Reuse normalized query latent to make narrower index heads.",
        formula: "iq = reshape(qr Windex_qᵀ)",
        params: projection("indexer.wq_b.weight", ih * id, q),
        input: shape("B", "S", q),
      },
      {
        id: "iqrope",
        label: "tail RoPE",
        width: ih * id,
        output: shape("B", "S", ih, id),
        what: "Rotate the index-query tail at the current token position.",
        formula: "iq_tail = RoPE(iq_tail)",
      },
      {
        id: "iqquant",
        label: "FP4 round",
        width: ih * id,
        output: shape("B", "S", ih, id),
        what: "Round index queries using the indexer’s quantization block size.",
        formula: "iq = fake_fp4(iq)",
        note: `Groups of ${f.quantBlock.value} with E8M0 scales.`,
      },
      {
        id: "iw",
        label: "head weights",
        width: ih,
        output: shape("B", "S", ih),
        what: "Project the original attention input to signed head weights.",
        formula: "w = x Wweightsᵀ / sqrt(index_dim · index_heads)",
        params: projection("indexer.weights_proj.weight", ih, d),
        input: stream,
      },
    ],
    "Indexer",
  );
}
export function indexKeyGraph() {
  return graph(
    [
      {
        id: "ik",
        label: "index wk",
        width: id,
        output: shape("B", "C", id),
        what: "Project unrotated compressor latent. Only compressed KV sources own these weights.",
        formula: "ik = latent Wkᵀ",
        params: projection("indexer.wk.weight", id, h),
        input: shape("B", "C", h),
      },
      {
        id: "iknorm",
        label: "RMSNorm",
        width: id,
        output: shape("B", "C", id),
        what: "Normalize each key independently.",
        formula: "ik = RMSNorm(ik)",
        params: norm("indexer.k_norm.weight", id),
      },
      {
        id: "ikrope",
        label: "tail RoPE",
        width: id,
        output: shape("B", "C", id),
        what: "Rotate the tail using the first token position of each compressed group.",
        formula: "ik_tail = RoPE(ik_tail, group_start)",
      },
      {
        id: "ikquant",
        label: "FP4 round",
        width: id,
        output: shape("B", "C", id),
        what: "Round index keys with the indexer block size and E8M0 scales.",
        formula: "ik = fake_fp4(ik)",
        note: `Indexer quantization block: ${f.quantBlock.value}; separate from compressed KV.`,
      },
      {
        id: "ikcache",
        label: "index-key cache",
        width: id,
        output: shape("B", "C", id),
        kind: "state",
        lane: "instance",
        what: "Publish keys for later indexers in the same KV ownership bundle.",
        formula: "shared.index_k = owner.k_cache",
      },
    ],
    "Indexer",
  );
}
export function selectionGraph(i: number) {
  return graph(
    [
      {
        id: "iscore",
        label: "ReLU score",
        width: f.indexTopK.value,
        output: shape("B", "S", "C"),
        what: "Score all completed compressed positions with the small side attention.",
        formula: "score = Σhead ReLU(iq · ik) · w",
        input: shape("B", "S", ih, id),
      },
      {
        id: "imask",
        label: "completed mask",
        width: f.indexTopK.value,
        output: shape("B", "S", "C"),
        what: "Exclude a compressed group until its last token is visible.",
        formula: "visible = floor((position+1)/ratio); later scores = −∞",
      },
      ...(i >= f.candidateSource.value
        ? [
            {
              id: "candidate",
              label: i === f.candidateSource.value ? "candidate blocks" : "candidate mask",
              width: f.candidateTopK.value * f.candidateBlock.value,
              output: shape("B", "S", "C"),
              what:
                i === f.candidateSource.value
                  ? "Rank blocks by their maximum score and force the newest reachable block into the selected set."
                  : "Read the candidate-source mask; later indexers score only the admitted blocks.",
              formula:
                i === f.candidateSource.value ? "keep top blocks(max score), pin newest" : "score[not candidate] = −∞",
            },
          ]
        : []),
      {
        id: "itop",
        label: "top-k positions",
        width: f.indexTopK.value,
        output: shape("B", "S", `min(${f.indexTopK.value}, C)`),
        what: "Choose strongest positions, then sort chosen indices by position.",
        formula: "indices = sort_positions(topk(score))",
      },
      {
        id: "ipublish",
        label: "publish indices",
        width: f.indexTopK.value,
        output: shape("B", "S", "K"),
        kind: "state",
        lane: "instance",
        what: "Shift compressed indices past the window region and publish them for consumers.",
        formula: "indices += window_offset; unreachable = −1",
      },
    ],
    "Indexer",
  );
}
export function attendGraph() {
  return graph(
    [
      {
        id: "gather",
        label: "window + selected",
        width: h,
        output: shape("B", "S", "K", h),
        what: "Read window positions and selected compressed vectors as one KV set.",
        formula: "reached = concat(window_indices, compressed_indices)",
        input: latent,
      },
      {
        id: "dot",
        label: "scaled dot",
        width: f.heads.value,
        output: shape("B", "S", f.heads.value, "K"),
        what: "Compare each query head to the same reached KV vectors.",
        formula: "scores = q · kv / sqrt(head_dim)",
      },
      {
        id: "sink",
        label: "softmax + sink",
        width: f.heads.value,
        output: shape("B", "S", f.heads.value, "K"),
        what: "The learned sink contributes to the denominator but carries no value vector.",
        formula: "a_j = exp(score_j) / (Σ exp(score) + exp(sink))",
        params: norm("attn_sink", f.heads.value),
      },
      {
        id: "attnvalue",
        label: "weighted KV",
        width: f.qWidth.value,
        output: query,
        what: "Use the reached key vectors as the values too.",
        formula: "o = Σj a_j kv_j",
      },
      {
        id: "inverse",
        label: "inverse RoPE",
        width: f.qWidth.value,
        output: query,
        what: "Remove the query rotation from the output tail before grouped projection.",
        formula: "o_tail = o_tail · exp(−i position frequency)",
      },
    ],
    "Attention.forward",
  );
}
export function outputGraph() {
  return graph(
    [
      {
        id: "ogroup",
        label: "group heads",
        width: f.groupWidth.value,
        output: shape("B", "S", f.groups.value, f.groupWidth.value),
        what: "Partition heads into independent output projection groups.",
        formula: "o = reshape(o, groups, heads_per_group · head_dim)",
        input: query,
      },
      {
        id: "woa",
        label: "grouped wo_a",
        width: f.outputLatent.value,
        output: shape("B", "S", f.groups.value, f.oRank.value),
        what: "Each group has its own low-rank projection; no cross-group mixing here.",
        formula: "z[b,s,g,r] = Σd o[b,s,g,d] W[g,r,d]",
        params: [["wo_a.weight", shape(f.groups.value * f.oRank.value, f.groupWidth.value)]],
      },
      {
        id: "flatten",
        label: "flatten groups",
        width: f.outputLatent.value,
        output: shape("B", "S", f.outputLatent.value),
        what: "Concatenate the independently projected group outputs.",
        formula: "z = flatten(group, rank)",
      },
      {
        id: "wob",
        label: "wo_b",
        width: d,
        output: stream,
        what: "Mix groups and return to the language stream width.",
        formula: "y = z Wo_bᵀ",
        params: projection("wo_b.weight", d, f.outputLatent.value),
      },
    ],
    "Attention.forward",
  );
}
export function routerGraph() {
  return graph(
    [
      {
        id: "router",
        label: "router linear",
        width: f.experts.value,
        output: shape("B·S", f.experts.value),
        what: "Project each normalized token to an expert logit in fp32.",
        formula: "logits = x Wgateᵀ / temperature",
        params: projection("gate.weight", f.experts.value, d),
      },
      {
        id: "score",
        label: "sqrtsoftplus",
        width: f.experts.value,
        output: shape("B·S", f.experts.value),
        what: "Convert logits to positive routing scores; this is not softmax.",
        formula: "score = sqrt(log(1+exp(logit)))",
      },
      {
        id: "bias",
        label: "selection bias",
        width: f.experts.value,
        output: shape("B·S", f.experts.value),
        what: "Choose text or image correction bias for expert selection only.",
        formula: "ranking = score + bias_text_or_image",
        params: [["bias + bias_vl", shape(2, f.experts.value)]],
      },
      {
        id: "topk",
        label: "top-k + gather",
        width: f.topK.value,
        output: shape("B·S", f.topK.value),
        what: "Select by corrected ranking but gather weights from uncorrected scores.",
        formula: "ids = topk(ranking); weights = gather(score,ids)",
      },
      {
        id: "renorm",
        label: "normalize + scale",
        width: f.topK.value,
        output: shape("B·S", f.topK.value),
        what: "Normalize selected scores and multiply the route scale.",
        formula: `w = score_selected / (Σselected score + eps) · ${f.routeScale.value}`,
        note: "No group pruning is present in the publisher Gate.forward.",
      },
    ],
    "Gate",
    "DeepseekV4DecoderLayer",
  );
}
export function expertGraph() {
  const m = f.inter.value;
  const g = graph(
    [
      {
        id: "gateproj",
        input: shape("T", d),
        label: "w1 · gate",
        width: m,
        output: shape("T", m),
        what: "Build the gate branch of a selected or shared expert.",
        formula: "g = x W1ᵀ",
        params: projection("w1.weight", m, d),
      },
      {
        id: "upproj",
        label: "w3 · up",
        width: m,
        output: shape("T", m),
        what: "Build the independent up branch from the same input.",
        formula: "u = x W3ᵀ",
        params: projection("w3.weight", m, d),
        col: 0,
        row: 1,
        input: shape("T", d),
      },
      {
        id: "gclamp",
        label: "upper clamp",
        width: m,
        output: shape("T", m),
        what: "Clip only the positive gate tail.",
        formula: `g = min(g, ${f.limit.value})`,
        col: 1,
        row: 0,
      },
      {
        id: "uclamp",
        label: "symmetric clamp",
        width: m,
        output: shape("T", m),
        what: "Clip both signs of the up branch.",
        formula: `u = clip(u, −${f.limit.value}, ${f.limit.value})`,
        col: 1,
        row: 1,
        input: shape("T", m),
      },
      {
        id: "silu",
        label: "SiLU × up",
        width: m,
        output: shape("T", m),
        what: "Activate the gate and multiply the up branch.",
        formula: "a = SiLU(g) · u",
        col: 2,
        row: 0,
      },
      {
        id: "weight",
        label: "route weight",
        width: m,
        output: shape("T", m),
        what: "Apply the selected routing weight before the down projection; shared expert omits this multiplication.",
        formula: "a = w_route · a",
        col: 3,
        row: 0,
      },
      {
        id: "down",
        label: "w2",
        width: d,
        output: shape("T", d),
        what: "Return expert activations to language width.",
        formula: "y = a W2ᵀ",
        params: projection("w2.weight", d, m),
        col: 4,
        row: 0,
      },
    ],
    "Expert",
    "DeepseekV4DecoderLayer",
  );
  g.edges = [
    { from: "gateproj", to: "gclamp" },
    { from: "upproj", to: "uclamp" },
    { from: "gclamp", to: "silu" },
    { from: "uclamp", to: "silu" },
    { from: "silu", to: "weight" },
    { from: "weight", to: "down" },
  ];
  return g;
}
export function blockGraph(i: number) {
  return graph(
    [
      ...(f.engramLayers.value.includes(i)
        ? [
            {
              id: "engram",
              label: "Engram add",
              width: f.residualWidth.value,
              output: residual,
              what: "Add gated hash-table values before this block computes any mixing coefficients.",
              formula: "H ← Engram(H, hashes)",
              lane: "instance" as const,
              input: residual,
            },
          ]
        : []),
      {
        id: "attnpre",
        label: "incoming collapse",
        width: d,
        output: stream,
        what: "Collapse with coefficients from the previous FFN, not this attention sublayer.",
        formula: "x = Σcopy incoming_pre · H",
        input: residual,
      },
      {
        id: "anorm",
        label: "RMSNorm",
        width: d,
        output: stream,
        what: "Normalize the collapsed attention input.",
        formula: "x = RMSNorm(x)",
        params: norm("attn_norm.weight", d),
      },
      {
        id: "attn",
        label: f.kindOf(i) === "window" ? "window attention" : "sparse + window",
        width: d,
        output: stream,
        what: f.layerDescription(i),
        formula: "y = Attention(x, owner state)",
      },
      {
        id: "apost",
        label: "attention HC post",
        width: f.residualWidth.value,
        output: residual,
        what: "Expand attention output and combine original copies with this sublayer’s Sinkhorn matrix.",
        formula: "H′_j = post_j y + Σi comb_ij H_i",
      },
      {
        id: "ffnpre",
        label: "attention collapse",
        width: d,
        output: stream,
        what: "Use attention’s pre coefficients to collapse the stream for FFN.",
        formula: "x = Σcopy attention_pre · H′",
        input: residual,
      },
      {
        id: "fnorm",
        label: "RMSNorm + MoE",
        width: d,
        output: stream,
        what: "Normalize then route to selected experts and the separate shared expert.",
        formula: "y = MoE(RMSNorm(x))",
        params: norm("ffn_norm.weight", d),
      },
      {
        id: "fpost",
        label: "FFN HC post",
        width: f.residualWidth.value,
        output: residual,
        what: "Mix the residual copies and publish FFN pre coefficients for the next block.",
        formula: "H″_j = post_j y + Σi comb_ij H′_i",
      },
    ],
    "Block.forward",
    "DeepseekV4DecoderLayer",
  );
}
export function mixingGraph() {
  return graph(
    [
      {
        id: "flat",
        label: "flatten copies",
        width: f.residualWidth.value,
        output: shape("B", "S", f.residualWidth.value),
        what: "Mixing statistics cover all copies jointly, unlike sublayer RMSNorm.",
        formula: "flat = flatten(H, copy, channel)",
        input: residual,
      },
      {
        id: "mixproj",
        label: "normalized linear",
        width: f.mixWidth.value,
        output: shape("B", "S", f.mixWidth.value),
        what: "Project the flattened stream and divide by its RMS.",
        formula: "mixes = flat Wmixᵀ · rsqrt(mean(flat²)+eps)",
        params: projection("hc_fn", f.mixWidth.value, f.residualWidth.value),
      },
      {
        id: "split",
        label: "split + scales",
        width: f.mixWidth.value,
        output: shape("B", "S", f.mixWidth.value),
        what: "Split collapse, expansion and residual-matrix logits with separate learned scales and bases.",
        formula: "pre=sigmoid(a)+eps; post=2 sigmoid(b); logits=c",
        params: [
          ["hc_scale", shape(3)],
          ["hc_base", shape(f.mixWidth.value)],
        ],
      },
      {
        id: "sinkhorn",
        label: "Sinkhorn",
        width: f.copies.value * f.copies.value,
        output: shape("B", "S", f.copies.value, f.copies.value),
        what: "Row softmax and alternating column/row normalization approximate a doubly stochastic residual matrix.",
        formula: "comb = alternating_normalize(softmax_rows(logits)+eps)",
        note: `${f.iterations.value} iterations; finite epsilon means sums are approximate.`,
      },
    ],
    "Block.hc_mixes",
    "DeepseekV4DecoderLayer",
  );
}
export function engramGraph() {
  return graph(
    [
      {
        id: "lookup",
        label: "hash lookup",
        width: f.hashWidth.value,
        output: shape("B", "S", f.hashColumns.value, f.hashDim.value),
        what: "Gather independent prime-bucket rows from the n-gram table and dequantize them.",
        formula: "e = table[hash_ids] · row_scales",
        params: [["embed.weight", shape("rows", f.hashDim.value)]],
        input: shape("B", "S", f.hashColumns.value),
      },
      {
        id: "ewkv",
        label: "flatten + wkv",
        width: (f.copies.value + 1) * d,
        output: shape("B", "S", (f.copies.value + 1) * d),
        what: "Turn concatenated hash embeddings into a key per residual copy and one shared value.",
        formula: "[keys,value] = flatten(e) Wkvᵀ",
        params: projection("wkv.weight", (f.copies.value + 1) * d, f.hashWidth.value),
      },
      {
        id: "edot",
        label: "normalized dot",
        width: f.copies.value,
        output: shape("B", "S", f.copies.value),
        what: "Compare each copy with its key, normalized independently over channels.",
        formula: "a = Σd H · q_gain · k_gain · key / (RMS(H) RMS(key) sqrt(d))",
        params: [["q_weight + k_weight", shape(2, f.copies.value, d)]],
      },
      {
        id: "egate",
        label: "signed sqrt gate",
        width: f.copies.value,
        output: shape("B", "S", f.copies.value),
        what: "Preserve dot-product sign before sigmoid. Image spans force the gate off.",
        formula: "gate = sigmoid(sign(a) sqrt(max(abs(a), clamp_eps)))",
      },
      {
        id: "eadd",
        label: "gated add",
        width: f.residualWidth.value,
        output: residual,
        what: "Broadcast the shared value to every copy with its own gate.",
        formula: "H′ = H + gate[...,None] · value[...,None,:]",
      },
    ],
    "Engram",
    "DeepseekV4DecoderLayer",
  );
}

export function boundaryGraph(i: number, which: "attn" | "ffn") {
  const attn = which === "attn";
  const g = graph(
    [
      {
        id: "saved",
        label: "saved residual",
        width: f.residualWidth.value,
        output: residual,
        input: residual,
        what:
          attn && f.engramLayers.value.includes(i)
            ? "Saved stream after the Engram addition, before attention."
            : "Save the incoming copies before collapse.",
        formula: "saved = H",
        col: 0,
        row: 0,
        lane: "skip",
      },
      {
        id: "collapse",
        label: attn ? "incoming collapse" : "attention collapse",
        width: d,
        output: stream,
        input: residual,
        what: "Read handed-off pre coefficients, not the current sublayer’s new pre.",
        formula: "x = Σcopy incoming_pre · H",
        col: 1,
        row: 0,
      },
      {
        id: "norm",
        label: "RMSNorm",
        width: d,
        output: stream,
        what: "Normalize collapsed input.",
        formula: "x = RMSNorm(x)",
        params: norm(which + "_norm.weight", d),
        col: 2,
        row: 0,
      },
      {
        id: "sublayer",
        label: attn ? (f.kindOf(i) === "window" ? "window attention" : "sparse + window") : "routed + shared MoE",
        width: d,
        output: stream,
        what: attn ? f.layerDescription(i) : "Run selected experts and independent shared expert.",
        formula: attn ? "y = Attention(x, owner state)" : "y = MoE(x)",
        col: 3,
        row: 0,
        lane: attn ? "high" : "low",
      },
      {
        id: "post",
        label: "HC post",
        width: f.residualWidth.value,
        output: residual,
        what: "Expand current output and mix the saved copies with current coefficients.",
        formula: "H′_j = post_j y + Σi comb_ij saved_i",
        col: 4,
        row: 0,
        lane: "skip",
      },
      {
        id: "incoming",
        label: attn ? "previous FFN pre" : "attention pre",
        width: f.copies.value,
        output: shape("B", "S", f.copies.value),
        what: attn
          ? "Previous FFN hands off collapse coefficients; initial block uses first-copy selection."
          : "Attention hands off its newly computed pre coefficients.",
        formula: "incoming_pre = previous_sublayer.pre",
        col: 0,
        row: 1,
        kind: "state",
        lane: "skip",
      },
      {
        id: "current",
        label: "current HC mixes",
        width: f.mixWidth.value,
        output: shape("B", "S", f.mixWidth.value),
        input: residual,
        what: "Compute current pre, post and comb from saved H. New pre is handed to the next sublayer.",
        formula: "pre,post,comb = hc_mixes(saved)",
        col: 2,
        row: 2,
        lane: "skip",
      },
    ],
    "Block.forward",
    "DeepseekV4DecoderLayer",
  );
  g.edges = [
    { from: "saved", to: "collapse" },
    { from: "incoming", to: "collapse", kind: "state" },
    { from: "collapse", to: "norm" },
    { from: "norm", to: "sublayer" },
    { from: "sublayer", to: "post" },
    { from: "saved", to: "post", kind: "skip" },
    { from: "saved", to: "current" },
    { from: "current", to: "post", kind: "state" },
  ];
  return g;
}

export function compressedCacheGraph(i: number) {
  const r = f.ratios.value[i];
  return graph(
    [
      {
        id: "cachelatent",
        label: "normalized latent",
        width: h,
        output: shape("B", `⌊S/${r}⌋`, h),
        input: shape("B", `⌊S/${r}⌋`, h),
        what: "Read compressor output after the index-key path has consumed the unrotated latent.",
        formula: "latent = Compressor(x)",
      },
      {
        id: "crope",
        label: "tail RoPE",
        width: h,
        output: shape("B", `⌊S/${r}⌋`, h),
        what: "Use the first token position of each complete group with compressed-base YaRN frequencies.",
        formula: "latent_tail = RoPE(latent_tail, group·ratio)",
      },
      {
        id: "cquant",
        label: "FP4 round",
        width: h,
        output: shape("B", `⌊S/${r}⌋`, h),
        what: "Round the rotated vector, including its rotary tail.",
        formula: "latent = fake_fp4(latent)",
        note: `Compressed KV uses groups of ${f.compressedQuantBlock.value} with E4M3 scales; indexer uses groups of ${f.quantBlock.value} with E8M0.`,
      },
      {
        id: "ccache",
        label: "owner cache write",
        width: h,
        output: shape("B", "C", h),
        kind: "state",
        lane: "instance",
        what: "Write complete vectors into the source cache, then expose that owner’s updated cache to later layers.",
        formula: "owner_cache[group_range] = latent; shared.compress_kv = owner_cache",
      },
    ],
    "Attention._compress_kv",
  );
}
