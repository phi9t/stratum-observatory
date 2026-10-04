import {
  AttentionGrid,
  Check,
  Figure,
  OpGraph,
  Prose,
  RopeDials,
  Section,
  Sentence,
  Slider,
  Takeaways,
  n,
  shape,
  useUrlNumber,
  type Op,
  type Edge,
} from "../../../kit";
import * as f from "../facts";
import { useLayer } from "./Stack";
const LAYOUT = { qcol: 0, kcol: 1, vcol: 2, cachecol: 3, fourth: 4 };
function op(
  id: string,
  label: string,
  col: number,
  row: number,
  width: number,
  output: string,
  what: string,
  formula: string,
  input: string,
  params: readonly (readonly [string, string])[] = [],
): Op {
  return {
    id,
    kind: id === "cache" ? "state" : "op",
    label,
    col,
    row,
    width: ["dot", "scale", "mask", "softmax"].includes(id) ? f.heads.value * f.DISPLAY.tokens : width,
    sub: ["dot", "scale", "mask", "softmax"].includes(id) ? "over available keys" : n(width),
    lane: ["freq", "trig", "rope", "rotate", "cache"].includes(id) ? "instance" : "high",
    detail: {
      what,
      formula,
      inputs:
        id === "dot"
          ? [
              ["Q", shape("B", f.heads.value, "S", f.headDim.value)],
              ["K repeated", shape("B", f.heads.value, "T", f.headDim.value)],
            ]
          : id === "weighted"
            ? [
                ["weights", input],
                ["V repeated", shape("B", f.heads.value, "T", f.headDim.value)],
              ]
            : id === "write"
              ? [
                  ["rotated K", input],
                  ["unrotated V", input],
                  ["prior K/V", shape("B", f.kvHeads.value, "past", f.headDim.value)],
                ]
              : id === "rope" || id === "rotate"
                ? [
                    ["Q", shape("B", f.heads.value, "S", f.headDim.value)],
                    ["K", shape("B", f.kvHeads.value, "S", f.headDim.value)],
                    ["cos / sin", shape("B", "S", f.headDim.value)],
                  ]
                : [["input", input]],
      output:
        id === "write" || id === "cache"
          ? ["K / V", output]
          : id === "rope" || id === "rotate"
            ? ["Q; K retains its KV-head axis", output]
            : [id, output],
      params,
      note: ["dot", "scale", "mask", "softmax"].includes(id)
        ? "Score gauges use the shown prefix length; T is the full available key length. The serving backend may fuse operations."
        : "No learned parameter is added unless listed. Shapes show the reference circuit; serving may fuse these operations.",
      refs: [
        { source: "training", symbol: "CausalSelfAttention._forward_training" },
        ...(id === "freq" || id === "trig"
          ? [
              { source: "reference", symbol: "LlamaRotaryEmbedding.forward" },
              { source: "rope", symbol: "_compute_default_rope_parameters" },
            ]
          : []),
        ...(id === "rope" || id === "rotate" ? [{ source: "reference", symbol: "apply_rotary_pos_emb" }] : []),
        { source: "reference", symbol: "LlamaAttention.forward" },
        ...(["repeat", "dot", "scale", "mask", "softmax", "weighted", "merge", "attend"].includes(id)
          ? [{ source: "reference", symbol: "eager_attention_forward" }]
          : []),
        { source: "serving", symbol: "LlamaAttention.forward" },
      ],
    },
  };
}
const chain = (ops: Op[]): Edge[] => ops.slice(1).map((o, i) => ({ from: ops[i].id, to: o.id }));
export function Attention() {
  const [layer] = useLayer();
  const [q, setQ] = useUrlNumber("q", 0, 0, f.DISPLAY.tokens - 1);
  const [pos, setPos] = useUrlNumber("ropePos", 0, 0, f.positions.value - 1);
  const d = f.hidden.value,
    h = f.headDim.value;
  const x = shape("B", "S", d),
    qs = shape("B", f.heads.value, "S", h),
    ks = shape("B", f.kvHeads.value, "S", h),
    score = shape("B", f.heads.value, "S", "T");
  const topology: Op[] = [
    op("stream", "normalized stream", 0, 0, d, x, "All projections read the same normalized input.", "x", x),
    op(
      "query",
      "Q: project",
      1,
      0,
      f.qWidth.value,
      qs,
      "Build queries from the normalized residual stream.",
      "x Wqᵀ",
      x,
    ),
    op(
      "kv",
      "K/V: project",
      1,
      1,
      f.kvWidth.value,
      ks,
      "Build narrow keys and values from the same stream.",
      "K=x Wkᵀ; V=x Wvᵀ",
      x,
    ),
    op(
      "rotate",
      "rotate Q/K",
      2,
      0,
      f.qWidth.value,
      qs,
      "Positions rotate queries and keys, not values.",
      "RoPE(Q,K)",
      qs,
    ),
    op(
      "cache",
      "KV cache",
      2,
      1,
      f.kvWidth.value,
      ks,
      "Store rotated keys and raw values, append old state.",
      "K,V ← cache.update(K,V)",
      ks,
    ),
    op(
      "attend",
      "attend + project",
      LAYOUT.cachecol,
      0,
      d,
      x,
      "Query heads gather from shared KV heads.",
      "softmax(mask(QKᵀ / sqrt(D))) V Woᵀ",
      qs,
    ),
  ];
  const project = (prefix: string, heads: number): Op[] => {
    const w = heads * h,
      s = shape("B", heads, "S", h);
    const out = [
      op(
        prefix + "proj",
        prefix + " projection",
        0,
        0,
        w,
        shape("B", "S", w),
        "Bias-free projection into per-head coordinates.",
        "y=x Wᵀ",
        x,
        [[prefix + "_proj.weight", shape(w, d)]],
      ),
      op(
        prefix + "shape",
        "reshape to heads",
        1,
        0,
        w,
        s,
        "Introduce the head axis and transpose sequence and head axes for attention.",
        "[B,S,H D] → [B,H,S,D]",
        shape("B", "S", w),
      ),
    ];
    return out;
  };
  const rot = [
    op(
      "freq",
      "position × frequency",
      0,
      0,
      f.rotaryPairs.value,
      shape("B", "S", f.rotaryPairs.value),
      "Compute shared rotary angles once for the whole stack.",
      `angle = position × ${n(f.theta.value)}^(-2i/${n(h)})`,
      shape("B", "S"),
    ),
    op(
      "trig",
      "cosine + sine",
      1,
      0,
      h,
      shape("B", "S", h),
      "Duplicate half-width frequencies, then compute cosine and sine in float32.",
      "concat(angle,angle), then cos and sin",
      shape("B", "S", f.rotaryPairs.value),
    ),
    op(
      "rope",
      "rotate Q and K",
      2,
      0,
      f.qWidth.value,
      qs,
      "Pair first-half and second-half coordinates; values bypass this operation.",
      "x cos + rotate_half(x) sin",
      qs,
    ),
    {
      ...op(
        "write",
        "cache write + read",
        LAYOUT.cachecol,
        0,
        f.kvWidth.value,
        shape("B", f.kvHeads.value, "T", h),
        "Append rotated keys and unrotated values to this layer’s prior state. Cache is optional in prefill.",
        "Kall,Vall = append(Kpast,Vpast,K,V)",
        ks,
      ),
      kind: "state" as const,
      lane: "instance" as const,
    },
  ];
  const mix = [
    op(
      "repeat",
      "repeat KV heads",
      0,
      0,
      f.qWidth.value,
      shape("B", f.heads.value, "T", h),
      `Each KV head is repeated for ${n(f.groups.value)} query heads in eager mode. The cache itself stays narrow.`,
      "repeat_interleave(K,V,groups)",
      shape("B", f.kvHeads.value, "T", h),
    ),
    op(
      "dot",
      "Q · Kᵀ",
      1,
      0,
      f.qWidth.value,
      score,
      "Compare each query against all available keys.",
      "scores = Q @ Kᵀ",
      qs,
    ),
    op(
      "scale",
      "scale scores",
      2,
      0,
      f.qWidth.value,
      score,
      "Scale dot products by head width.",
      `scores × ${n(h)}^-0.5`,
      score,
    ),
    op(
      "mask",
      "causal + padding mask",
      LAYOUT.cachecol,
      0,
      f.qWidth.value,
      score,
      "Future and padding positions receive an exclusion score. The unpadded demonstration uses only causality.",
      "scores + mask",
      score,
    ),
    op(
      "softmax",
      "softmax",
      LAYOUT.fourth,
      0,
      f.qWidth.value,
      score,
      "Normalize over keys in float32. Dropout is disabled for this checkpoint.",
      "exp(scores) / sum(exp(scores))",
      score,
    ),
  ];
  const output = [
    op(
      "weighted",
      "weights · V",
      0,
      0,
      f.qWidth.value,
      qs,
      "Average values separately for each query head.",
      "weights @ V",
      score,
    ),
    op(
      "merge",
      "merge heads",
      1,
      0,
      f.qWidth.value,
      shape("B", "S", f.qWidth.value),
      "Transpose back and flatten the head axis.",
      "[B,H,S,D] → [B,S,H D]",
      qs,
    ),
    op(
      "out",
      "output projection",
      2,
      0,
      d,
      x,
      "Return the combined heads to the residual width.",
      "y=mixed Woᵀ",
      shape("B", "S", f.qWidth.value),
      [["o_proj.weight", shape(d, f.qWidth.value)]],
    ),
  ];
  return (
    <Section
      id="attention"
      title="Attention: project, rotate, share"
      lede="Queries and keys are projected directly before positions rotate them. The key/value cache stays narrow even though every query head attends independently."
      source={[{ source: "reference", symbol: "LlamaAttention" }]}
    >
      <OpGraph
        id="topology"
        heading="The attention circuit"
        title={`Attention topology, layer ${n(layer)}`}
        ops={topology}
        edges={[
          { from: "stream", to: "query" },
          { from: "stream", to: "kv" },
          { from: "query", to: "rotate" },
          { from: "kv", to: "rotate" },
          { from: "rotate", to: "cache", kind: "state" },
          { from: "kv", to: "cache", kind: "state", label: "V" },
          { from: "rotate", to: "attend" },
          { from: "cache", to: "attend", kind: "state" },
        ]}
      />
      <Prose>
        The upper query branch and lower state branch meet at attention. The next drawings unpack those branches without
        changing their dependencies.
      </Prose>
      <OpGraph
        id="projections"
        heading="Project and expose the head axes"
        title="Independent Q, K and V branches"
        ops={(["q", "k", "v"] as const).flatMap((p, row) =>
          project(p, p === "q" ? f.heads.value : f.kvHeads.value).map((o) => ({ ...o, row })),
        )}
        edges={(["q", "k", "v"] as const).map((p) => ({ from: p + "proj", to: p + "shape" }))}
      />
      <Prose>
        All branches read the block RMSNorm output. Queries and keys have no extra head norms. Queries keep the residual
        width; keys and values each use a narrower head axis.
      </Prose>
      <OpGraph
        id="rotary"
        heading="Position and persistent state"
        title="RoPE then cache"
        ops={rot}
        edges={chain(rot).map((e) => (e.to === "write" ? { ...e, kind: "state" as const, label: "K" } : e))}
      />
      <Prose>
        The frequency tables are shared across layers; cached keys have already been rotated. These are dependencies,
        not a serial transformation of queries into keys.
      </Prose>
      <OpGraph id="scores" heading="Compare and normalize" title="Grouped causal scores" ops={mix} edges={chain(mix)} />
      <Prose>
        The probability tensor reads repeated values; the backend may avoid materializing either repetition or the score
        matrix.
      </Prose>
      <OpGraph id="attendout" heading="Gather and return" title="Attention output" ops={output} edges={chain(output)} />
      <Slider label="Query position" min={0} max={f.DISPLAY.tokens - 1} value={q} onChange={setQ} />
      <Sentence>
        Query {n(q)} can read {n(q + 1)} shown positions. Later positions are masked.
      </Sentence>
      <Figure
        label="Causal visibility"
        caption="The rule is exact; the diagram shows a short prefix, not learned scores."
      >
        <AttentionGrid
          title="Full causal attention"
          size={f.DISPLAY.tokens}
          query={q}
          onQuery={setQ}
          cell={(a, b) => (b > a ? "masked" : b === a ? "self" : "attend")}
          legend={[
            { state: "attend", label: "earlier key" },
            { state: "self", label: "self" },
            { state: "masked", label: "future key" },
          ]}
        />
      </Figure>
      <Slider label="Rotary position" min={0} max={f.positions.value - 1} value={pos} onChange={setPos} />
      <Sentence>
        At position {n(pos)}, frequency pairs turn at different rates. The paired coordinates are separated by half a
        head.
      </Sentence>
      <Figure label="Rotary frequencies" caption={`All ${n(h)} head coordinates rotate; values do not.`}>
        <RopeDials
          title="Default rotary positions"
          position={pos}
          pairs={f.DISPLAY.pairs}
          theta={f.theta.value}
          rotaryDim={h}
        />
      </Figure>
      <Check question="Does repeating KV heads require a wider cache?">
        No. Replication happens after the cache update. Each stored head serves {n(f.groups.value)} query heads, giving
        a {n(f.groups.value)}-fold reduction relative to caching a key and value for every query head.
      </Check>
      <Takeaways
        items={[
          "Q and K rotate after projection; V bypasses rotation.",
          "Cache stores rotated keys before head repetition.",
          "Causal visibility does not imply equal attention weights.",
        ]}
      />
    </Section>
  );
}
