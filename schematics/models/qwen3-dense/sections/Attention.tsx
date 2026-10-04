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
    label,
    col,
    row,
    width: ["dot", "scale", "mask", "softmax"].includes(id) ? f.heads.value * f.DISPLAY.tokens : width,
    sub: ["dot", "scale", "mask", "softmax"].includes(id) ? "over available keys" : n(width),
    lane: id === "freq" || id === "trig" || id === "rope" ? "instance" : "high",
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
        { source: "training", symbol: "Qwen3Attention.forward" },
        ...(id === "freq" || id === "trig"
          ? [
              { source: "reference", symbol: "Qwen3RotaryEmbedding.forward" },
              { source: "rope", symbol: "_compute_default_rope_parameters" },
            ]
          : []),
        ...(id === "rope" || id === "rotate" ? [{ source: "reference", symbol: "apply_rotary_pos_emb" }] : []),
        { source: "reference", symbol: "Qwen3Attention.forward" },
        ...(["repeat", "dot", "scale", "mask", "softmax", "weighted", "merge", "attend"].includes(id)
          ? [{ source: "reference", symbol: "eager_attention_forward" }]
          : []),
        { source: "serving", symbol: "Qwen3Attention.forward" },
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
      "Q: project + norm",
      1,
      0,
      f.qWidth.value,
      qs,
      "Build queries and normalize each head.",
      "RMSNorm(x Wqᵀ)",
      x,
    ),
    op(
      "kv",
      "K/V: project",
      1,
      1,
      f.kvWidth.value,
      ks,
      "Build keys and values; normalize keys only.",
      "K=RMSNorm(x Wkᵀ); V=x Wvᵀ",
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
  const project = (prefix: string, heads: number, norm: boolean): Op[] => {
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
        norm ? shape("B", "S", heads, h) : s,
        norm
          ? "Introduce the head axis before normalization. Transpose follows the norm."
          : "Introduce the value head axis and transpose for attention.",
        norm ? "[B,S,H D] → [B,S,H,D]" : "[B,S,H,D] → [B,H,S,D]",
        shape("B", "S", w),
      ),
    ];
    if (norm)
      out.push(
        op(
          prefix + "norm",
          "head RMSNorm",
          2,
          0,
          w,
          s,
          "Normalize the last axis independently for each head, with one shared gain vector; then transpose sequence and head axes.",
          `y=x / sqrt(mean(x²)+${String(f.normEps.value)}) · g`,
          shape("B", "S", heads, h),
          [[prefix + "_norm.weight", shape(h)]],
        ),
      );
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
      title="Attention: normalize, rotate, share"
      lede="Queries and keys are normalized per head before positions rotate them. The key/value cache stays narrow even though every query head attends independently."
      source={[{ source: "reference", symbol: "Qwen3Attention" }]}
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
      {(["q", "k", "v"] as const).map((p) => {
        const ops = project(p, p === "q" ? f.heads.value : f.kvHeads.value, p !== "v");
        return (
          <div key={p}>
            <OpGraph
              id={p + "build"}
              heading={`Build ${p.toUpperCase()}`}
              title={`${p} projection and head coordinates`}
              ops={ops}
              edges={chain(ops)}
            />
            <Prose>
              {p === "v"
                ? "Values are neither normalized nor rotated."
                : "The gain is shared across heads, but the RMS is computed independently within each head."}
            </Prose>
          </div>
        );
      })}
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
          "Q/K normalization precedes rotation; V bypasses both.",
          "Cache stores rotated keys before head repetition.",
          "Causal visibility does not imply equal attention weights.",
        ]}
      />
    </Section>
  );
}
