import {
  Section,
  OpGraph,
  Figure,
  AttentionGrid,
  Slider,
  Sentence,
  Readout,
  Prose,
  Sub,
  Takeaways,
  Check,
  useUrlNumber,
  shape,
  n,
  type Edge,
  type Op,
} from "../../../kit";
import * as f from "../facts";
import { op, chain, stream, normFormula } from "./graph";
import { yarnFrequency } from "../reference/model";
const LAYOUT = { radius: 32, pitch: 140, cx: 60, cy: 60 };
export function Attention() {
  const [query, setQuery] = useUrlNumber("query", f.demo.tokens - 1, 0, f.demo.tokens - 1);
  const [position, setPosition] = useUrlNumber("position", f.original.value, 0, f.context.value - 1);
  const s = "DeepseekV3Attention";
  const qShape = shape("B", f.heads.value, "S", f.qHead.value);
  const scoreShape = shape("B", f.heads.value, "S", "T");
  const topology: Op[] = [
    op(
      "input",
      "normalized stream",
      f.hidden.value,
      { col: 0, row: 0 },
      s,
      "Attention reads the normalized branch, leaving its residual elsewhere.",
      "x",
      stream,
    ),
    op(
      "qpath",
      "query path",
      f.qWidth.value,
      { col: 1, row: 0 },
      s,
      "Compress, normalize and rebuild head-specific queries.",
      "q_b(RMSNorm(q_a(x)))",
      stream,
      qShape,
    ),
    op(
      "kvpath",
      "key / value path",
      f.kvWidth.value,
      { col: 1, row: 1 },
      s,
      "Compress and rebuild keys and values; positional key is shared.",
      "kv_b(RMSNorm(content(kv_a(x))))",
      stream,
      shape("B", f.heads.value, "S", f.nope.value + f.v.value),
    ),
    op(
      "attend",
      "causal attention",
      f.valueWidth.value,
      { col: 2, row: 0 },
      s,
      "Compare queries and keys, then mix values over visible positions.",
      "softmax(mask(scale(q kᵀ))) v",
      qShape,
      shape("B", "S", f.valueWidth.value),
    ),
    op(
      "out",
      "output projection",
      f.hidden.value,
      { col: 3, row: 0 },
      s,
      "Map merged heads back to the residual width.",
      "merged W_oᵀ",
      shape("B", "S", f.valueWidth.value),
      stream,
    ),
  ];
  const qops = [
    op(
      "qa",
      "q_a_proj",
      f.qRank.value,
      { col: 0, row: 0 },
      s,
      "Compress each token's query input.",
      "x W_qaᵀ",
      stream,
      shape("B", "S", f.qRank.value),
      [["q_a_proj.weight", shape(f.qRank.value, f.hidden.value)]],
    ),
    op(
      "qn",
      "query RMSNorm",
      f.qRank.value,
      { col: 1, row: 0 },
      s,
      "Normalize the query bottleneck.",
      normFormula,
      shape("B", "S", f.qRank.value),
      shape("B", "S", f.qRank.value),
      [["query gain", shape(f.qRank.value)]],
    ),
    op(
      "qb",
      "q_b_proj",
      f.qWidth.value,
      { col: 2, row: 0 },
      s,
      "Rebuild all query heads from the bottleneck.",
      "q_norm W_qbᵀ",
      shape("B", "S", f.qRank.value),
      shape("B", "S", f.qWidth.value),
      [["q_b_proj.weight", shape(f.qWidth.value, f.qRank.value)]],
    ),
    op(
      "qh",
      "reshape heads",
      f.qHead.value,
      { col: 3, row: 0 },
      s,
      "Introduce the independent head axis and move sequence after heads.",
      "view(B,S,H,D).transpose(S,H)",
      shape("B", "S", f.qWidth.value),
      qShape,
    ),
    op(
      "qs",
      "split content / pos",
      f.rope.value,
      { col: 4, row: 0 },
      s,
      "Separate unrotated content from the positional slice.",
      `split last axis: ${n(f.nope.value)} + ${n(f.rope.value)}`,
      qShape,
      `${shape("B", f.heads.value, "S", f.nope.value)}; ${shape("B", f.heads.value, "S", f.rope.value)}`,
    ),
  ];
  const kvops = [
    op(
      "kva",
      "kv_a_proj",
      f.latentWidth.value,
      { col: 0, row: 0 },
      s,
      "Produce a shared content latent and positional key.",
      "x W_kvaᵀ",
      stream,
      shape("B", "S", f.latentWidth.value),
      [["kv_a weight", shape(f.latentWidth.value, f.hidden.value)]],
      "high",
    ),
    op(
      "splitkv",
      "split latent / pos",
      f.kvRank.value,
      { col: 1, row: 0 },
      s,
      "Keep the positional slice outside latent normalization.",
      `split: ${n(f.kvRank.value)} + ${n(f.rope.value)}`,
      shape("B", "S", f.latentWidth.value),
      `${shape("B", "S", f.kvRank.value)}; ${shape("B", 1, "S", f.rope.value)}`,
    ),
    op(
      "kvnorm",
      "latent RMSNorm",
      f.kvRank.value,
      { col: 2, row: 0 },
      s,
      "Normalize the compressed content representation only.",
      normFormula,
      shape("B", "S", f.kvRank.value),
      shape("B", "S", f.kvRank.value),
      [["latent gain", shape(f.kvRank.value)]],
    ),
    op(
      "kvb",
      "kv_b_proj",
      f.kvWidth.value,
      { col: 3, row: 0 },
      s,
      "Expand the normalized latent to head-specific content keys and values.",
      "latent W_kvbᵀ",
      shape("B", "S", f.kvRank.value),
      shape("B", "S", f.kvWidth.value),
      [["kv_b_proj.weight", shape(f.kvWidth.value, f.kvRank.value)]],
    ),
    op(
      "kvheads",
      "reshape heads",
      f.nope.value + f.v.value,
      { col: 4, row: 0 },
      s,
      "Introduce the head axis before splitting keys from values.",
      "view(B,S,H,D).transpose(S,H)",
      shape("B", "S", f.kvWidth.value),
      shape("B", f.heads.value, "S", f.nope.value + f.v.value),
    ),
    op(
      "kvsplit",
      "split key / value",
      f.v.value,
      { col: 5, row: 0 },
      s,
      "Content keys and values get separate meanings after the expansion.",
      `split: ${n(f.nope.value)} + ${n(f.v.value)}`,
      shape("B", f.heads.value, "S", f.nope.value + f.v.value),
      `${shape("B", f.heads.value, "S", f.nope.value)}; ${shape("B", f.heads.value, "S", f.v.value)}`,
    ),
  ];
  const joins = [
    op(
      "freq",
      "YaRN frequencies",
      f.rope.value,
      { col: 0, row: 1 },
      "DeepseekV3YarnRotaryEmbedding",
      "Blend extrapolated and stretched frequencies, then build cosines and sines at absolute positions.",
      "ω = ω_extra × (1 − ramp) + ω_extra / factor × ramp",
      shape("S"),
      shape("S", f.rope.value),
      [],
      "instance",
      `Amplitude ratio cancels because mscale = mscale_all_dim. Both β values are ${f.betaFast.value}.`,
    ),
    op(
      "qr",
      "rotate q slice",
      f.rope.value,
      { col: 1, row: 0 },
      s,
      "Reorder adjacent pairs into halves, rotate each query positional slice.",
      "q_pos × cos + rotate_half(q_pos) × sin",
      shape("B", f.heads.value, "S", f.rope.value),
      shape("B", f.heads.value, "S", f.rope.value),
      [],
      "high",
    ),
    op(
      "kr",
      "rotate k slice",
      f.rope.value,
      { col: 1, row: 1 },
      s,
      "Rotate the positional key before broadcasting to all heads.",
      "k_pos × cos + rotate_half(k_pos) × sin",
      shape("B", 1, "S", f.rope.value),
      shape("B", 1, "S", f.rope.value),
      [],
      "high",
    ),
    op(
      "qjoin",
      "join query",
      f.qHead.value,
      { col: 2, row: 0 },
      s,
      "Rejoin content and rotated positional query channels.",
      "concat(q_content, q_rotary)",
      shape("B", f.heads.value, "S", f.nope.value),
      qShape,
    ),
    op(
      "kjoin",
      "broadcast + join key",
      f.qHead.value,
      { col: 2, row: 1 },
      s,
      "Copy the shared rotated key into each head's positional slice.",
      "concat(k_content, broadcast(k_rotary))",
      shape("B", 1, "S", f.rope.value),
      qShape,
    ),
    op(
      "cache",
      "cache update / read",
      f.expandedCache.value,
      { col: 3, row: 1 },
      s,
      "Append fresh expanded keys and values; read the full prefix for attention.",
      "K,V = past.update(K_fresh,V_fresh,layer)",
      qShape,
      `${shape("B", f.heads.value, "T", f.qHead.value)}; ${shape("B", f.heads.value, "T", f.v.value)}`,
      [],
      "instance",
      "Reference caches expanded tensors. vLLM MLA keeps latent plus positional key; see Serving.",
    ),
  ];
  const weights = [
    op(
      "dot",
      "q · kᵀ",
      f.heads.value,
      { col: 0, row: 0 },
      s,
      "Compute a comparison for every visible query-key pair per head.",
      "q @ transpose(k)",
      qShape,
      scoreShape,
    ),
    op(
      "scale",
      "YaRN scale",
      f.heads.value,
      { col: 1, row: 0 },
      s,
      "Apply both the head-width scale and the YaRN all-dimension multiplier.",
      `scores × ${f.softmaxScale.value}`,
      scoreShape,
      scoreShape,
    ),
    op(
      "mask",
      "causal mask",
      f.heads.value,
      { col: 2, row: 0 },
      s,
      "Prevent each query from reading later tokens or padding.",
      "scores + mask; future → −∞",
      scoreShape,
      scoreShape,
      [],
      "instance",
    ),
    op(
      "softmax",
      "softmax",
      f.heads.value,
      { col: 3, row: 0 },
      s,
      "Turn the visible comparisons into normalized value-mixing weights.",
      "exp(score − max) / Σ exp(score − max)",
      scoreShape,
      scoreShape,
      [],
      "high",
      `Computed in float32 and cast back. Dropout is ${f.dropout.value}; no dropout operation changes inference values.`,
    ),
    op(
      "mix",
      "weights · v",
      f.v.value,
      { col: 4, row: 0 },
      s,
      "Average values separately for every query and head.",
      "weights @ V",
      scoreShape,
      shape("B", f.heads.value, "S", f.v.value),
    ),
    op(
      "merge",
      "merge heads",
      f.valueWidth.value,
      { col: 5, row: 0 },
      s,
      "Move head axis next to value channels and flatten it.",
      "transpose(H,S).reshape(B,S,H×V)",
      shape("B", f.heads.value, "S", f.v.value),
      shape("B", "S", f.valueWidth.value),
    ),
    op(
      "o",
      "o_proj",
      f.hidden.value,
      { col: 6, row: 0 },
      s,
      "Return the attention update to the stream's width.",
      "merged W_oᵀ",
      shape("B", "S", f.valueWidth.value),
      stream,
      [["o_proj.weight", shape(f.hidden.value, f.valueWidth.value)]],
    ),
  ];
  const joinEdges: Edge[] = [
    { from: "freq", to: "qr", kind: "state" },
    { from: "freq", to: "kr", kind: "state" },
    { from: "qr", to: "qjoin" },
    { from: "kr", to: "kjoin" },
    { from: "kjoin", to: "cache", kind: "state" },
  ];
  return (
    <Section
      id="attention"
      title="Attention through a latent"
      source={[{ source: "reference", symbol: s }]}
      lede="Queries and keys pass through narrow learned bottlenecks. Content and position then meet in a head-specific comparison; only the positional key is shared."
    >
      <OpGraph
        id="attention-topology"
        title="Attention topology"
        heading="The paths before the parts"
        ops={topology}
        edges={[
          { from: "input", to: "qpath" },
          { from: "input", to: "kvpath" },
          { from: "qpath", to: "attend" },
          { from: "kvpath", to: "attend" },
          { from: "attend", to: "out" },
        ]}
      />
      <Prose>
        The upper path builds queries. The lower path rebuilds keys and values from a shared latent. Their projections
        run independently and meet at the score calculation.
      </Prose>
      <OpGraph id="query-ops" title="Low rank queries" heading="Build the queries" ops={qops} edges={chain(qops)} />
      <Prose>
        The key/value path has a separate bottleneck. Its positional output bypasses content normalization and
        expansion.
      </Prose>
      <OpGraph
        id="kv-ops"
        title="Latent keys and values"
        heading="Build content keys and values"
        ops={kvops}
        edges={chain(kvops)}
      />
      <Prose>
        Rotate only the positional slices, then join them with content. The cache boundary is explicit here: T includes
        the cached prefix and the fresh sequence.
      </Prose>
      <OpGraph
        id="position-ops"
        title="Rotary joins and expanded cache"
        heading="Position and cached state"
        ops={joins}
        edges={joinEdges}
      />
      <Prose>
        The joined queries compare with the cached keys. The resulting weights mix values, which have no positional
        slice.
      </Prose>
      <OpGraph
        id="attend-ops"
        title="Attend and project"
        heading="Compare, mask and mix"
        ops={weights}
        edges={chain(weights)}
      />
      <Sub>Who can read whom</Sub>
      <Slider label="Query position" value={query} onChange={setQuery} min={0} max={f.demo.tokens - 1} />
      <Sentence>
        Query {n(query)} reads {n(query + 1)} positions, from the beginning through itself.
      </Sentence>
      <Figure
        label="Causal attention visibility"
        caption="Every head and every layer uses this full causal pattern. There is no sparse indexer in this checkpoint."
      >
        <AttentionGrid
          title="Full causal attention"
          size={f.demo.tokens}
          query={query}
          onQuery={setQuery}
          cell={(q, k) => (k > q ? "masked" : k === q ? "self" : "attend")}
          legend={[
            { state: "attend", label: "earlier, readable" },
            { state: "self", label: "current position" },
            { state: "masked", label: "future, masked" },
          ]}
        />
      </Figure>
      <Sub>Position stretches frequencies, not the stream</Sub>
      <Slider label="Absolute position" value={position} onChange={setPosition} min={0} max={f.context.value - 1} />
      <Sentence>
        At position {n(position)}, the slow rotary frequencies use stretched angles; the fastest retain the original
        base frequency.
      </Sentence>
      <Figure
        label="Actual YaRN frequency rotations"
        caption={`Angles use the checkpoint's correction ramp, base ${n(f.theta.value)} and stretch ${n(f.factor.value)}. The arrows show phase, not trained activations.`}
      >
        <svg
          className="ms-fig ms-dial"
          role="img"
          aria-label="YaRN rotation phases"
          width={640}
          height={160}
          viewBox="0 0 640 160"
        >
          {f.rotaryPairs.value.map((pair, index) => {
            const angle = position * yarnFrequency(pair, f.real);
            return (
              <g key={pair}>
                <circle cx={LAYOUT.cx + index * LAYOUT.pitch} cy={LAYOUT.cy} r={LAYOUT.radius} />
                <line
                  data-ghost="true"
                  x1={LAYOUT.cx + index * LAYOUT.pitch}
                  x2={LAYOUT.cx + index * LAYOUT.pitch + LAYOUT.radius}
                  y1={LAYOUT.cy}
                  y2={LAYOUT.cy}
                />
                <line
                  x1={LAYOUT.cx + index * LAYOUT.pitch}
                  x2={LAYOUT.cx + index * LAYOUT.pitch + LAYOUT.radius * Math.cos(angle)}
                  y1={LAYOUT.cy}
                  y2={LAYOUT.cy - LAYOUT.radius * Math.sin(angle)}
                />
                <text className="ms-fig__note" textAnchor="middle" x={LAYOUT.cx + index * LAYOUT.pitch} y={116}>
                  pair {n(pair)}
                </text>
                <text className="ms-node__line" textAnchor="middle" x={LAYOUT.cx + index * LAYOUT.pitch} y={136}>
                  {(angle % (2 * Math.PI)).toFixed(f.demo.radDigits)} rad
                </text>
              </g>
            );
          })}
        </svg>
      </Figure>
      <Readout
        label="Computed values"
        items={[
          { term: "Latent cache width", value: n(f.latentWidth.value), note: "content + positional key" },
          { term: "Expanded cache width", value: n(f.expandedCache.value), note: "all head keys and values" },
          {
            term: "Scale",
            value: f.softmaxScale.value.toFixed(f.demo.radDigits),
            note: "YaRN-adjusted attention scale",
          },
        ]}
      />
      <Check question="Why must the positional key stay outside the shared content latent?">
        Its rotation depends on absolute position. The implementation expands content separately and rotates the
        positional slice directly: the content latent and the positional key are both produced before per-head
        expansion.
      </Check>
      <Takeaways
        items={[
          `Latent storage needs ${n(f.latentWidth.value)} scalars per token per layer; expanded reference storage needs ${n(f.expandedCache.value)}.`,
          "YaRN also changes the score scale; a frequency-only implementation would miss part of this forward pass.",
          "The shared positional key does not erase head-specific content comparisons.",
        ]}
      />
    </Section>
  );
}
