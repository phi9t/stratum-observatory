import {
  Section,
  OpGraph,
  Prose,
  Figure,
  AttentionGrid,
  RopeDials,
  Slider,
  Sentence,
  Readout,
  Sub,
  Check,
  Takeaways,
  n,
  shape,
  useUrlNumber,
  type Op,
  type Edge,
} from "../../../kit";
import * as f from "../facts";
export function Attention({ layer }: { layer: number }) {
  const [query, setQuery] = useUrlNumber("q", f.DISPLAY_TOKENS - 1, 0, f.DISPLAY_TOKENS - 1);
  const [position, setPosition] = useUrlNumber("pos", f.DEMO_CONTEXT, 0, f.positions.value - 1);
  const s = shape("B", "S", f.hidden.value),
    q = shape("B", f.heads.value, "S", f.headDim.value),
    k = shape("B", f.kvHeads.value, "S", f.headDim.value),
    // Before the transpose: the head axis sits after the sequence axis while the norm runs.
    qPre = shape("B", "S", f.heads.value, f.headDim.value),
    kPre = shape("B", "S", f.kvHeads.value, f.headDim.value),
    qt = shape("B", f.heads.value, "S", "T");
  const refs = [
    { source: "reference", symbol: "Qwen3MoeAttention.forward" },
    { source: "serving", symbol: "Qwen3MoeAttention.forward" },
  ];
  const make = (
    id: string,
    label: string,
    width: number,
    out: string,
    what: string,
    formula: string,
    params: readonly (readonly [string, string])[] = [],
    inputs: readonly (readonly [string, string])[] = [["hidden", s]],
  ): Omit<Op, "col" | "row"> => ({
    id,
    label,
    width,
    sub: n(width),
    lane: id.startsWith("q") ? "high" : id.startsWith("k") || id.startsWith("v") ? "low" : "instance",
    detail: {
      what,
      formula,
      params,
      inputs,
      output: [id, out],
      refs,
      note: `Parameter namespace model.layers.${n(layer)}.self_attn. All linear weights are biasless. Float32 norm/softmax accumulation is cast back to the input dtype in upstream.`,
    },
  });
  const topology: Op[] = [
    {
      ...make(
        "input",
        "normed stream",
        f.hidden.value,
        s,
        "Input from the block's first RMSNorm.",
        "x = input_layernorm(hidden)",
      ),
      lane: "merged",
      kind: "tensor",
      col: 0,
      row: 1,
    },
    {
      ...make(
        "queries",
        "query path",
        f.qWidth.value,
        q,
        "Project, head-normalize and rotate queries.",
        "Q = RoPE(q_norm(q_proj(x)))",
      ),
      lane: "high",
      col: 1,
      row: 0,
    },
    {
      ...make(
        "kv",
        "key / value path",
        f.kvWidth.value,
        k,
        "Project keys and values; normalize and rotate only keys.",
        "K = RoPE(k_norm(k_proj(x))); V = v_proj(x)",
      ),
      lane: "low",
      col: 1,
      row: 2,
    },
    {
      ...make(
        "cache",
        "KV cache",
        f.cachePerTokenLayer.value,
        shape("B", f.kvHeads.value, "T", f.headDim.value),
        "Retain post-rotation keys and raw values, then read the past plus current token.",
        "cache.append(K,V); K,V = cache.read()",
      ),
      kind: "state",
      col: 2,
      row: 2,
    },
    {
      ...make(
        "attend",
        "attend + mix",
        f.qWidth.value,
        shape("B", "S", f.qWidth.value),
        "Compare queries with all allowed cached keys and mix the corresponding values.",
        "softmax(mask(QKᵀ · scale))V",
      ),
      lane: "high",
      col: 3,
      row: 1,
    },
    {
      ...make(
        "output",
        "output projection",
        f.hidden.value,
        s,
        "Return the wider attention result to the residual width.",
        "y = mixed · W_oᵀ",
      ),
      lane: "high",
      col: 4,
      row: 1,
    },
  ];
  const topEdges: Edge[] = [
    { from: "input", to: "queries" },
    { from: "input", to: "kv" },
    { from: "kv", to: "cache", kind: "state" },
    { from: "queries", to: "attend" },
    { from: "cache", to: "attend", kind: "state" },
    { from: "attend", to: "output" },
  ];
  const build: Op[] = [
    {
      ...make(
        "q_proj",
        "q projection",
        f.qWidth.value,
        shape("B", "S", f.qWidth.value),
        "Create the wider query vector.",
        "q_flat = x · W_qᵀ",
        [["q_proj.weight", shape(f.qWidth.value, f.hidden.value)]],
      ),
      col: 0,
      row: 0,
    },
    {
      ...make(
        "q_heads",
        "q to heads",
        f.qWidth.value,
        qPre,
        "Introduce an independent query-head axis.",
        "view(B, S, Hq, D); the transpose comes after the head norm",
        [],
        [["q_flat", shape("B", "S", f.qWidth.value)]],
      ),
      col: 1,
      row: 0,
    },
    {
      ...make(
        "q_norm",
        "q RMSNorm",
        f.qWidth.value,
        q,
        "Normalize each head with the same learned head-width gain.",
        `q / sqrt(mean(q², D) + ${String(f.normEps.value)}) · g_q, then transpose(1, 2)`,
        [["q_norm.weight", shape(f.headDim.value)]],
        [["q", qPre]],
      ),
      col: 2,
      row: 0,
    },
    {
      ...make(
        "k_proj",
        "k projection",
        f.kvWidth.value,
        shape("B", "S", f.kvWidth.value),
        "Create fewer key channels than query channels.",
        "k_flat = x · W_kᵀ",
        [["k_proj.weight", shape(f.kvWidth.value, f.hidden.value)]],
      ),
      col: 0,
      row: 1,
    },
    {
      ...make(
        "k_heads",
        "k to heads",
        f.kvWidth.value,
        kPre,
        "Introduce the smaller KV-head axis.",
        "view(B, S, Hkv, D); the transpose comes after the head norm",
        [],
        [["k_flat", shape("B", "S", f.kvWidth.value)]],
      ),
      col: 1,
      row: 1,
    },
    {
      ...make(
        "k_norm",
        "k RMSNorm",
        f.kvWidth.value,
        k,
        "Normalize keys per head before applying position information.",
        `k / sqrt(mean(k², D) + ${String(f.normEps.value)}) · g_k, then transpose(1, 2)`,
        [["k_norm.weight", shape(f.headDim.value)]],
        [["k", kPre]],
      ),
      col: 2,
      row: 1,
    },
    {
      ...make(
        "v_proj",
        "v projection",
        f.kvWidth.value,
        shape("B", "S", f.kvWidth.value),
        "Create value vectors without a norm or position rotation.",
        "v_flat = x · W_vᵀ",
        [["v_proj.weight", shape(f.kvWidth.value, f.hidden.value)]],
      ),
      col: 0,
      row: 2,
    },
    {
      ...make(
        "v_heads",
        "v to heads",
        f.kvWidth.value,
        k,
        "Arrange values on the KV-head axis.",
        "view(B,S,Hkv,D); transpose S,Hkv",
        [],
        [["v_flat", shape("B", "S", f.kvWidth.value)]],
      ),
      col: 1,
      row: 2,
    },
  ];
  const buildEdges: Edge[] = [
    { from: "q_proj", to: "q_heads" },
    { from: "q_heads", to: "q_norm" },
    { from: "k_proj", to: "k_heads" },
    { from: "k_heads", to: "k_norm" },
    { from: "v_proj", to: "v_heads" },
  ];
  const pos: Op[] = [
    {
      ...make(
        "freq",
        "position phases",
        f.headDim.value,
        shape("B", "S", f.headDim.value),
        "Compute shared phases; duplicate the frequency half to match rotate_half.",
        `angle(p,i) = p · ${n(f.theta.value)}^(−2i/${n(f.headDim.value)})`,
        [],
        [["position_ids", shape("B", "S")]],
      ),
      col: 0,
      row: 0,
    },
    {
      ...make(
        "trig",
        "cos / sin",
        f.headDim.value,
        shape("B", "S", f.headDim.value),
        "Convert phases to rotation coefficients once for the entire decoder.",
        "cos = cos(angle); sin = sin(angle)",
      ),
      col: 1,
      row: 0,
    },
    {
      ...make(
        "q_rope",
        "rotate q",
        f.qWidth.value,
        q,
        "Rotate the normalized query in split-half channel pairs.",
        "q′ = q · cos + rotate_half(q) · sin",
        [],
        [
          ["q", q],
          ["cos,sin", shape("B", "S", f.headDim.value)],
        ],
      ),
      col: 2,
      row: 0,
    },
    {
      ...make(
        "k_rope",
        "rotate k",
        f.kvWidth.value,
        k,
        "Rotate keys using the same position phases as queries.",
        "k′ = k · cos + rotate_half(k) · sin",
        [],
        [
          ["k", k],
          ["cos,sin", shape("B", "S", f.headDim.value)],
        ],
      ),
      col: 2,
      row: 1,
    },
    {
      ...make(
        "write",
        "append K / V",
        f.cachePerTokenLayer.value,
        k,
        "Write new rotated keys and unrotated values into this layer's cache.",
        "cache[layer].append(k′,v)",
        [],
        [
          ["new K", k],
          ["new V", k],
        ],
      ),
      kind: "state",
      col: 3,
      row: 1,
    },
    {
      ...make(
        "read",
        "read past + now",
        f.cachePerTokenLayer.value,
        shape("B", f.kvHeads.value, "T", f.headDim.value),
        "Read complete keys and values for all retained positions; queries remain current.",
        "K,V = concat(past,current)",
        [],
        [["cache", shape("B", f.kvHeads.value, "T", f.headDim.value)]],
      ),
      kind: "state",
      col: 4,
      row: 1,
    },
  ];
  const posEdges: Edge[] = [
    { from: "freq", to: "trig" },
    { from: "trig", to: "q_rope" },
    { from: "trig", to: "k_rope" },
    { from: "k_rope", to: "write", kind: "state" },
    { from: "write", to: "read", kind: "state" },
  ];
  pos.forEach((o, index) => {
    pos[index] = {
      ...o,
      detail: {
        ...o.detail,
        refs: [
          ...refs,
          { source: "reference", symbol: "Qwen3MoeRotaryEmbedding.forward" },
          { source: "reference", symbol: "apply_rotary_pos_emb" },
          { source: "rope-utils", symbol: "_compute_default_rope_parameters" },
        ],
      },
    };
  });
  const mix: Op[] = [
    {
      ...make(
        "repeat",
        "repeat KV",
        f.qWidth.value,
        shape("B", f.heads.value, "T", f.headDim.value),
        "Eager attention repeats each KV head for its query group; storage stays compact.",
        `K_rep,V_rep = repeat_kv(K,V,${n(f.kvReuse.value)})`,
        [],
        [["K,V", shape("B", f.kvHeads.value, "T", f.headDim.value)]],
      ),
      lane: "low",
      col: 0,
      row: 0,
    },
    {
      ...make(
        "dot",
        "q · kᵀ",
        f.heads.value,
        qt,
        "Compare each current query with retained keys in its group.",
        "scores = Q @ K_repᵀ",
        [],
        [
          ["Q", q],
          ["K_rep", shape("B", f.heads.value, "T", f.headDim.value)],
        ],
      ),
      lane: "high",
      col: 1,
      row: 0,
    },
    {
      ...make(
        "scale",
        "scale",
        f.heads.value,
        qt,
        "Apply head-width scaling before the mask and softmax.",
        `${n(f.headDim.value)}^(−0.5) · scores`,
        [],
        [["scores", qt]],
      ),
      lane: "high",
      col: 2,
      row: 0,
    },
    {
      ...make(
        "mask",
        "causal mask",
        f.heads.value,
        qt,
        "Add a large negative value to future or padded positions.",
        "scores += mask; allowed iff key_position ≤ query_position",
        [],
        [
          ["scaled", qt],
          ["mask", shape("B", 1, "S", "T")],
        ],
      ),
      col: 3,
      row: 0,
    },
    {
      ...make(
        "weights",
        "softmax",
        f.heads.value,
        qt,
        "Normalize independently over retained keys for each query head; dropout disabled.",
        "P = exp(scores) / Σ_T exp(scores)",
        [],
        [["masked", qt]],
      ),
      lane: "high",
      col: 4,
      row: 0,
    },
    {
      ...make(
        "mixed",
        "P · V",
        f.qWidth.value,
        q,
        "Mix values using each query head's own weights.",
        "mixed = P @ V_rep",
        [],
        [
          ["P", qt],
          ["V_rep", shape("B", f.heads.value, "T", f.headDim.value)],
        ],
      ),
      lane: "high",
      col: 5,
      row: 0,
    },
  ];
  mix.forEach((o, index) => {
    mix[index] = {
      ...o,
      detail: {
        ...o.detail,
        refs: [
          ...refs,
          { source: "reference", symbol: "eager_attention_forward" },
          { source: "reference", symbol: "repeat_kv" },
        ],
      },
    };
  });
  const out: Op[] = [
    {
      ...make(
        "merge",
        "merge heads",
        f.qWidth.value,
        shape("B", "S", f.qWidth.value),
        "Move sequence before heads and concatenate query-head outputs.",
        "transpose(B,Hq,S,D) → (B,S,Hq,D); reshape",
        [],
        [["mixed", q]],
      ),
      lane: "high",
      col: 0,
      row: 0,
    },
    {
      ...make(
        "o_proj",
        "o projection",
        f.hidden.value,
        s,
        "Project the expanded attention channels back to the residual stream.",
        "y = mixed_flat · W_oᵀ",
        [["o_proj.weight", shape(f.hidden.value, f.qWidth.value)]],
        [["mixed_flat", shape("B", "S", f.qWidth.value)]],
      ),
      lane: "high",
      col: 1,
      row: 0,
    },
  ];
  return (
    <Section
      id="attention"
      title="Attention: wide queries, narrow memory"
      source={refs}
      lede={
        <>
          Each group of {n(f.kvReuse.value)} query heads shares a key head and a value head. Learned per-head norms come
          before position rotations; only keys and values survive into the next decode step.
        </>
      }
    >
      <OpGraph
        id="attentionTopology"
        title="Attention topology"
        heading="The paths and where they meet"
        ops={topology}
        edges={topEdges}
      />
      <Prose>
        The parallel paths below begin at the same normalized stream. A head reshape changes the meaning of the axes
        without changing the number of scalars.
      </Prose>
      <OpGraph
        id="projectOp"
        title="Q K V projection and per-head norm"
        heading="Build queries, keys and values"
        ops={build}
        edges={buildEdges}
      />
      <Prose>
        Position phases are shared across layers, but each layer rotates its own newly projected queries and keys. The
        cache never stores a query.
      </Prose>
      <OpGraph id="positionOp" title="RoPE and cache update" heading="Position and retain" ops={pos} edges={posEdges} />
      <Prose>
        With retained keys and values available, the eager reference repeats KV only for the attention calculation. A
        serving kernel can read the same compact heads directly.
      </Prose>
      <OpGraph
        id="mixOp"
        title="Causal attention arithmetic"
        heading="Compare and mix"
        ops={mix}
        edges={mix.slice(1).map((o, i) => ({ from: mix[i].id, to: o.id }))}
      />
      <Prose>
        The mixed tensor still has the wider query-head dimension. This final seam returns it to the block's stream.
      </Prose>
      <OpGraph
        id="outputOp"
        title="Merge and output projection"
        heading="Return to the stream"
        ops={out}
        edges={[{ from: "merge", to: "o_proj" }]}
      />
      <Sub>Which positions are reachable?</Sub>
      <Slider label="Query position" value={query} onChange={setQuery} min={0} max={f.DISPLAY_TOKENS - 1} />
      <Sentence>
        Query {n(query)} reads {n(query + 1)} of the first {n(f.DISPLAY_TOKENS)} positions. Every head uses this same
        causal boundary.
      </Sentence>
      <Figure
        label="Causal reachability"
        caption="This is a permission map, not learned attention weights. The diagonal is allowed; future columns are masked."
      >
        <AttentionGrid
          title="Full causal attention"
          size={f.DISPLAY_TOKENS}
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
      <Sub>Position rotates channel pairs</Sub>
      <Slider label="Rotary position" value={position} onChange={setPosition} min={0} max={f.positions.value - 1} />
      <Sentence>
        At absolute position {n(position)}, the shown frequency pairs rotate at different rates. Channels pair across
        the head's halves; values are not rotated.
      </Sentence>
      <Figure
        label="RoPE frequencies"
        caption={`Base ${n(f.theta.value)}, full rotary width ${n(f.headDim.value)}. These dials show coefficients, not trained vectors; no scaling override is configured.`}
      >
        <RopeDials
          title="Checkpoint rotary phases"
          pairs={f.ROPE_PAIRS}
          position={position}
          theta={f.theta.value}
          rotaryDim={f.headDim.value}
        />
      </Figure>
      <Readout
        label="Attention dimensions"
        items={[
          { term: "Queries", value: n(f.qWidth.value), note: `${n(f.heads.value)} × ${n(f.headDim.value)}` },
          { term: "Stored K or V", value: n(f.kvWidth.value), note: `${n(f.kvHeads.value)} × ${n(f.headDim.value)}` },
          {
            term: "Head norm parameters",
            value: n(f.qkNormCount.value * f.headDim.value),
            note: "gains shared across heads",
          },
        ]}
      />
      <Check question="Would caching keys before RoPE produce the same decode path?">
        Not without rotating all retained keys again when read. The reference rotates new keys once and stores the
        rotated form; retained values have no rotation.
      </Check>
      <Takeaways
        items={[
          "Query-head normalization is shared across heads but operates separately on each head vector.",
          <>
            The logical KV payload is {n(f.kvReuse.value)} times smaller than separate KV per query head; the explicit
            eager repetition does not change what is retained.
          </>,
        ]}
      />
    </Section>
  );
}
