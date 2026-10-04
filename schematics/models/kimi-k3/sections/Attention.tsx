import {
  Section,
  Prose,
  Sub,
  Figure,
  AttentionGrid,
  Slider,
  Sentence,
  Takeaways,
  Check,
  OpGraph,
  useUrlNumber,
  n,
  shape,
  type Op,
} from "../../../kit";
import * as f from "../facts";
import { Graph, type Step } from "./graph";
const ref = [
  { source: "reference", symbol: "KimiMLAAttention.forward" },
  { source: "serving", symbol: "KimiK3MLAAttention.forward" },
];
export function Attention() {
  const [q, setQ] = useUrlNumber("q", f.DEMO.grid - 1, 0, f.DEMO.grid - 1);
  const qSteps: Step[] = [
    {
      id: "qa",
      label: "q_a_proj",
      width: f.qRank.value,
      what: "Compress the full stream into a query latent.",
      formula: "q_a = x W_qaᵀ",
      params: [["weight", shape(f.qRank.value, f.hidden.value)]],
    },
    {
      id: "qn",
      label: "latent RMSNorm",
      width: f.qRank.value,
      what: "Normalize the latent before expanding to heads.",
      formula: `q_n = q_a / sqrt(mean(q_a²) + ${n(f.normLatentEps.value)}) × gain`,
      params: [["gain", shape(f.qRank.value)]],
      note: "Publisher MLA latent norm uses the constructor default epsilon. SGLang base MLA uses config epsilon; numerical equivalence is not established.",
    },
    {
      id: "qb",
      label: "q_b_proj",
      width: f.heads.value * f.qDim.value,
      what: "Produce all query-head channels.",
      formula: "q = q_n W_qbᵀ",
      params: [["weight", shape(f.heads.value * f.qDim.value, f.qRank.value)]],
    },
    {
      id: "qheads",
      label: "reshape + split",
      width: f.qDim.value,
      what: "Introduce head and channel axes, split named channel groups. No rotary transform follows.",
      formula: "[B,S,H×Q] → [B,H,S,Q]; Q = q_pass ⊕ q_rot",
      output: shape("B", f.heads.value, "S", f.qDim.value),
      note: "q_rot is a historical name; use_nope is active and rotary_emb is None.",
    },
  ];
  const kvSteps: Step[] = [
    {
      id: "kva",
      label: "kv_a_proj",
      width: f.absorbedCacheWidth.value,
      what: "Produce a KV latent plus shared key channels.",
      formula: "[c,k_shared] = x W_kvaᵀ",
      params: [["weight", shape(f.absorbedCacheWidth.value, f.hidden.value)]],
    },
    {
      id: "split",
      label: "split latent / key",
      width: f.kvRank.value,
      what: "Keep the shared key channels aside while normalizing only the latent.",
      formula: `[${n(f.absorbedCacheWidth.value)}] → [${n(f.kvRank.value)}] ⊕ [${n(f.sharedKeyDim.value)}]`,
      note: "The shared-key branch bypasses latent RMSNorm and KV expansion.",
    },
    {
      id: "kvn",
      label: "latent RMSNorm",
      width: f.kvRank.value,
      what: "Normalize compressed content channels.",
      formula: `c_n = c / sqrt(mean(c²) + ${n(f.normLatentEps.value)}) × gain`,
      params: [["gain", shape(f.kvRank.value)]],
    },
    {
      id: "kvb",
      label: "kv_b_proj",
      width: f.heads.value * (f.nopeDim.value + f.vDim.value),
      what: "Expand normalized content into separate head-local key and value channels.",
      formula: "[k_pass,v] = c_n W_kvbᵀ",
      params: [["weight", shape(f.heads.value * (f.nopeDim.value + f.vDim.value), f.kvRank.value)]],
    },
    {
      id: "kvheads",
      label: "reshape + split",
      width: f.nopeDim.value + f.vDim.value,
      what: "Split each head into key-content and value slices.",
      formula: "[B,S,H×(K+V)] → [B,H,S,K] ⊕ [B,H,S,V]",
      output: shape("B", f.heads.value, "S", f.nopeDim.value + f.vDim.value),
    },
    {
      id: "concat",
      label: "broadcast + concat",
      width: f.qDim.value,
      what: "Broadcast the shared key slice to every head and join it to the key-content slice.",
      formula: "k = concat(k_pass, broadcast(k_shared))",
      input: shape("B", f.heads.value, "S", f.nopeDim.value),
      output: shape("B", f.heads.value, "S", f.qDim.value),
    },
    {
      id: "cache",
      label: "KV cache update",
      width: f.logicalCacheWidth.value,
      kind: "state",
      lane: "instance",
      what: "Append expanded keys and values at each full-attention layer.",
      formula: "K_cache ← concat_time(K_cache,k); V_cache ← concat_time(V_cache,v)",
      output: `K ${shape("B", f.heads.value, "T", f.qDim.value)}; V ${shape("B", f.heads.value, "T", f.vDim.value)}`,
      note: "This is publisher storage. The absorbed serving path caches the normalized latent and shared key instead.",
    },
  ];
  const attend: Step[] = [
    {
      id: "dot",
      label: "query · keyᵀ",
      width: f.qDim.value,
      what: "Compare each query with available cached or current keys.",
      formula: "scores = q Kᵀ",
      input: shape("B", f.heads.value, "S", f.qDim.value),
      output: shape("B", f.heads.value, "S", "T"),
    },
    {
      id: "scale",
      label: "scale",
      width: f.qDim.value,
      what: "Scale by the complete query width, including the unrotated shared-key slice.",
      formula: `scores × ${n(f.qDim.value)}^(-1/2)`,
      output: shape("B", f.heads.value, "S", "T"),
    },
    {
      id: "mask",
      label: "causal mask",
      width: f.qDim.value,
      lane: "instance",
      what: "Remove future positions from the readable set.",
      formula: "scores += causal_mask (0 or −∞)",
      output: shape("B", f.heads.value, "S", "T"),
      note: "Eager-equivalent arithmetic; the publisher always dispatches to FlashAttention, which does not add this mask in this form.",
    },
    {
      id: "softmax",
      label: "softmax",
      width: f.qDim.value,
      what: "Normalize scores over key positions; the eager function does this in float32.",
      formula: "P = softmax_key(scores)",
      output: shape("B", f.heads.value, "S", "T"),
      note: "Eager-equivalent arithmetic; the publisher always dispatches to FlashAttention.",
    },
    {
      id: "weighted",
      label: "weighted values",
      width: f.vDim.value,
      what: "Read a weighted sum of the values in each head.",
      formula: "o = P V",
      output: shape("B", f.heads.value, "S", f.vDim.value),
      note: "FlashAttention pads the value channels to query width and slices the output back. No extra learned parameters.",
    },
    {
      id: "merge",
      label: "merge heads",
      width: f.wide.value,
      what: "Put the head outputs back into a flat per-token vector.",
      formula: "[B,H,S,V] → [B,S,H×V]",
    },
  ];
  const output: Step[] = [
    {
      id: "gproj",
      label: "g_proj",
      width: f.wide.value,
      what: "Read the same normalized input stream to build a full-width output gate.",
      formula: "g = x W_gᵀ",
      params: [["weight", shape(f.wide.value, f.hidden.value)]],
    },
    {
      id: "sigmoid",
      label: "sigmoid",
      width: f.wide.value,
      what: "Bound the output gate per channel.",
      formula: "g ← sigmoid(g)",
    },
    {
      id: "multiply",
      label: "gate × output",
      width: f.wide.value,
      kind: "mul",
      what: "Gate the merged attended values before the final projection.",
      formula: "o ← attended × g",
      note: "The attended input comes from the preceding graph; this is an output gate, not attention probability.",
    },
    {
      id: "oproj",
      label: "o_proj",
      width: f.hidden.value,
      what: "Return the gated values to stream width.",
      formula: "y = o W_oᵀ",
      params: [["weight", shape(f.hidden.value, f.wide.value)]],
    },
  ];
  const topology: Op[] = [
    {
      id: "x",
      label: "normed stream",
      sub: n(f.hidden.value),
      width: f.hidden.value,
      kind: "tensor",
      col: 0,
      row: 0,
      detail: {
        what: "Both latent paths read the same stream.",
        output: ["x", shape("B", "S", f.hidden.value)],
        refs: ref,
      },
    },
    {
      id: "qpath",
      label: "query latent path",
      sub: n(f.qRank.value),
      width: f.qRank.value,
      lane: "high",
      col: 1,
      row: 0,
      detail: { what: "Compress, normalize and expand queries.", refs: ref },
    },
    {
      id: "kvpath",
      label: "KV latent path",
      sub: n(f.kvRank.value),
      width: f.kvRank.value,
      lane: "instance",
      col: 1,
      row: 1,
      detail: { what: "Build keys/values and update token cache.", refs: ref },
    },
    {
      id: "attend",
      label: "causal attention",
      sub: n(f.wide.value),
      width: f.wide.value,
      lane: "high",
      col: 2,
      row: 0,
      detail: { what: "Queries read the assembled keys and values.", refs: ref },
    },
    {
      id: "gatepath",
      label: "sigmoid gate",
      sub: n(f.wide.value),
      width: f.wide.value,
      col: 2,
      row: 1,
      detail: { what: "Full-rank gate reads the original input.", refs: ref },
    },
    {
      id: "out",
      label: "gate + project",
      sub: n(f.hidden.value),
      width: f.hidden.value,
      col: 3,
      row: 0,
      detail: { what: "Multiply attention output by gate and return to stream width.", refs: ref },
    },
  ];
  const summaries: Record<
    string,
    { formula: string; inputs: readonly (readonly [string, string])[]; output: readonly [string, string] }
  > = {
    x: {
      formula: "x = input_layernorm(depth_mix)",
      inputs: [["depth mixture", shape("B", "S", f.hidden.value)]],
      output: ["x", shape("B", "S", f.hidden.value)],
    },
    qpath: {
      formula: "q = reshape(q_b(RMSNorm(q_a(x))))",
      inputs: [["x", shape("B", "S", f.hidden.value)]],
      output: ["q", shape("B", f.heads.value, "S", f.qDim.value)],
    },
    kvpath: {
      formula: "K,V = append_cache(expand_normalized_latent_and_shared_key(x))",
      inputs: [["x", shape("B", "S", f.hidden.value)]],
      output: [
        "K,V",
        `K ${shape("B", f.heads.value, "T", f.qDim.value)}; V ${shape("B", f.heads.value, "T", f.vDim.value)}`,
      ],
    },
    attend: {
      formula: "o = softmax(qKᵀ / sqrt(Q) + causal_mask) V",
      inputs: [
        ["q", shape("B", f.heads.value, "S", f.qDim.value)],
        ["K,V", "cached head tensors"],
      ],
      output: ["o", shape("B", "S", f.wide.value)],
    },
    gatepath: {
      formula: "g = sigmoid(g_proj(x))",
      inputs: [["x", shape("B", "S", f.hidden.value)]],
      output: ["g", shape("B", "S", f.wide.value)],
    },
    out: {
      formula: "y = o_proj(o × g)",
      inputs: [["o,g", shape("B", "S", f.wide.value)]],
      output: ["y", shape("B", "S", f.hidden.value)],
    },
  };
  const detailedTopology = topology.map((op) => ({ ...op, detail: { ...op.detail, ...summaries[op.id] } }));
  return (
    <Section
      id="attention"
      title="Full attention, without text rotation"
      source={ref}
      lede="Queries and key/value content pass through different narrow latents. Shared key channels keep their historical rotary names, but this checkpoint never rotates them."
    >
      <OpGraph
        id="mla-topology"
        title="Parallel latent paths"
        heading="The paths and where they meet"
        ops={detailedTopology}
        edges={[
          { from: "x", to: "qpath" },
          { from: "x", to: "kvpath" },
          { from: "x", to: "gatepath" },
          { from: "qpath", to: "attend" },
          { from: "kvpath", to: "attend" },
          { from: "attend", to: "out" },
          { from: "gatepath", to: "out" },
        ]}
      />
      <Prose>
        The query path ends at head-local queries; the KV path builds both key content and values. Only their attention
        scores exchange information across token positions.
      </Prose>
      <Graph
        id="mla-query"
        title="Query construction"
        heading="Build queries"
        steps={qSteps}
        source="KimiMLAAttention.forward"
        serving="KimiK3MLAAttention.__init__"
      />
      <Prose>
        In parallel, the KV projection keeps a shared key slice aside while expanding normalized latent content.
      </Prose>
      <Graph
        id="mla-kv"
        title="Key and value construction"
        heading="Build keys, values and cache"
        steps={kvSteps}
        edges={[
          ...kvSteps.slice(1).map((s, i) => ({ from: kvSteps[i].id, to: s.id })),
          { from: "split", to: "concat", kind: "skip" },
          { from: "kvheads", to: "cache", kind: "skip" },
        ]}
        source="KimiMLAAttention.forward"
        serving="KimiK3MLAAttention.__init__"
      />
      <Prose>
        Current and cached keys participate in the same causal score calculation. The query slice called q_rot remains
        unrotated. The graph below is eager-equivalent arithmetic; the publisher always dispatches to FlashAttention.
      </Prose>
      <Graph
        id="mla-attend"
        title="Causal attention arithmetic"
        heading="Attend"
        steps={attend}
        source="eager_attention_forward"
        serving="KimiK3MLAAttention.forward"
      />
      <Prose>
        The output gate is computed from the normalized stream independently of the attention probabilities.
      </Prose>
      <Graph
        id="mla-output"
        title="Gated output"
        heading="Gate and return"
        steps={output}
        source="KimiMLAAttention.forward"
        serving="KimiK3MLAAttention.__init__"
      />
      <Sub>What the causal mask permits</Sub>
      <Slider label="Query position" min={0} max={f.DEMO.grid - 1} value={q} onChange={setQ} />
      <Sentence>
        Query {n(q)} can read positions through {n(q)}; {n(q + 1)} of the displayed keys are readable. This shows the
        mask, not learned probabilities.
      </Sentence>
      <Figure label="Causal full-attention pattern">
        <AttentionGrid
          title="Allowed query/key pairs"
          size={f.DEMO.grid}
          query={q}
          onQuery={setQ}
          cell={(a, b) => (b > a ? "masked" : a === b ? "self" : "attend")}
          legend={[
            { state: "attend", label: "earlier key" },
            { state: "self", label: "self" },
            { state: "masked", label: "future key" },
          ]}
        />
      </Figure>
      <Check question="Does the name qk_rope_head_dim establish that text positions are rotated?">
        No. The forward path concatenates these channels without a rotation. The publisher sets rotary_emb to None and
        SGLang passes skip_rope=True.
      </Check>
      <Takeaways
        items={[
          `Publisher expanded KV storage has ${n(f.logicalCacheWidth.value)} scalars per token in each full-attention layer.`,
          `The absorbed serving representation needs ${n(f.absorbedCacheWidth.value)} cached scalars instead; attention kernels still compute the same conceptual keys and values.`,
          `Latent norm epsilon differs between the inspected publisher and serving constructors; component parity does not settle their trained-output equivalence.`,
        ]}
      />
    </Section>
  );
}
