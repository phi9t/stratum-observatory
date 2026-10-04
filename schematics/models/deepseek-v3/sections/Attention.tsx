import {
  Section,
  OpGraph,
  Figure,
  AttentionGrid,
  Slider,
  Sentence,
  Readout,
  Sub,
  Prose,
  Check,
  Takeaways,
  n,
  shape,
  useUrlNumber,
  type Op,
} from "../../../kit";
import * as f from "../facts";
import { operation as op, chain, attentionRefs as refs, stream, linearDetail, normFormula } from "../graph";
const LAYOUT = { columns: [0, 1, 2, 3, 4, 5, 6, 7] };

export function Attention() {
  const [query, setQuery] = useUrlNumber("q", f.DEMO.query, 0, f.DEMO.tokens - 1);
  const d = f.hidden.value,
    qr = f.qRank.value,
    kr = f.kvRank.value,
    h = f.heads.value,
    r = f.rotary.value;
  const qShape = shape("B", h, "S", f.qDim.value),
    kvShape = shape("B", h, "S", f.nope.value + f.valueDim.value),
    scoreShape = shape("B", h, "S", "T");
  const q: Op[] = [
    op(
      "qa",
      "q_a projection",
      0,
      qr,
      linearDetail(
        "Compress each normalized token for the query path.",
        d,
        qr,
        refs,
        "q_lora_rank is present, so the direct q_proj branch is disabled. vLLM merges this projection with KV A.",
      ),
    ),
    op("qn", "query RMSNorm", 1, qr, {
      what: "Normalize the compressed query before expanding heads.",
      formula: normFormula,
      inputs: [["q_a", shape("B", "S", qr)]],
      output: ["q_latent", shape("B", "S", qr)],
      params: [["gain", shape(qr)]],
      refs,
    }),
    op(
      "qb",
      "q_b projection",
      2,
      f.qWidth.value,
      linearDetail("Expand the query latent into every head's channels.", qr, f.qWidth.value, refs),
    ),
    op("qh", "reshape heads", LAYOUT.columns[3], f.qWidth.value, {
      what: "Introduce separate attention heads; transpose sequence and head axes.",
      formula: "q = q.view(B,S,H,Dq).transpose(S,H)",
      inputs: [["q_flat", shape("B", "S", f.qWidth.value)]],
      output: ["q", qShape],
      refs,
    }),
    op("qs", "split query", LAYOUT.columns[4], f.qWidth.value, {
      what: "Separate nonpositional query channels from the channels that will rotate.",
      formula: "q_nope, q_pe = split(q, [Dnope, Drope])",
      inputs: [["q", qShape]],
      output: ["q_nope", shape("B", h, "S", f.nope.value)],
      refs,
      note: `The other output q_pe is ${shape("B", h, "S", r)}. Both are carried forward.`,
    }),
  ];
  const kv: Op[] = [
    op(
      "ka",
      "kv_a projection",
      0,
      f.kvAWidth.value,
      linearDetail("Produce a compressed latent and a separate shared positional key.", d, f.kvAWidth.value, refs),
      "high",
    ),
    op("ks", "split KV A", 1, f.kvAWidth.value, {
      what: "Keep the latent and positional key distinct; only the latent is normalized.",
      formula: "c, k_pe = split(kv_a, [Rkv, Drope])",
      inputs: [["kv_a", shape("B", "S", f.kvAWidth.value)]],
      output: ["c", shape("B", "S", kr)],
      refs,
      note: `k_pe is reshaped/transposed to ${shape("B", 1, "S", r)}; it is shared across all heads.`,
    }),
    op("kn", "KV RMSNorm", 2, kr, {
      what: "Normalize the latent before expanding nonpositional keys and values.",
      formula: normFormula,
      inputs: [["c", shape("B", "S", kr)]],
      output: ["c_norm", shape("B", "S", kr)],
      params: [["gain", shape(kr)]],
      refs,
    }),
    op(
      "kb",
      "kv_b projection",
      LAYOUT.columns[3],
      f.kvWidth.value,
      linearDetail(
        "Expand the same latent into head-specific keys and values.",
        kr,
        f.kvWidth.value,
        refs,
        "Absorbed decode reorders this expansion rather than applying it to every historical token.",
      ),
    ),
    op("kh", "reshape heads", LAYOUT.columns[4], f.kvWidth.value, {
      what: "Introduce the head axis and place sequence after it for matmul.",
      formula: "kv = kv.view(B,S,H,Dnope+Dv).transpose(S,H)",
      inputs: [["kv_flat", shape("B", "S", f.kvWidth.value)]],
      output: ["kv", kvShape],
      refs,
    }),
    op("kvsplit", "split K and V", LAYOUT.columns[5], f.kvWidth.value, {
      what: "Separate nonpositional key channels and values within each head.",
      formula: "k_nope, v = split(kv, [Dnope, Dv])",
      inputs: [["kv", kvShape]],
      output: ["v", shape("B", h, "S", f.valueDim.value)],
      refs,
      note: `k_nope has shape ${shape("B", h, "S", f.nope.value)}.`,
    }),
  ];
  const assemble: Op[] = [
    op(
      "freq",
      "YaRN cos / sin",
      0,
      r,
      {
        what: "Build trigonometric tables for absolute positions including the cache history.",
        formula: "angle = position · blended_frequency",
        inputs: [["position_ids", shape(1, "S")]],
        output: ["cos / sin", shape("T", r)],
        refs: [
          { source: "reference", symbol: "DeepseekV3YarnRotaryEmbedding" },
          { source: "publisher", symbol: "precompute_freqs_cis" },
        ],
        note: "The positional section plays the actual blended frequencies; these buffers are not trainable parameters.",
      },
      "instance",
    ),
    op(
      "rotate",
      "rotate Q and K",
      1,
      r,
      {
        what: "Rotate only positional channels; first permute adjacent pairs into the half-channel layout used by rotate_half.",
        formula: "rot(x) = x · cos + rotate_half(x) · sin",
        inputs: [
          ["q_pe", shape("B", h, "S", r)],
          ["k_pe", shape("B", 1, "S", r)],
        ],
        output: ["q_pe rotated", shape("B", h, "S", r)],
        refs: [...refs, { source: "reference", symbol: "apply_rotary_pos_emb" }],
        note: "The shared k_pe is rotated too, without expanding its head axis yet.",
      },
      "instance",
    ),
    op("joinq", "assemble Q", 2, f.qWidth.value, {
      what: "Recombine rotated and nonpositional query channels.",
      formula: "q = concat(q_nope, q_pe_rotated)",
      inputs: [
        ["q_nope", shape("B", h, "S", f.nope.value)],
        ["q_pe", shape("B", h, "S", r)],
      ],
      output: ["q", qShape],
      refs,
    }),
    op(
      "joink",
      "assemble K",
      2,
      f.qWidth.value,
      {
        what: "Broadcast the shared rotated key to each head and join it to that head's nonpositional key.",
        formula: "k = concat(k_nope, broadcast(k_pe_rotated))",
        inputs: [
          ["k_nope", shape("B", h, "S", f.nope.value)],
          ["k_pe", shape("B", 1, "S", r)],
        ],
        output: ["k", qShape],
        refs,
      },
      "high",
      1,
    ),
    op(
      "write",
      "cache write",
      LAYOUT.columns[3],
      f.eagerCacheWidth.value,
      {
        what: "Append this layer's expanded keys and values to persistent history.",
        formula: "cache[layer].append(k, v)",
        inputs: [
          ["k", qShape],
          ["v", shape("B", h, "S", f.valueDim.value)],
        ],
        output: ["cached K", shape("B", h, "T", f.qDim.value)],
        refs,
        note: "The reference calls the cache\u2019s update, which returns the concatenated history (the cache class itself is not pinned here). This and the next box expose those two effects of the same call. FlashAttention pads V before this call; compressed serving stores a different form.",
      },
      "instance",
      1,
      "state",
    ),
    op(
      "read",
      "cache read",
      LAYOUT.columns[4],
      f.eagerCacheWidth.value,
      {
        what: "Use the returned full history for this query sequence.",
        formula: "K_history, V_history = cache.update(k, v, layer)",
        inputs: [
          ["cache K", shape("B", h, "T", f.qDim.value)],
          ["cache V", shape("B", h, "T", f.valueDim.value)],
        ],
        output: ["K_history", shape("B", h, "T", f.qDim.value)],
        refs,
        note: `V_history is ${shape("B", h, "T", f.valueDim.value)}. Queries are fresh and are never stored.`,
      },
      "instance",
      1,
      "state",
    ),
  ];
  const attend: Op[] = [
    op("dot", "Q · Kᵀ", 0, h * f.DEMO.tokens, {
      what: "Compare each current query with every historical key independently per head.",
      formula: "scores = Q @ K_history.transpose(-2,-1)",
      inputs: [
        ["Q", qShape],
        ["K", shape("B", h, "T", f.qDim.value)],
      ],
      output: ["scores", scoreShape],
      refs,
    }),
    op("scale", "scale", 1, h * f.DEMO.tokens, {
      what: "Scale dot products using both query width and the YaRN amplitude adjustment.",
      formula: `scores *= ${n(f.attentionScale.value)}`,
      inputs: [["scores", scoreShape]],
      output: ["scaled", scoreShape],
      refs,
      note: `Exact scale = (1 + ${n(f.yarnCoefficient.value)} · ${n(f.mscaleAll.value)} · ln(${n(f.ropeFactor.value)}))² / sqrt(${n(f.qDim.value)}).`,
    }),
    op(
      "mask",
      "causal + padding",
      2,
      h * f.DEMO.tokens,
      {
        what: "Add the mask so forbidden future or padding positions cannot contribute.",
        formula: "scores += mask",
        inputs: [
          ["scores", scoreShape],
          ["mask", shape("B", 1, "S", "T")],
        ],
        output: ["masked", scoreShape],
        refs,
      },
      "instance",
    ),
    op("softmax", "softmax", LAYOUT.columns[3], h * f.DEMO.tokens, {
      what: "Normalize over keys in float32 and cast probabilities back to query dtype.",
      formula: "p = softmax(scores, axis=T, dtype=float32).to(Q.dtype)",
      inputs: [["masked", scoreShape]],
      output: ["p", scoreShape],
      refs,
    }),
    op("drop", "dropout identity", LAYOUT.columns[4], h * f.DEMO.tokens, {
      what: "The source calls dropout, but it changes no values in this eval checkpoint.",
      formula: `dropout(p, probability=${n(f.dropout.value)}, training=false) = p`,
      inputs: [["p", scoreShape]],
      output: ["p", scoreShape],
      refs,
    }),
    op("weighted", "P · V", LAYOUT.columns[5], f.outputWidth.value, {
      what: "Average historical values with the query's attention probabilities.",
      formula: "out = p @ V_history",
      inputs: [
        ["p", scoreShape],
        ["V", shape("B", h, "T", f.valueDim.value)],
      ],
      output: ["out", shape("B", h, "S", f.valueDim.value)],
      refs,
    }),
  ];
  const output: Op[] = [
    op("merge", "merge heads", 0, f.outputWidth.value, {
      what: "Move sequence before heads and flatten all head outputs into one token vector.",
      formula: "out = out.transpose(H,S).reshape(B,S,H·Dv)",
      inputs: [["out", shape("B", h, "S", f.valueDim.value)]],
      output: ["merged", shape("B", "S", f.outputWidth.value)],
      refs,
    }),
    op(
      "o",
      "o projection",
      1,
      d,
      linearDetail("Return the attention correction to the residual width.", f.outputWidth.value, d, refs),
    ),
  ];
  const topology: Op[] = [
    op(
      "input",
      "normalized stream",
      0,
      d,
      {
        what: "The block's attention norm supplies both projection paths.",
        formula: "x = input_layernorm(hidden)",
        inputs: [["hidden", stream]],
        output: ["x", stream],
        refs,
      },
      "merged",
      0,
      "tensor",
      stream,
    ),
    op("queries", "Query path", 1, f.qWidth.value, {
      what: "The Build queries graph compresses, normalizes and expands queries.",
      formula: "Q = Q_B(RMSNorm(Q_A(x)))",
      inputs: [["x", stream]],
      output: ["query channels", qShape],
      refs,
    }),
    op(
      "keyvalues",
      "KV path",
      1,
      f.kvWidth.value,
      {
        what: "The Build keys and values graph produces a normalized latent, shared positional key and expanded head-specific content.",
        formula: "K_content,V = KV_B(RMSNorm(c)); c,k_pe = split(KV_A(x))",
        inputs: [["x", stream]],
        output: ["KV", kvShape],
        refs,
      },
      "high",
      1,
    ),
    op("position", "Rotate + assemble", 2, f.qWidth.value, {
      what: "The positional graph rotates Q/K slices and assembles full queries and keys. Values bypass rotation.",
      formula: "Q,K = concat(content, rotated_position)",
      inputs: [
        ["queries", qShape],
        ["KV", kvShape],
      ],
      output: ["Q,K", qShape],
      refs,
    }),
    op(
      "history",
      "Cache history",
      2,
      f.eagerCacheWidth.value,
      {
        what: "The eager cache appends full keys and values and returns historical K/V.",
        formula: "K,V = cache.update(K,V,layer)",
        inputs: [
          ["K", qShape],
          ["V", shape("B", h, "S", f.valueDim.value)],
        ],
        output: ["K history", shape("B", h, "T", f.qDim.value)],
        refs,
      },
      "instance",
      1,
      "state",
    ),
    op("attend", "Attend over history", LAYOUT.columns[3], f.outputWidth.value, {
      what: "The Attend over history graph compares queries to cached keys, masks, normalizes and averages values.",
      formula: "A = softmax(mask(scale(Q Kᵀ))) V",
      inputs: [
        ["Q", qShape],
        ["K,V history", shape("B", h, "T", f.qDim.value)],
      ],
      output: ["head outputs", shape("B", h, "S", f.valueDim.value)],
      refs,
    }),
    op("output", "Merge + project", LAYOUT.columns[4], d, {
      what: "The Return to the stream graph merges heads and projects the correction back to residual width.",
      formula: "out = O(concat_heads(A))",
      inputs: [["head outputs", shape("B", h, "S", f.valueDim.value)]],
      output: ["attention correction", stream],
      refs,
    }),
  ];
  return (
    <Section
      id="attention"
      title="Attention through a latent bottleneck"
      source={[{ source: "reference", symbol: "DeepseekV3Attention.forward" }]}
      lede="Queries and KV use separate compressed paths. Both expand for the eager forward pass, but only a slice of each query and key carries rotary position information."
    >
      <OpGraph
        heading="How the attention paths connect"
        id="attention-topology"
        title="Attention topology"
        ops={topology}
        edges={[
          { from: "input", to: "queries" },
          { from: "input", to: "keyvalues" },
          { from: "queries", to: "position" },
          { from: "keyvalues", to: "position" },
          { from: "position", to: "history", kind: "state" },
          { from: "keyvalues", to: "history" },
          { from: "position", to: "attend" },
          { from: "history", to: "attend", kind: "state" },
          { from: "attend", to: "output" },
        ]}
        caption="Each box summarizes a detailed graph below. The query and KV paths share an input; rotary assembly joins them, and historical K/V meet fresh queries at attention."
      />
      <Prose>
        Start with the query path, then follow the parallel KV path. Both feed assembly; only keys and values persist in
        history.
      </Prose>
      <OpGraph heading="Build queries" id="query-op" title="Query path" ops={q} edges={chain(q)} />
      <Prose>
        The same normalized stream also feeds KV A. Its positional key bypasses the latent norm and expansion.
      </Prose>
      <OpGraph heading="Build keys and values" id="kv-op" title="Key and value path" ops={kv} edges={chain(kv)} />
      <Prose>
        The query split and KV split meet at the positional operation below. Assembly consumes both paths; the sequence
        of boxes follows execution order, with their actual dependencies named in the inspector.
      </Prose>
      <OpGraph
        heading="Rotate, assemble and keep history"
        id="assembly-op"
        title="Position and eager cache"
        ops={assemble}
        edges={[
          { from: "freq", to: "rotate", kind: "state" },
          { from: "rotate", to: "joinq" },
          { from: "rotate", to: "joink" },
          { from: "joink", to: "write", kind: "state" },
          { from: "write", to: "read", kind: "state" },
        ]}
      />
      <Prose>
        The returned keys and values span T positions; the current queries span S. The mask includes the offset into
        that history.
      </Prose>
      <OpGraph
        heading="Attend over history"
        caption={`Gauges count all heads; score width uses ${n(f.DEMO.tokens)} illustrated history positions. Actual shapes retain symbolic T.`}
        id="attend-op"
        title="Eager causal attention"
        ops={attend}
        edges={chain(attend)}
      />
      <Prose>
        Each head returns a value-width vector. The output projection mixes the merged head outputs, ready for the
        block's first residual addition.
      </Prose>
      <OpGraph
        heading="Return to the stream"
        id="output-op"
        title="Attention output"
        ops={output}
        edges={chain(output)}
      />
      <Sub>Who can read whom</Sub>
      <Slider label="Query position" value={query} min={0} max={f.DEMO.tokens - 1} onChange={setQuery} />
      <Sentence>
        Query {n(query)} can read {n(query + 1)} shown keys: itself and all earlier positions.
      </Sentence>
      <Figure
        label="Causal attention pattern"
        caption="The grid shows permitted connections, not learned attention probabilities. All heads and layers obey the same causal rule."
      >
        <AttentionGrid
          title="Causal mask"
          size={f.DEMO.tokens}
          query={query}
          onQuery={setQuery}
          cell={(q, k) => (k > q ? "masked" : k === q ? "self" : "attend")}
          legend={[
            { state: "attend", label: "earlier, readable" },
            { state: "self", label: "query itself" },
            { state: "masked", label: "future, masked" },
          ]}
        />
      </Figure>
      <Readout
        label="Attention widths"
        items={[
          { term: "Query / key", value: n(f.qDim.value), note: `${n(f.nope.value)} nonpositional + ${n(r)} rotary` },
          { term: "Value", value: n(f.valueDim.value) },
          { term: "Query latent", value: n(qr) },
          { term: "KV latent", value: n(kr) },
        ]}
      />
      <Check question="Does the shared positional key make all heads' complete keys identical?">
        No. Only the rotary slice is shared. KV B produces a separate nonpositional key for each head, so each query
        still compares against head-specific content.
      </Check>
      <Takeaways
        items={[
          `The query expands from ${n(qr)} to ${n(f.qWidth.value)} channels; KV expands from ${n(kr)} to ${n(f.kvWidth.value)}.`,
          "The eager reference's low-rank construction does not by itself produce a compressed cache.",
          "FlashAttention fuses scoring, masking and softmax; it pads the values before caching and removes that padding after attention.",
        ]}
      />
    </Section>
  );
}
