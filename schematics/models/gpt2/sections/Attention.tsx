import { AttentionGrid, Check, Figure, n, OpGraph, Readout, Section, Sentence, shape, Slider, Sub, Takeaways, useUrlNumber, type Edge, type Op } from "../../../kit";
import * as f from "../facts";
import { useLayer } from "./Stack";

const LAYOUT = { grid: 24, startQuery: 9 };

export function Attention() {
  const [layer] = useLayer();
  const [query, setQuery] = useUrlNumber("q", LAYOUT.startQuery, 0, LAYOUT.grid - 1);
  const d = f.hidden.value;
  const h = f.heads.value;
  const hd = f.headDim.value;
  const x = shape("B", "S", d);
  const perHead = shape("B", h, "S", hd);
  const scores = shape("B", h, "S", "S");
  const fwd = { source: "reference", symbol: "GPT2Attention.forward" };
  const eager = { source: "reference", symbol: "eager_attention_forward" };
  const serve = { source: "serving", symbol: "GPT2Attention.forward" };
  const ops: Op[] = [
    { id: "a-in", width: d, label: "normed stream", sub: x, kind: "tensor", col: 0, row: 0,
      detail: { what: "Output of ln_1: the only thing attention reads.", output: ["hidden_states", x], refs: [fwd] } },
    { id: "a-qkv", width: f.qkvWidth.value, label: "c_attn", sub: `${n(d)} → ${n(f.qkvWidth.value)}`, lane: "high", col: 1, row: 0,
      detail: { what: "One projection computes queries, keys and values together.", formula: "qkv = x @ W + b",
        inputs: [["x", x]], output: ["qkv", shape("B", "S", f.qkvWidth.value)],
        params: [["weight", shape(d, f.qkvWidth.value)], ["bias", shape(f.qkvWidth.value)]],
        refs: [fwd, { source: "utils", symbol: "Conv1D" }, serve],
        note: "Conv1D is a linear layer whose weight is stored [in, out]. vllm loads it into a fused QKVParallelLinear and transposes the weight on load." } },
    { id: "a-split", width: d, label: "split ×3", sub: `3 × ${n(d)}`, lane: "high", col: 2, row: 0,
      detail: { what: "Cuts the joint projection into query, key and value, each as wide as the stream.", formula: "q, k, v = qkv.split(embed_dim, dim=2)",
        inputs: [["qkv", shape("B", "S", f.qkvWidth.value)]], output: ["q, k, v", `3 × ${x}`], refs: [fwd, serve] } },
    { id: "a-heads", width: hd, label: "to heads", sub: `${n(h)} × ${n(hd)}`, lane: "high", col: 3, row: 0,
      detail: { what: "Reshapes each of q, k, v so that every head gets its own slice of the width.", formula: "view(B, S, heads, head_dim).transpose(1, 2)",
        inputs: [["q, k, v", `3 × ${x}`]], output: ["q, k, v", `3 × ${perHead}`], refs: [fwd] } },
    { id: "a-cache", width: hd, label: "KV cache", sub: `2 × ${shape(h, "S", hd)}`, kind: "state", lane: "instance", col: 3, row: 1,
      detail: { what: "When decoding, this layer's keys and values for earlier positions are kept, so only the new position is projected and the stored ones are prepended.",
        formula: "k, v = past_key_values.update(k, v, layer_idx)", output: ["k, v", `2 × ${perHead}`], refs: [fwd],
        note: `Layer ${n(layer)} owns its own entry in the cache. vllm keeps this cache inside its Attention layer instead of passing it through forward.` } },
    { id: "a-score", label: "q · kᵀ", sub: scores, lane: "high", col: 4, row: 0,
      detail: { what: "Dot product of every query with every key, per head: how much each position wants each other position.", formula: "attn_weights = q @ k.transpose(-1, -2)",
        inputs: [["q", perHead], ["k", perHead]], output: ["attn_weights", scores], refs: [eager] } },
    { id: "a-scale", label: "scale", sub: `× ${n(hd)}^-0.5`, lane: "high", col: 5, row: 0,
      detail: { what: "Keeps the scores' spread independent of head width.", formula: `attn_weights *= head_dim ** -0.5   (head_dim = ${n(hd)})`,
        inputs: [["attn_weights", scores]], output: ["attn_weights", scores], refs: [{ source: "reference", symbol: "GPT2Attention" }, eager],
        note: "The code can also divide by layer_idx + 1; this checkpoint's config leaves scale_attn_by_inverse_layer_idx off, and vllm asserts that it is off." } },
    { id: "a-mask", label: "causal mask", sub: "+ (0 | −inf)", lane: "instance", col: 6, row: 0,
      detail: { what: "Adds a very negative number wherever a query would look at a later position, so those scores vanish in the softmax.", formula: "attn_weights = attn_weights + attention_mask",
        inputs: [["attn_weights", scores], ["attention_mask", shape("B", 1, "S", "S")]], output: ["attn_weights", scores], refs: [eager] } },
    { id: "a-softmax", label: "softmax", sub: "over keys", lane: "high", col: 7, row: 0,
      detail: { what: "Turns each query's scores into weights that sum to one over the positions it may read.", formula: "attn_weights = softmax(attn_weights, dim=-1)",
        inputs: [["attn_weights", scores]], output: ["attn_weights", scores], refs: [eager], note: "Dropout on these weights applies in training only." } },
    { id: "a-mix", width: hd, label: "weights · v", sub: perHead, lane: "high", col: 8, row: 0,
      detail: { what: "Each position's output is the weighted average of the values it attended to.", formula: "attn_output = attn_weights @ v",
        inputs: [["attn_weights", scores], ["v", perHead]], output: ["attn_output", perHead], refs: [eager] } },
    { id: "a-merge", width: d, label: "merge heads", sub: x, lane: "high", col: 9, row: 0,
      detail: { what: "Puts the heads' slices back side by side.", formula: "transpose(1, 2).reshape(B, S, heads * head_dim)",
        inputs: [["attn_output", perHead]], output: ["attn_output", x], refs: [eager, fwd] } },
    { id: "a-proj", width: d, label: "c_proj", sub: `${n(d)} → ${n(d)}`, lane: "high", col: 10, row: 0,
      detail: { what: "A final projection lets the heads' results mix before they are added to the stream.", formula: "attn_output = attn_output @ W + b",
        inputs: [["attn_output", x]], output: ["attn_output", x], params: [["weight", shape(d, d)], ["bias", shape(d)]], refs: [fwd, serve],
        note: "Residual dropout follows in training only." } },
  ];
  const edges: Edge[] = [
    { from: "a-in", to: "a-qkv" }, { from: "a-qkv", to: "a-split" }, { from: "a-split", to: "a-heads" }, { from: "a-heads", to: "a-score" },
    { from: "a-heads", to: "a-cache", kind: "state" }, { from: "a-cache", to: "a-score", kind: "state" },
    { from: "a-score", to: "a-scale" }, { from: "a-scale", to: "a-mask" }, { from: "a-mask", to: "a-softmax" }, { from: "a-softmax", to: "a-mix" },
    { from: "a-mix", to: "a-merge" }, { from: "a-merge", to: "a-proj" },
  ];
  const visible = query + 1;
  return (
    <Section
      id="attention"
      lede="The one place where positions exchange information. Twelve heads each compare every position with every earlier one and average what they find."
      source={[{ source: "reference", symbol: "GPT2Attention" }, { source: "reference", symbol: "eager_attention_forward" }]}
      title="Self-attention"
    >
      <OpGraph edges={edges} id="attn" ops={ops} title={`GPT-2 self-attention, layer ${n(layer)}`} />
      <Sub>Who can read whom</Sub>
      <Slider label="Query position" max={LAYOUT.grid - 1} min={0} onChange={setQuery} value={query} valueText={`position ${n(query)}`} />
      <Sentence>
        Position {n(query)} reads positions 0 to {n(query)}: itself and everything before it, nothing after.
      </Sentence>
      <Figure caption="Rows are queries, columns are keys. The pattern is the same in every head and every layer; only the learned weights inside the allowed cells differ." label="Causal mask drawing">
        <AttentionGrid
          cell={(q, k) => (k > q ? "masked" : k === q ? "self" : "attend")}
          legend={[{ state: "attend", label: "earlier position, readable" }, { state: "self", label: "the position itself" }, { state: "masked", label: "later position, masked" }]}
          onQuery={setQuery}
          query={query}
          size={LAYOUT.grid}
          title="Causal attention mask"
        />
      </Figure>
      <Readout
        items={[
          { term: "Readable keys", value: n(visible), note: `of the first ${n(LAYOUT.grid)} positions` },
          { term: "Heads", value: n(h), note: `each ${n(hd)} wide` },
          { term: "Scores per head", value: shape("S", "S"), note: "one per query-key pair" },
        ]}
        label="Attention readout"
      />
      <Check question="The joint projection makes a tensor three times as wide as the stream. Does attention therefore cache three tensors per layer?">
        No, two. The query is used once, for the position being computed, and thrown away. Only keys and values are needed again by later
        positions, so the cache holds k and v: {shape(h, "S", hd)} each.
      </Check>
      <Takeaways
        items={[
          "Attention is the only operation that moves information between positions; everything else acts on one position at a time.",
          "Heads are not separate layers. They are slices of one projection, attended independently and stitched back together.",
          "The causal mask is added to the scores before the softmax, so a masked position gets weight zero rather than being removed.",
        ]}
      />
    </Section>
  );
}
