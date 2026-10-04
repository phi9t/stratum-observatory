import {
  Section,
  Slider,
  Segments,
  Sentence,
  Readout,
  Table,
  Takeaways,
  Check,
  Prose,
  OpGraph,
  n,
  shape,
  approx,
  useUrlNumber,
  useUrlChoice,
  SourceRef,
  type Op,
} from "../../../kit";
import * as f from "../facts";
import { operation as op } from "../graph";
const LAYOUT = { columns: [0, 1, 2, 3, 4, 5, 6, 7] };

export function Serving() {
  const [batch, setBatch] = useUrlNumber("batch", f.DEMO.batch, 1, f.DEMO.maxBatch);
  const [ctx, setCtx] = useUrlNumber("ctx", f.DEMO.context, 1, f.positions.value);
  const [mode, setMode] = useUrlChoice("cache", "compressed", ["compressed", "eager", "flash"] as const);
  const width =
    mode === "compressed" ? f.kvAWidth.value : mode === "eager" ? f.eagerCacheWidth.value : f.flashCacheWidth.value;
  const scalars = f.cacheScalars(batch, ctx, mode);
  const refs = [
    { source: "publisher", symbol: "MLA.forward" },
    { source: "mla-backend", symbol: "MultiHeadLatentAttentionWrapper.forward" },
  ];
  const absorb: Op[] = [
    op("absorb", "absorb key map", 0, f.heads.value * f.kvRank.value, {
      what: "Move the nonpositional key expansion to the query side of the dot product.",
      formula: "q_latent[h] = q_nope[h] @ W_K[h]",
      inputs: [
        ["q_nope", shape("B", f.heads.value, 1, f.nope.value)],
        ["W_K", shape(f.heads.value, f.nope.value, f.kvRank.value)],
      ],
      output: ["q_latent", shape("B", f.heads.value, 1, f.kvRank.value)],
      params: [["W_K slice of KV B", shape(f.heads.value, f.nope.value, f.kvRank.value)]],
      refs,
    }),
    op(
      "cache",
      "write latent",
      0,
      f.kvAWidth.value,
      {
        what: "Keep the normalized latent and rotated shared positional key for each new token.",
        formula: "cache.append(c_norm, k_pe_rotated)",
        inputs: [
          ["c_norm", shape("B", 1, f.kvRank.value)],
          ["k_pe", shape("B", 1, f.rotary.value)],
        ],
        output: ["cache", shape("B", "T", f.kvAWidth.value)],
        refs,
        note: "The publisher keeps latent and positional key in separate buffers; the combined shape names their total storage, not an extra concatenation operation.",
      },
      "instance",
      1,
      "state",
    ),
    op("scores", "latent + rotary", 1, f.heads.value * f.DEMO.tokens, {
      what: "Sum the latent-space and positional dot products against cached history, then scale, mask and softmax.",
      formula: "p = softmax((q_latent·cᵀ + q_pe·k_peᵀ)·scale + mask)",
      inputs: [
        ["cache", shape("B", "T", f.kvAWidth.value)],
        ["q_latent", shape("B", f.heads.value, 1, f.kvRank.value)],
        ["q_pe", shape("B", f.heads.value, 1, f.rotary.value)],
      ],
      output: ["p", shape("B", f.heads.value, 1, "T")],
      refs,
      note: "This is a comparison diagram; the separate mathematical scoring operations are drawn in the eager attention section. Gauge uses the illustrated history length.",
    }),
    op("average", "latent average", LAYOUT.columns[2], f.heads.value * f.kvRank.value, {
      what: "Average compressed history before expanding values.",
      formula: "latent_out = p @ c_history",
      inputs: [
        ["p", shape("B", f.heads.value, 1, "T")],
        ["c", shape("B", "T", f.kvRank.value)],
      ],
      output: ["latent_out", shape("B", f.heads.value, 1, f.kvRank.value)],
      refs,
    }),
    op("expand", "expand values", LAYOUT.columns[3], f.outputWidth.value, {
      what: "Apply the value slice of KV B after the history reduction.",
      formula: "value_out[h] = latent_out[h] @ W_V[h]ᵀ",
      inputs: [["latent_out", shape("B", f.heads.value, 1, f.kvRank.value)]],
      output: ["value_out", shape("B", 1, f.outputWidth.value)],
      note: "Per-head values are flattened across heads before o_proj; this reshape is included in the output shape.",
      params: [["W_V slice of KV B", shape(f.heads.value, f.valueDim.value, f.kvRank.value)]],
      refs,
    }),
    op("output", "o projection", LAYOUT.columns[4], f.hidden.value, {
      what: "Mix the head outputs back into the residual width.",
      formula: "a = value_out @ W_Oᵀ",
      inputs: [["value_out", shape("B", 1, f.outputWidth.value)]],
      output: ["a", shape("B", 1, f.hidden.value)],
      params: [["W_O", shape(f.hidden.value, f.outputWidth.value)]],
      refs,
    }),
  ];
  return (
    <Section
      id="serving"
      title="What serving changes"
      source={[
        { source: "serving", symbol: "DeepseekV2MLAAttention" },
        { source: "publisher", symbol: "MLA.forward" },
      ]}
      lede="Absorbed decode moves key expansion to the query and value expansion after the weighted average. The same algebra therefore needs much less persistent state than eager expanded attention."
    >
      <OpGraph
        heading="Absorbed decode, algebraic comparison"
        id="absorb-op"
        title="Compressed decode reordering"
        ops={absorb}
        edges={[
          { from: "absorb", to: "scores" },
          { from: "cache", to: "scores", kind: "state" },
          { from: "scores", to: "average" },
          { from: "cache", to: "average", kind: "state" },
          { from: "average", to: "expand" },
          { from: "expand", to: "output" },
        ]}
      />
      <Prose>
        <SourceRef source="serving" symbol="DeepseekV3ForCausalLM" /> inherits the implementation in this file. Its MLA
        wrapper passes normalized latent and rotary key to the attention backend; the publisher's absorption branch
        above makes the reordering explicit. Backend-specific kernels and allocator overhead are outside this size
        calculation.
      </Prose>
      <Segments
        legend="Cache representation"
        options={[
          { id: "compressed", label: "Compressed MLA" },
          { id: "eager", label: "Reference eager" },
          { id: "flash", label: "Reference FlashAttention" },
        ]}
        value={mode}
        onChange={setMode}
      />
      <Slider label="Batch" value={batch} min={1} max={f.DEMO.maxBatch} onChange={setBatch} />
      <Slider label="Context length" value={ctx} min={1} max={f.positions.value} onChange={setCtx} />
      <Sentence>
        {n(batch)} {batch === 1 ? "sequence" : "sequences"} with {n(ctx)} cached tokens use {n(scalars)} scalars (
        {approx(scalars * f.DEMO.bytes)}B at the illustrative BF16 cache precision) in {mode} storage across the main
        layers.
      </Sentence>
      <Readout
        label="Cache budget terms"
        items={[
          { term: "Per token / layer", value: n(width), note: "scalars" },
          { term: "Main layers", value: n(f.layers.value) },
          { term: "Formula", value: `${n(batch)} × ${n(ctx)} × ${n(f.layers.value)} × ${n(width)}` },
          {
            term: "Logical payload",
            value: n(scalars * f.DEMO.bytes),
            note: `bytes at ${n(f.DEMO.bytes)} bytes/scalar`,
          },
        ]}
      />
      <Table
        caption="Concrete implementation differences"
        head={["Change", "Implementation evidence"]}
        rows={[
          [
            "Projection fusion",
            <>
              vLLM merges Q A and KV A into width {n(f.fusedAWidth.value)}.{" "}
              <SourceRef source="serving" symbol="DeepseekV2MLAAttention" />
            </>,
          ],
          [
            "FFN fusion",
            <>
              Merged gate/up plus SiluAndMul. <SourceRef source="serving" symbol="DeepseekV2MLP.forward" />
            </>,
          ],
          [
            "Residual fusion",
            <>
              Residual is carried separately and added in the next norm; the final norm completes the addition.{" "}
              <SourceRef source="serving" symbol="DeepseekV2DecoderLayer.forward" />
            </>,
          ],
          [
            "Expanded fallback",
            <>
              vLLM can select expanded attention instead of MLA. Reference FlashAttention pads V to query width before
              caching. <SourceRef source="reference" symbol="DeepseekV3FlashAttention2.forward" />
            </>,
          ],
          [
            "Auxiliary tensors",
            <>
              Main-model loading skips speculative layers.{" "}
              <SourceRef source="serving" symbol="DeepseekV2Model.load_weights" />
            </>,
          ],
          [
            "Training",
            <>
              torchtitan expands KV and uses token-flattened axes and distributed sharding; its grouped selector matches
              the reference. <SourceRef source="training" symbol="Attention.forward" />{" "}
              <SourceRef source="training-moe" symbol="DeepSeekV3Router._select_experts" />
            </>,
          ],
        ]}
      />
      <Check question="Does the cache reduction mean fewer attention heads are being computed?">
        No. Each query still has its own per-head output. The compression avoids storing head-specific historical
        expansions; it changes when linear maps are applied, not how many head results exist.
      </Check>
      <Takeaways
        items={[
          `Compressed storage keeps ${n(f.kvRank.value)} latent and ${n(f.rotary.value)} positional scalars per token per layer.`,
          "Logical payload excludes block padding, replication, cache quantization scales and implementation workspaces; it is not a GPU memory or throughput measurement.",
        ]}
      />
    </Section>
  );
}
