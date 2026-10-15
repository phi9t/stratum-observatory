import {
  AttentionGrid,
  Boundary,
  Check,
  FactList,
  Glossary,
  LayerMap,
  Ledger,
  MoeRouter,
  OpGraph,
  Pipe,
  Pipes,
  Prose,
  Readout,
  Section,
  Segments,
  Sentence,
  Slider,
  Sub,
  Table,
  Takeaways,
  Tensor,
  approx,
  n,
  opItem,
  routeToken,
  shape,
  tensorItem,
  useUrlChoice,
  useUrlNumber,
  type Op,
  type Edge,
} from "../../../kit";
import * as f from "../facts";
import {
  ArchitectureCircuit,
  ContextBudgetFigure,
  IndexPoolDiagram,
  MemoryBudgetBars,
  MhcDiagram,
  ParallelismDiagram,
  ParameterSummary,
} from "../widgets";
import sources from "../sources.json";

const ref = (symbol: string) => ({ source: "reference", symbol });
const serving = (symbol: string) => ({ source: "serving", symbol });
const servingAttention = (symbol: string) => ({ source: "serving_attention", symbol });
const servingKda = (symbol: string) => ({ source: "serving_kda", symbol });
const servingMtp = (symbol: string) => ({ source: "serving_mtp", symbol });

export function Overview({ layer }: { layer: number }) {
  return (
    <Section
      id="overview"
      title="The released circuit"
      source={[ref("Glm5NextTextModel.forward"), ref("Glm5NextForConditionalGeneration.forward")]}
      lede="GLM-5.3-Flash is a 45-layer text decoder with a three-to-one KDA versus sparse-MLA rhythm, four residual streams, routed experts after the dense prefix, and one stored MTP layer."
    >
      <ArchitectureCircuit selected={layer} />
      <FactList
        items={[
          [
            "Hybrid stack",
            <>
              {n(f.kdaCount.value)} KDA layers and {n(f.sparseAttentionCount.value)} sparse MLA layers over{" "}
              {n(f.layers.value)} base layers.
            </>,
          ],
          [
            "NoPE text attention",
            <>
              The text checkpoint has rotary width {n(f.ropeDim.value)} and passes no text position embeddings into
              attention.
            </>,
          ],
          [
            "Dense prefix",
            <>
              Layers{" "}
              {f.mlpLayerTypes.value
                .slice(0, f.denseFirst.value)
                .map((_kind, index) => n(index))
                .join(", ")}
              use dense SwiGLU; later layers use routed experts plus a shared expert.
            </>,
          ],
          [
            "Parameter boundary",
            <>
              {ParameterSummary()} reconcile to {n(f.parameterTotal.value)} stored tensors.
            </>,
          ],
        ]}
      />
      <Boundary>
        The page focuses on text decoding, MTP, serving, and TorchTitan training/scaling evidence. The vision tower is
        accounted for in the parameter ledger but not drawn op by op.
      </Boundary>
      <Takeaways
        items={[
          "The base decoder is not a reused-indexer GLM-5.2-style schedule; every sparse base layer runs a full indexer.",
          "mHC changes the residual shape to four streams even when attention and MLP sublayers consume one collapsed stream.",
          "Stored parameters and active per-token computation differ sharply because most routed experts are inactive for a token.",
        ]}
      />
    </Section>
  );
}

export function Data() {
  return (
    <Section
      id="data"
      title="Token, stream and cache axes"
      source={[ref("Glm5NextTextModel.forward"), ref("Glm5NextTextAttention.expand_kv")]}
      lede="The text path starts with token embeddings, expands into four residual streams, and only collapses those streams at sublayer boundaries."
    >
      <Pipes>
        <Pipe
          lane="merged"
          label="Text prefill"
          items={[
            tensorItem(shape("B", "S")),
            opItem("embed"),
            tensorItem(shape("B", "S", f.hidden.value)),
            opItem("expand mHC"),
            tensorItem(shape("B", "S", f.hcMult.value, f.hidden.value)),
          ]}
        />
        <Pipe
          lane="high"
          label="Sparse layer state"
          items={[
            opItem("q latent", n(f.qRank.value)),
            opItem("KV latent", n(f.kvRank.value)),
            opItem("index scores"),
            tensorItem(shape("B", "S", f.indexTopK.value + f.indexKpool.value - 1)),
          ]}
        />
        <Pipe
          lane="low"
          label="MoE token route"
          items={[
            opItem("router sigmoid"),
            tensorItem(`${n(f.topK.value)} / ${n(f.experts.value)} experts`),
            opItem("routed sum"),
            opItem("shared expert"),
          ]}
        />
      </Pipes>
      <Table
        caption="Primary tensor widths"
        head={["Tensor", "Shape", "Why it matters"]}
        rows={[
          [
            "hidden stream",
            <Tensor name="x" shape={shape("B", "S", f.hidden.value)} />,
            "The width all layers return.",
          ],
          [
            "mHC streams",
            <Tensor name="streams" shape={shape("B", "S", f.hcMult.value, f.hidden.value)} />,
            "The residual boundary stored through depth.",
          ],
          [
            "KDA heads",
            <Tensor name="q,k,v" shape={shape("B", "S", f.linearHeads.value, f.linearHeadDim.value)} />,
            "The recurrent linear-attention state is head-local.",
          ],
          [
            "MLA key/value",
            <Tensor name="kv" shape={shape("B", f.heads.value, "S", f.nopeDim.value + f.valueDim.value)} />,
            "Sparse attention expands latent KV before attention.",
          ],
        ]}
      />
      <Glossary
        terms={[
          [
            "KDA",
            "Kimi Delta Attention, the recurrent linear-attention branch used on three layers out of every four.",
          ],
          ["NoPE", "No positional embedding in the text attention path; this checkpoint has zero rotary text width."],
          [
            "IndexPool",
            "The sparse indexer groups four key vectors and scores one pooled key before expanding selected pools.",
          ],
        ]}
      />
      <Takeaways
        items={[
          "The residual boundary is wider than a normal decoder stream because mHC keeps multiple streams alive.",
          "Sparse-layer state includes both latent MLA cache and indexer-selected token positions.",
          "MoE routing is a token-local path; the stored expert table is much wider than the active route.",
        ]}
      />
    </Section>
  );
}

export function Stack({ layer, onLayer }: { layer: number; onLayer: (next: number) => void }) {
  const selectedKind = f.layerKind(layer);
  return (
    <Section
      id="stack"
      title="Layer schedule"
      source={[ref("Glm5NextTextDecoderLayer.__init__"), ref("Glm5NextTextAttention.__init__")]}
      lede="A selected layer reveals which attention family and MLP family the checkpoint actually instantiates."
    >
      <LayerMap
        label="GLM-5.3-Flash layers"
        count={f.layers.value}
        first={0}
        selected={layer}
        onSelect={onLayer}
        kindOf={(index) => f.layerKind(index)}
        kinds={[
          { id: "kda-dense", label: "KDA + dense MLP", lane: "instance" },
          { id: "kda-moe", label: "KDA + MoE", lane: "low" },
          { id: "sparse-moe", label: "sparse MLA + full indexer + MoE", lane: "high" },
        ]}
      />
      <Readout
        label="Selected layer"
        items={[
          { term: "layer", value: n(layer) },
          { term: "kind", value: selectedKind, text: true },
          { term: "attention", value: f.isSparseAttentionLayer(layer) ? "sparse MLA" : "KDA", text: true },
          { term: "MLP", value: f.isMoELayer(layer) ? "MoE" : "dense", text: true },
        ]}
      />
      <Check question="Which base decoder sparse layers reuse a previous layer's top-k?">
        None in this checkpoint. The reference code supports shared top-k layers, but this config lists every entry of{" "}
        <code>indexer_types</code> as <code>full</code>.
      </Check>
      <Takeaways
        items={[
          "The sparse layers are exactly the layers where the repeating schedule reaches the fourth slot.",
          "Only the first dense-prefix layers avoid MoE; later KDA layers still route through experts.",
          "Changing the selected layer changes both the attention family and the feed-forward family.",
        ]}
      />
    </Section>
  );
}

export function Block() {
  const ops: Op[] = [
    {
      id: "streams",
      label: "mHC streams",
      sub: shape("B", "S", f.hcMult.value, f.hidden.value),
      kind: "tensor",
      lane: "skip",
      col: 0,
      row: 0,
      width: f.mHcStreamWidth.value,
      detail: {
        what: "Layer input carries four residual streams per token.",
        output: ["streams", shape("B", "S", f.hcMult.value, f.hidden.value)],
        refs: [ref("Glm5NextTextHyperConnection.forward")],
      },
    },
    {
      id: "attn-pre",
      label: "attn mHC pre",
      sub: "collapse",
      lane: "skip",
      col: 1,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "Compute pre, post and comb weights, then collapse four streams to one attention input.",
        formula: "x_attn = sum_stream pre_stream * stream",
        params: [["fn", shape(f.mHcMixWidth.value, f.mHcStreamWidth.value)]],
        refs: [ref("Glm5NextTextHyperConnection.forward")],
      },
    },
    {
      id: "attn-norm",
      label: "input RMSNorm",
      sub: n(f.hidden.value),
      col: 2,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "Normalize the collapsed attention input.",
        params: [["weight", shape(f.hidden.value)]],
        refs: [ref("Glm5NextTextDecoderLayer.forward"), ref("Glm5NextTextRMSNorm.forward")],
      },
    },
    {
      id: "attention",
      label: "KDA or sparse MLA",
      sub: n(f.hidden.value),
      lane: "high",
      col: 3,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "Dispatch by layer type: KDA on most layers, sparse MLA and indexer on every fourth layer.",
        refs: [
          ref("Glm5NextTextDecoderLayer.forward"),
          ref("Glm5NextTextLinearAttention.forward"),
          ref("Glm5NextTextAttention.forward"),
        ],
        note: "The selected layer map above shows the exact branch for each layer.",
      },
    },
    {
      id: "attn-post",
      label: "post + comb",
      sub: "expand",
      lane: "skip",
      col: 4,
      row: 0,
      width: f.mHcStreamWidth.value,
      detail: {
        what: "Expand the attention result back to four streams and mix with the residual streams.",
        formula: "streams = post * y + comb^T * residual",
        refs: [ref("Glm5NextTextDecoderLayer.forward"), serving("Glm5NextDecoderLayer.forward")],
        note: "vLLM fuses this post with the next pre when it can.",
      },
    },
    {
      id: "ffn-pre",
      label: "ffn mHC pre",
      sub: "collapse",
      lane: "skip",
      col: 5,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "Collapse streams again before the feed-forward branch.",
        refs: [ref("Glm5NextTextDecoderLayer.forward")],
      },
    },
    {
      id: "ffn-norm",
      label: "post-attn norm",
      sub: n(f.hidden.value),
      col: 6,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "Normalize the feed-forward input after attention-side mixing.",
        params: [["weight", shape(f.hidden.value)]],
        refs: [ref("Glm5NextTextDecoderLayer.forward")],
      },
    },
    {
      id: "mlp",
      label: "dense or MoE",
      sub: n(f.hidden.value),
      lane: "low",
      col: 7,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "Layers before the dense cutoff use a dense SwiGLU; later layers route through MoE and a shared expert.",
        refs: [ref("Glm5NextTextMLP.forward"), ref("Glm5NextTextMoE.forward")],
      },
    },
    {
      id: "ffn-post",
      label: "post + comb",
      sub: "streams",
      lane: "skip",
      col: 8,
      row: 0,
      width: f.mHcStreamWidth.value,
      detail: {
        what: "Return four streams for the next layer.",
        output: ["streams", shape("B", "S", f.hcMult.value, f.hidden.value)],
        refs: [ref("Glm5NextTextDecoderLayer.forward")],
      },
    },
  ];
  const edges: Edge[] = [
    { from: "streams", to: "attn-pre" },
    { from: "attn-pre", to: "attn-norm" },
    { from: "attn-norm", to: "attention" },
    { from: "attention", to: "attn-post" },
    { from: "attn-post", to: "ffn-pre" },
    { from: "ffn-pre", to: "ffn-norm" },
    { from: "ffn-norm", to: "mlp" },
    { from: "mlp", to: "ffn-post" },
  ];
  return (
    <Section
      id="block"
      title="One decoder block"
      source={[ref("Glm5NextTextDecoderLayer.forward"), serving("Glm5NextDecoderLayer.forward")]}
      lede="The block shape is fixed by mHC: every attention or MLP branch gets one collapsed stream, while the layer boundary keeps four streams."
    >
      <MhcDiagram />
      <OpGraph id="block-op" title="Decoder layer operation graph" ops={ops} edges={edges} />
      <Takeaways
        items={[
          "mHC is not a side annotation; it changes the layer-boundary tensor shape.",
          "Attention and MLP each get their own mHC pre/post pair.",
          "Serving preserves the same math while fusing adjacent mHC post/pre work.",
        ]}
      />
    </Section>
  );
}

export function Kda() {
  const ops: Op[] = [
    {
      id: "x",
      label: "normed input",
      sub: n(f.hidden.value),
      kind: "tensor",
      col: 0,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "KDA consumes the collapsed and normalized stream.",
        output: ["x", shape("B", "S", f.hidden.value)],
        refs: [ref("Glm5NextTextLinearAttention.forward")],
      },
    },
    {
      id: "qkv",
      label: "q/k/v projections",
      sub: n(f.kdaWidth.value),
      col: 1,
      row: 0,
      width: f.kdaWidth.value,
      detail: {
        what: "Three independent projections produce query, key and value channels.",
        params: [
          ["q_proj", shape(f.kdaWidth.value, f.hidden.value)],
          ["k_proj", shape(f.kdaWidth.value, f.hidden.value)],
          ["v_proj", shape(f.kdaWidth.value, f.hidden.value)],
        ],
        refs: [ref("Glm5NextTextLinearAttention.__init__")],
      },
    },
    {
      id: "conv",
      label: "causal conv",
      sub: n(f.convKernel.value),
      col: 2,
      row: 0,
      width: f.kdaWidth.value,
      detail: {
        what: "Depthwise causal convolution mixes short token history before recurrent KDA.",
        params: [["conv1d", shape(f.kdaWidth.value * f.denseFirst.value, 1, f.convKernel.value)]],
        refs: [ref("Glm5NextTextLinearAttention.forward"), servingKda("Glm5NextLinearAttention.forward")],
      },
    },
    {
      id: "forget",
      label: "bounded forget",
      sub: n(f.gateLowerBound.value),
      col: 3,
      row: 0,
      width: f.linearHeadDim.value,
      detail: {
        what: "The forget path maps the stream through a low-rank gate and clamps log decay with a safe lower bound.",
        formula: "g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))",
        params: [
          ["f_a_proj", shape(f.linearHeadDim.value, f.hidden.value)],
          ["f_b_proj", shape(f.kdaWidth.value, f.linearHeadDim.value)],
        ],
        refs: [ref("Glm5NextTextForgetGate.forward")],
      },
    },
    {
      id: "beta",
      label: "beta sigmoid",
      sub: n(f.linearHeads.value),
      col: 4,
      row: 0,
      width: f.linearHeads.value,
      detail: {
        what: "A one-logit-per-head beta gate controls KDA state updates.",
        params: [["b_proj", shape(f.linearHeads.value, f.hidden.value)]],
        refs: [ref("Glm5NextTextLinearAttention.forward")],
      },
    },
    {
      id: "recur",
      label: "KDA recurrence",
      sub: shape("H", "D", "D"),
      kind: "state",
      lane: "instance",
      col: 5,
      row: 0,
      width: f.linearHeadDim.value,
      detail: {
        what: "Chunk or recurrent KDA updates a per-head memory matrix and emits a per-token value.",
        inputs: [
          ["q,k,v", shape("B", "S", f.linearHeads.value, f.linearHeadDim.value)],
          ["g,beta", shape("B", "S", f.linearHeads.value, f.linearHeadDim.value)],
        ],
        refs: [ref("Glm5NextTextLinearAttention.forward"), servingKda("Glm5NextLinearAttention._forward")],
      },
    },
    {
      id: "out-gate",
      label: "gated RMSNorm",
      sub: n(f.linearHeadDim.value),
      col: 6,
      row: 0,
      width: f.linearHeadDim.value,
      detail: {
        what: "Normalize the recurrent output per head and multiply by a sigmoid output gate.",
        params: [["o_norm.weight", shape(f.linearHeadDim.value)]],
        refs: [ref("Glm5NextTextRMSNormGated.forward")],
      },
    },
    {
      id: "out",
      label: "o_proj",
      sub: n(f.hidden.value),
      col: 7,
      row: 0,
      width: f.hidden.value,
      detail: {
        what: "Project the flattened KDA heads back to the model stream.",
        params: [["o_proj", shape(f.hidden.value, f.kdaWidth.value)]],
        refs: [ref("Glm5NextTextLinearAttention.forward")],
      },
    },
  ];
  const edges: Edge[] = [
    { from: "x", to: "qkv" },
    { from: "qkv", to: "conv" },
    { from: "conv", to: "recur" },
    { from: "x", to: "forget" },
    { from: "forget", to: "recur" },
    { from: "x", to: "beta" },
    { from: "beta", to: "recur" },
    { from: "recur", to: "out-gate" },
    { from: "out-gate", to: "out" },
  ];
  return (
    <Section
      id="kda"
      title="KDA layers"
      source={[ref("Glm5NextTextLinearAttention.forward"), ref("Glm5NextTextForgetGate.forward")]}
      lede="Most layers use Kimi Delta Attention: projected q/k/v streams, short causal convolution, bounded decay, beta-gated recurrent state and a gated output norm."
    >
      <OpGraph id="kda-op" title="KDA operation graph" ops={ops} edges={edges} />
      <Readout
        label="KDA facts"
        items={[
          { term: "layers", value: `${n(f.kdaCount.value)} of ${n(f.layers.value)}`, text: true },
          { term: "heads", value: n(f.linearHeads.value) },
          { term: "head dim", value: n(f.linearHeadDim.value) },
          { term: "conv kernel", value: n(f.convKernel.value) },
        ]}
      />
      <Check question="What does KDA buy compared with sparse MLA on its layers?">
        It carries recurrent per-head state through depth-local update rules, so those layers avoid building an explicit
        sparse attention score grid.
      </Check>
      <Takeaways
        items={[
          "KDA is the common layer path in this checkpoint, not an occasional special case.",
          "The short convolution and forget gate are part of the attention mechanism, not a separate MLP.",
          "The output returns to hidden width before mHC expands the layer boundary again.",
        ]}
      />
    </Section>
  );
}

export function Sparse() {
  const [query, setQuery] = useUrlNumber("query", f.indexKpool.value, 0, f.demo.tokens - 1);
  return (
    <Section
      id="sparse"
      title="Sparse MLA and IndexPool"
      source={[
        ref("Glm5NextTextAttention.forward"),
        ref("Glm5NextTextIndexer.forward"),
        ref("Glm5NextTextIndexer.get_pooled_states"),
      ]}
      lede="Every sparse base layer runs its own NoPE MLA and its own lightweight indexer; pooled candidates are expanded back into sparse attention indices."
    >
      <IndexPoolDiagram />
      <Slider
        label="Query row"
        min={0}
        max={f.demo.tokens - 1}
        value={query}
        onChange={setQuery}
        display={n(query)}
        lane="high"
      />
      <AttentionGrid
        title="Toy sparse visibility"
        size={f.demo.tokens}
        query={query}
        onQuery={setQuery}
        legend={[
          { state: "self", label: "self and causal recent positions" },
          { state: "sparse", label: "pooled sparse selections" },
          { state: "masked", label: "masked future positions" },
        ]}
        cell={(q, k) => {
          if (k > q) {
            return "masked";
          }
          if (k === q) {
            return "self";
          }
          return k % f.indexKpool.value === q % f.indexKpool.value ? "sparse" : "masked";
        }}
      />
      <Table
        caption="Sparse attention facts"
        head={["Part", "Checkpoint value", "Source behavior"]}
        rows={[
          ["MLA q rank", n(f.qRank.value), "Latent query projection before per-head q expansion."],
          ["MLA KV rank", n(f.kvRank.value), "Latent KV is cached before expansion."],
          ["Indexer heads", n(f.indexHeads.value), "Separate lightweight scoring heads."],
          [
            "IndexPool",
            `${n(f.indexKpool.value)} keys -> pooled key`,
            "Weighted by gate score plus learned slot bias.",
          ],
          ["Top-k", n(f.indexTopK.value), "Selected raw key positions after expanding pooled selections."],
        ]}
      />
      <Check question="Why is the score grid still the long-context pressure point?">
        IndexPool shrinks the candidate key axis by the pool size, but the current score tensor still grows with query
        length times pooled key length.
      </Check>
      <Takeaways
        items={[
          "NoPE means the sparse text path has no text rotary slice.",
          "The current implementation still has an indexer score term shaped like L times L divided by the pool size.",
          "MTP serving can reuse top-k between draft iterations, but the base sparse layers here do not reuse another layer's top-k.",
        ]}
      />
    </Section>
  );
}

export function Experts() {
  const [seed, setSeed] = useUrlNumber("route", f.topK.value, 0, f.demo.routerSeedMax);
  const routing = routeToken({
    experts: f.experts.value,
    topK: f.topK.value,
    scoring: "sigmoid",
    seed,
    groups: f.routerGroups.value,
    topGroups: f.routerTopGroups.value,
    normalize: f.normTopk.value,
    scale: f.routerScale.value,
  });
  return (
    <Section
      id="experts"
      title="Router and experts"
      source={[
        ref("Glm5NextTextTopkRouter.forward"),
        ref("Glm5NextTextExperts.forward"),
        ref("Glm5NextTextMoE.forward"),
      ]}
      lede="The router selects routed experts with a sigmoid score plus correction bias, then the selected original sigmoid scores become normalized weights. The shared expert is always on."
    >
      <Slider
        label="Router seed"
        min={0}
        max={f.demo.routerSeedMax}
        value={seed}
        onChange={setSeed}
        display={n(seed)}
        lane="merged"
      />
      <MoeRouter
        title="Sigmoid top-k router toy"
        routing={routing}
        experts={f.experts.value}
        shared={f.sharedExperts.value}
      />
      <FactList
        items={[
          [
            "Selection",
            <>
              Top-{n(f.topK.value)} of {n(f.experts.value)} routed experts, using sigmoid scores plus{" "}
              <code>e_score_correction_bias</code>.
            </>,
          ],
          [
            "Weights",
            <>Original sigmoid scores are gathered, normalized, then multiplied by {n(f.routerScale.value)}.</>,
          ],
          [
            "Shared expert",
            <>
              {n(f.sharedExperts.value)} shared expert path is added after the routed sum and consumes no router logit.
            </>,
          ],
        ]}
      />
      <Check question="Does a selected routed expert count prove all experts are active?">
        No. The checkpoint stores all routed expert weights, but each token uses only the selected top-k routed experts
        plus the shared path.
      </Check>
      <Takeaways
        items={[
          "The correction bias changes selection, while the gathered original sigmoid scores become route weights.",
          "The shared expert is a separate always-on path and does not consume router logits.",
          "Per-token active compute is top-k routed experts plus the shared expert, not the full stored expert table.",
        ]}
      />
    </Section>
  );
}

export function Training() {
  const [mode, setMode] = useUrlChoice<"fsdp" | "ep">("parallel", "ep", ["fsdp", "ep"]);
  const [ctxIndex, setCtxIndex] = useUrlNumber("ctx", 0, 0, f.longContextEstimates.length - 1);
  const ctx = f.longContextEstimates[ctxIndex] ?? f.longContextEstimates[0];
  return (
    <Section
      id="training"
      title="Training and scaling"
      source={[serving("Glm5NextMoE.forward"), serving("Glm5NextModel.__init__")]}
      lede="TorchTitan campaign evidence separates what is derived from config from what was measured on 8x B200."
    >
      <Sub>LoRA SFT memory</Sub>
      <MemoryBudgetBars />
      <Table
        caption="LoRA memory evidence"
        head={["Claim", "Value", "Evidence"]}
        rows={[
          [
            "Routed experts",
            `${approx(f.memoryEvidence.routedExpertParams)} params -> ${n(f.memoryEvidence.routedExpertBf16GiB)} GiB`,
            f.trainingSources.paramCounts,
          ],
          [
            "EP8 static floor",
            `${f.memoryEvidence.staticFloorEp8GiB.toFixed(2)} GiB/GPU`,
            f.trainingSources.paramCounts,
          ],
          [
            "Measured peak",
            `${f.memoryEvidence.measuredPeakEp8GiB.toFixed(2)} GiB/GPU`,
            `${f.trainingSources.t08fLogs} · measured on 8x B200, TorchTitan`,
          ],
          [
            "EP4 static estimate",
            `${f.memoryEvidence.staticFloorEp4GiB.toFixed(2)} GiB/GPU`,
            f.trainingSources.paramCounts,
          ],
          [
            "Full fine-tune",
            `about ${f.fullFinetuneTb.value.toFixed(1)} TB`,
            <>
              DERIVED: text stored tensors x ({n(f.memoryEvidence.rank16OptimizerBytesPerParam)} B fp32 master, grad and
              AdamW state + {n(f.bytesPerBf16.value)} B bf16 weights)
            </>,
          ],
        ]}
      />
      <Sub>Parallelism</Sub>
      <Segments
        legend="Parallelism diagram"
        value={mode}
        onChange={setMode}
        options={[
          { id: "fsdp", label: "FSDP all-gather" },
          { id: "ep", label: "EP token dispatch" },
        ]}
      />
      <ParallelismDiagram mode={mode} />
      <Table
        caption="Measured timing comparison"
        head={["Path", "Scope", "ms/step", "GiB/rank", "Evidence"]}
        rows={f.parallelEvidence.map((row) => [
          row.label,
          row.scope,
          row.msPerStep.toFixed(2),
          row.memoryGiB.toFixed(2),
          row.source,
        ])}
      />
      <Sub>Long context estimates</Sub>
      <Slider
        label="Context scenario"
        min={0}
        max={f.longContextEstimates.length - 1}
        value={ctxIndex}
        onChange={setCtxIndex}
        display={ctx.minTokens === ctx.maxTokens ? n(ctx.minTokens) : `${n(ctx.minTokens)}-${n(ctx.maxTokens)}`}
        lane="skip"
      />
      <ContextBudgetFigure tokens={ctx.minTokens} />
      <Sentence>
        {ctx.label}: {ctx.minTokens === ctx.maxTokens ? n(ctx.minTokens) : `${n(ctx.minTokens)}-${n(ctx.maxTokens)}`}{" "}
        tokens per GPU is an estimate, not a measured result.
      </Sentence>
      <Table
        caption="Countdown SFT ladder status"
        head={["Rung", "Model", "Status", "Evidence"]}
        rows={f.sftLadder.map((row) => [row.rung, row.model, row.status, row.source])}
      />
      <Boundary>
        DeepEP is pinned at commit {f.deepEpEvidence.commit}; GLM DeepEP training did not complete, so the page makes no
        GLM DeepEP loss, gradient, or step-time claim.
      </Boundary>
      <Check question="Why does expert parallelism change the memory story?">
        Expert parallelism shards the routed expert weights across ranks and moves token batches to their owner, so each
        rank avoids holding every routed expert locally.
      </Check>
      <Takeaways
        items={[
          "The measured TorchTitan values are labeled separately from config-derived estimates.",
          "Full fine-tuning is excluded by optimizer-state scale, while rank-limited LoRA fits the measured EP run.",
          "DeepEP is source-pinned but not claimed as a completed GLM training result.",
        ]}
      />
    </Section>
  );
}

export function Run() {
  const [step, setStep] = useUrlNumber("run", 0, 0, f.layers.value + f.nextn.value);
  const layer = Math.min(step, f.layers.value - 1);
  const afterLayers = step >= f.layers.value;
  return (
    <Section
      id="run"
      title="Run the stack"
      source={[ref("Glm5NextTextModel.forward"), servingMtp("Glm5NextMultiTokenPredictor.forward")]}
      lede="A scrubber follows the base decoder and then the stored MTP layer, using the checkpoint's actual layer schedule."
    >
      <Slider
        label="Execution step"
        min={0}
        max={f.layers.value + f.nextn.value}
        value={step}
        onChange={setStep}
        display={afterLayers ? "MTP / head" : `layer ${n(layer)}`}
        lane="merged"
      />
      <Sentence>
        {afterLayers
          ? "The base decoder has produced the final hidden state; the MTP layer is stored for draft-token serving."
          : `Layer ${n(layer)} runs ${f.isSparseAttentionLayer(layer) ? "sparse MLA with a full indexer" : "KDA"} and ${
              f.isMoELayer(layer) ? "MoE" : "dense SwiGLU"
            }.`}
      </Sentence>
      <Readout
        label="Run step"
        items={[
          { term: "hidden", value: shape("B", "S", f.hidden.value), text: true },
          { term: "streams", value: shape("B", "S", f.hcMult.value, f.hidden.value), text: true },
          { term: "top-k width", value: n(f.indexOutputWidth.value) },
          { term: "MTP layers", value: n(f.nextn.value) },
        ]}
      />
      <Takeaways
        items={[
          "The output head is untied from the token embedding.",
          "MTP is a serving/draft layer in the stored checkpoint, not another mHC base decoder layer.",
          "Top-k sharing belongs to MTP draft iterations here, not to base sparse layers.",
        ]}
      />
    </Section>
  );
}

export function Weights() {
  return (
    <Section
      id="weights"
      title="Stored tensors"
      source={[ref("Glm5NextModel.__init__"), ref("Glm5NextVisionModel.__init__")]}
      lede="The parameter ledger is a stored-tensor inventory that reconciles the BF16 Hub total, including text, MTP and vision weights."
    >
      <Ledger
        caption="Stored checkpoint tensor ledger"
        unit="Parameters"
        rows={f.parameterLedger.value}
        footnote="Stored tensor count; not active-per-token compute and not allocator bytes."
      />
      <Prose>{f.parameterNote}</Prose>
      <Boundary>
        The page draws the text/MTP circuit. Vision classes are covered here as stored tensors, not as an op-level
        vision schematic.
      </Boundary>
      <Takeaways
        items={[
          "The ledger reconciles stored tensors, not active-per-token compute.",
          "Routed expert tensors dominate stored parameters even though each token selects only a small route.",
          "Vision and MTP are included in accounting, while the op-level drawing stays focused on text decoding.",
        ]}
      />
    </Section>
  );
}

export function Serving() {
  return (
    <Section
      id="serving"
      title="Serving path"
      source={[
        serving("Glm5NextModel.__init__"),
        servingAttention("Indexer.forward"),
        servingKda("Glm5NextLinearAttention.forward"),
        servingMtp("Glm5NextMultiTokenPredictor.set_skip_topk"),
      ]}
      lede="vLLM keeps the same graph but changes where buffers live and which adjacent operations are fused."
    >
      <Table
        caption="Serving implementation differences"
        head={["Area", "Serving behavior", "Why it matters"]}
        rows={[
          [
            "Top-k buffer",
            `width ${n(f.indexTopK.value)} plus tail room rounded to sparse tiles`,
            "Padded slots stay masked.",
          ],
          [
            "Indexer cache",
            `one state per ${n(f.indexKpool.value)} tokens`,
            "Compressed pooled state lowers cache frequency.",
          ],
          ["Tail cache", "raw K and gate scores for incomplete pool", "Keeps tail selection exact."],
          ["KDA", "merged projections and fused recurrent kernels", "Same recurrence, lower overhead."],
          ["mHC", "deferred post fused with next pre", "Layer-boundary fusion without changing math."],
          ["MTP", "step zero computes top-k, later draft steps may reuse it", "This is not base-layer reuse."],
        ]}
      />
      <Boundary>
        The serving commit is pinned in `sources.json`. It is not a numerical parity certificate for this page.
      </Boundary>
      <Takeaways
        items={[
          "Serving changes buffer placement and fuses adjacent operations without changing the decoded graph identity.",
          "MTP top-k reuse is serving-draft behavior, not evidence of base-layer sparse reuse.",
          "The vLLM comparison is source-backed implementation reading, not a benchmark claim.",
        ]}
      />
    </Section>
  );
}

export function Sources() {
  return (
    <Section
      id="sources"
      title="Sources and boundaries"
      source={[ref("Glm5NextForConditionalGeneration"), serving("Glm5NextForCausalLM")]}
      lede="Every code-linked claim is bound to a fetched public source. TorchTitan results are local measured evidence and are displayed with repo-relative citations."
    >
      <Table
        caption="Pinned public source files"
        head={["Role", "Repository", "Revision / path"]}
        rows={sources.sources.map((source) => [
          source.role,
          source.repo,
          `${source.revision.slice(0, f.topK.value + f.sharedExperts.value)} · ${source.path}`,
        ])}
      />
      <Table
        caption="Local training evidence citations"
        head={["Evidence", "Citation"]}
        rows={Object.entries(f.trainingSources).map(([key, path]) => [key, path])}
      />
      <Boundary>
        Whole-model numerical parity is unverified. The clean reference is shape-only and checks tensor inventory,
        schedules and parameter accounting without loading weights.
      </Boundary>
      <Takeaways
        items={[
          "Public code sources are fetched and symbol-linked; local TorchTitan reports are cited by repo-relative path and commit.",
          "Measured training claims come only from the TorchTitan campaign rows shown on the page.",
          "The page does not claim released-weight logits, GPU kernels or benchmark parity.",
        ]}
      />
    </Section>
  );
}
