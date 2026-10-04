import {
  Section,
  Figure,
  StackOverview,
  Table,
  FactList,
  Takeaways,
  TensorScale,
  LayerMap,
  Sentence,
  OpGraph,
  Prose,
  Check,
  Slider,
  RopeDials,
  Readout,
  Ledger,
  Boundary,
  Glossary,
  SourceRef,
  n,
  shape,
  Heatmap,
  Button,
  ref,
  useUrlNumber,
  type Routing,
} from "../../../kit";
import * as f from "../facts";
import * as a from "../architecture";
import { forward } from "../reference/model";
import sources from "../sources.json";
import { ToyRouter } from "../ToyRouter";
const TOY = f.toyConfig.value;
const LAYOUT = { tokenPx: 8, widthPx: 0.007 };
const refs = (symbol: string) => [{ source: "reference", symbol }];
export function Overview({ layer }: { layer: number }) {
  return (
    <Section
      id="overview"
      title="The whole circuit"
      source={refs("GlmMoeDsaModel.forward")}
      lede="A wide residual stream passes through latent attention and token-wise experts. Sparse search selects positions; selected experts transform channels. These are independent decisions."
    >
      <Figure
        label="GLM-5.3 whole decoder"
        caption="Ink carries the residual stream; blue denotes attention, orange denotes experts, violet denotes positions and state, and teal denotes residual bypasses. Layer tiles encode the combined layer kind."
      >
        <StackOverview
          title="GLM-5.3 decoder"
          before={[
            { title: "token ids", line: shape("B", "S"), lane: "merged", kind: "tensor" },
            { title: "embedding", line: n(f.hidden.value), lane: "merged" },
          ]}
          after={[
            { title: "RMSNorm", line: n(f.hidden.value), lane: "merged" },
            { title: "logits · untied head", line: n(f.vocab.value), lane: "merged" },
          ]}
          count={f.layers.value}
          kinds={f.kinds}
          kindOf={f.kindOf}
          selected={layer}
          stackTitle={`${n(f.layers.value)} decoder layers`}
          stackLine={shape("B", "S", f.hidden.value)}
          side={{ title: "positions + causal mask", line: "top-k propagated between layers", lane: "instance" }}
        />
      </Figure>
      <Table
        caption="Stages and outputs"
        head={["Stage", "Operation", "Output"]}
        rows={[
          ["Embedding", "Select a vocabulary row.", shape("B", "S", f.hidden.value)],
          ["Decoder", "Normalize, attend, add; normalize, transform, add.", shape("B", "S", f.hidden.value)],
          ["Final norm", "Normalize before the language head.", shape("B", "S", f.hidden.value)],
          ["Language head", "Project using a separate weight matrix.", shape("B", "S", f.vocab.value)],
        ]}
      />
      <Takeaways
        items={[
          "Token selection changes which positions attention reads; expert routing changes which weights transform a token.",
          "Shared selections do not imply shared attention weights or shared caches.",
        ]}
      />
    </Section>
  );
}
export function Data() {
  return (
    <Section
      id="data"
      title="What a token carries"
      source={refs("GlmMoeDsaAttention.forward")}
      lede="The stream is wide, but decode stores a compressed content vector and a shared positional key in every layer. Expansion happens after the cache boundary."
    >
      <Figure
        label="Latent widths to scale"
        caption={`Main cache: ${n(f.cacheWidth.value)} scalars per token per layer, versus ${n(f.expandedWidth.value)} for expanded keys and values. Search keys are additional state in full-indexer layers.`}
      >
        <TensorScale
          title="Stream and compressed cache"
          pxPerToken={LAYOUT.tokenPx}
          pxPerWidth={LAYOUT.widthPx}
          footnote="Equal scales: height is channel width; horizontal extent is token count."
          stages={[
            {
              label: "residual",
              tensors: [
                { key: "x", name: "hidden", tokens: f.demoTokens.value, width: f.hidden.value, lane: "merged" },
              ],
            },
            {
              label: "query bottleneck",
              tensors: [
                {
                  key: "q",
                  from: "x",
                  name: "q latent",
                  tokens: f.demoTokens.value,
                  width: f.qRank.value,
                  lane: "high",
                },
              ],
            },
            {
              label: "cached state",
              tensors: [
                {
                  key: "c",
                  name: "content + RoPE",
                  tokens: f.demoTokens.value,
                  width: f.cacheWidth.value,
                  lane: "instance",
                },
                { key: "i", name: "index key", tokens: f.demoTokens.value, width: f.indexDim.value, lane: "instance" },
              ],
            },
          ]}
        />
      </Figure>
      <Table
        caption="Representation contracts"
        head={["Tensor", "Shape", "Meaning"]}
        rows={[
          ["input_ids", shape("B", "S"), "Integer vocabulary identities."],
          ["position_ids", shape(1, "S"), "Absolute positions; decode starts after cached tokens."],
          [
            "query",
            shape("B", f.heads.value, "S", f.nope.value + f.rope.value),
            "Content joined to rotated position channels.",
          ],
          ["expanded values", shape("B", f.heads.value, "T", f.valueDim.value), "Distinct values for each head."],
          ["logits", shape("B", "S", f.vocab.value), "Next-token scores before sampling."],
        ]}
      />
      <Takeaways
        items={[
          "The positional key is shared across heads, while expanded content keys and values are head-specific.",
          "The cache contains normalized latent content; it does not contain the unnormalized projection.",
        ]}
      />
    </Section>
  );
}
export function Layers({ layer, onLayer }: { layer: number; onLayer: (n: number) => void }) {
  return (
    <Section
      id="layers"
      title="Choose a layer"
      source={refs("GlmMoeDsaDecoderLayer")}
      lede="The first dense layers run their own search. Later expert layers either search afresh or reuse the last full layer’s selected positions."
    >
      <LayerMap
        label="GLM-5.3 layers"
        count={f.layers.value}
        selected={layer}
        onSelect={onLayer}
        kinds={f.kinds}
        kindOf={f.kindOf}
      />
      <Sentence>
        Layer {n(layer)} uses {f.mlpTypes.value[layer] === "dense" ? "a dense MLP" : "routed and shared experts"}. Its
        attention selection comes from layer {n(f.selectionOwner(layer))};{" "}
        {f.indexTypes.value[layer] === "full" ? "it runs its own indexer" : "it recomputes attention on reused indices"}
        .
      </Sentence>
      <OpGraph id="block" title={`Layer ${n(layer)} · ${f.kindOf(layer)}`} {...a.blockGraph(layer)} />
      <Prose>
        The bar under a box is its output width relative to the widest output in that figure. Selecting a shared layer
        changes the selection owner, not the residual wiring.
      </Prose>
      <Check question="If a layer reuses the preceding selection, does it also reuse the preceding attention output?">
        No. The layer has its own query, key/value and output projections and its own latent cache; only token indices
        are reused.
      </Check>
      <Takeaways
        items={[
          `${n(f.fullCount.value)} layers own indexers; the remaining layers reuse selections.`,
          "Both sublayers read normalized copies and add corrections to saved residuals.",
        ]}
      />
    </Section>
  );
}
export function Attention() {
  const [position, onPosition] = useUrlNumber("position", 0, 0, f.maxContext.value - 1);
  return (
    <Section
      id="attention"
      title="Compress, rotate, attend"
      source={refs("GlmMoeDsaAttention")}
      lede="Queries compress and expand independently of the key/value path. The key/value cache stays compact even though the eager reference expands it to perform attention."
    >
      <Prose>
        Score gauges use {n(f.defaultContext.value)} retained keys for the symbolic T axis. Other gauges use the
        displayed channel axis; inspectors preserve the complete tensor shapes.
      </Prose>
      <OpGraph
        id="attention-topology"
        heading="The parallel paths"
        title="Attention topology"
        {...a.attentionTopology()}
      />
      <Table
        caption="Parallel paths and their meeting point"
        head={["Path", "Route", "Meets at"]}
        rows={[
          ["Query", "q compress → RMSNorm → q expand → split → rotate → join", "attention scores"],
          [
            "Key/value",
            "KV compress → split → normalize content / rotate position → cache → expand",
            "attention scores and weighted values",
          ],
          ["Indexer", "Search or reuse selection → causal intersection", "attention mask"],
        ]}
      />
      <OpGraph id="query" heading="Build queries" title="Query branch" {...a.queryGraph()} />
      <Prose>
        The other branch reads the same normalized input. Its positional slice bypasses latent normalization.
      </Prose>
      <OpGraph id="kv" heading="Build cached state" title="Latent and positional cache" {...a.kvGraph()} />
      <Prose>
        Expanded keys and values are computed from the cached content. The positional key is broadcast across heads.
      </Prose>
      <OpGraph
        id="expand"
        heading="Expand for eager attention"
        title="Head-specific keys and values"
        {...a.expandGraph()}
      />
      <Prose>Queries meet expanded keys only after sparse selection and causality have been intersected.</Prose>
      <OpGraph id="attend" heading="Attend and project" title="Attention arithmetic" {...a.attendGraph()} />
      <Slider label="Absolute position" value={position} min={0} max={f.maxContext.value - 1} onChange={onPosition} />
      <Sentence>
        At position {n(position)}, each positional pair rotates at a different frequency. Content channels remain
        unchanged.
      </Sentence>
      <Figure
        label="Rotary frequencies"
        caption="These dials show angles, not checkpoint activations. Both the main attention and indexer use the same positional slice width."
      >
        <RopeDials
          title="RoPE pairs"
          position={position}
          pairs={f.dialPairs.value}
          theta={f.theta.value}
          rotaryDim={f.rope.value}
        />
      </Figure>
      <Check question="Why can the positional key be stored once per token instead of once per head?">
        The compression projection produces a single positional slice; expand_kv broadcasts that slice to every
        content-key head.
      </Check>
      <Takeaways
        items={[
          `The query latent is ${n(f.qRank.value)} wide, while the content KV latent is ${n(f.kvRank.value)} wide.`,
          `The reference’s latent norms use epsilon ${n(f.latentEps.value)}, while vLLM uses ${n(f.eps.value)}; these paths are not numerically identical.`,
        ]}
      />
    </Section>
  );
}
export function Indexer({ layer }: { layer: number }) {
  return (
    <Section
      id="indexer"
      title="Search once, reuse positions"
      source={refs("GlmMoeDsaIndexer")}
      lede="A lightweight search scores past tokens before full attention. Search heads use ReLU similarity and signed learned weights; there is no search softmax."
    >
      <Sentence>
        Layer {n(layer)}{" "}
        {f.indexTypes.value[layer] === "full"
          ? "computes the search below"
          : "reuses the search from layer " + n(f.selectionOwner(layer)) + " and skips these projections"}
        .
      </Sentence>
      <OpGraph
        id="indexkeys"
        heading="Search queries and keys"
        title="Independent index projections"
        {...a.indexQGraph()}
      />
      <Prose>
        Search queries read the main query latent. Search keys and head weights read the normalized residual input
        directly; those branches run independently.
      </Prose>
      <OpGraph
        id="indexscores"
        heading="Scores to token identities"
        title="Causal token selection"
        {...a.indexScoreGraph()}
      />
      <Readout
        label="Indexer budget"
        items={[
          { term: "selection cap", value: n(f.indexTopK.value), note: "positions per query" },
          { term: "index key", value: n(f.indexDim.value), note: "scalars per token" },
          { term: "search heads", value: n(f.indexHeads.value) },
          { term: "indexer owners", value: n(f.fullCount.value) },
        ]}
      />
      <Check question="Can a shared layer attend to a future token if the reused indices contain one?">
        No. Its own attention mask still intersects the reused index set with causality. Early rows may have masked
        placeholder choices when fewer past tokens exist than the selection cap.
      </Check>
      <Takeaways
        items={[
          "The indexer returns identities, not attention probabilities.",
          `Selection sharing reduces full indexer executions from ${n(f.layers.value)} to ${n(f.fullCount.value)} per backbone pass; it does not reduce the number of attention layers.`,
        ]}
      />
    </Section>
  );
}
export function Experts({ layer }: { layer: number }) {
  return (
    <Section
      id="experts"
      title="Choose experts, keep the shared path"
      source={refs("GlmMoeDsaMoE")}
      lede="The router chooses a small subset of experts using corrected scores. Mixture weights come from uncorrected sigmoid scores; a shared expert always runs alongside them."
    >
      <OpGraph
        id="router"
        heading="Routing rule in expert layers"
        title="Choices and mixture weights"
        {...a.routerGraph()}
      />
      <Prose>
        Each selected expert sees the same normalized token. The dense MLP and the shared expert use the same SiLU-gated
        computation, with different intermediate widths.
      </Prose>
      <OpGraph
        id="expert"
        heading={`Layer ${n(layer)} channel transform`}
        title="SiLU-gated feed-forward"
        {...a.expertGraph(layer)}
      />
      <Prose>
        The routed branch dispatches and accumulates weighted outputs; the shared branch reads the input directly.
      </Prose>
      <OpGraph
        id="mixture"
        heading="Expert-layer recombination"
        title="Independent shared and routed paths"
        {...a.moeGraph()}
      />
      <Table
        caption="Dispatch and recombination"
        head={["Operation", "Rule", "Output"]}
        rows={[
          ["Dispatch", "Gather token rows for each chosen expert.", shape("selected tokens", f.hidden.value)],
          ["Expert output", "SiLU(gate) × up → down.", shape("selected tokens", f.hidden.value)],
          [
            "Weight and scatter-add",
            "Multiply by each routed weight; add at original token indices.",
            shape("B", "S", f.hidden.value),
          ],
          ["Shared branch", "Run without routing and add to the routed sum.", shape("B", "S", f.hidden.value)],
        ]}
      />
      <Check question="If correction bias changes the selected experts, does it directly increase their mixture weights?">
        No. The bias affects choices only. The router gathers the original sigmoid scores and renormalizes those scores.
      </Check>
      <Takeaways
        items={[
          `${n(f.topK.value)} routed experts out of ${n(f.experts.value)} execute for each token; ${n(f.shared.value)} shared expert runs independently.`,
          `Only selected routed weights sum to ${n(f.scale.value)}; the shared branch is added without that router scale.`,
        ]}
      />
    </Section>
  );
}
export function Run() {
  const [seed, onSeed] = useUrlNumber("seed", 1, 1, f.seedMax.value);
  const [layer, onLayer] = useUrlNumber("toy_layer", 0, 0, TOY.mlpTypes.length - 1);
  const [head, onHead] = useUrlNumber("head", 0, 0, TOY.heads - 1);
  const [query, onQuery] = useUrlNumber("query", TOY.indexTopK, 0, f.demoTokens.value - 1);
  const ctx = new ref.Ctx("value", null, seed);
  forward(ctx, TOY, [Array.from({ length: f.demoTokens.value }, (_, i) => i % TOY.vocab)]);
  const att = ctx.trace.get(`l${layer}.attention`)!.data!;
  const values = Array.from({ length: f.demoTokens.value }, (_, q) =>
    Array.from({ length: f.demoTokens.value }, (_, k) => att[(head * f.demoTokens.value + q) * f.demoTokens.value + k]),
  );
  const ix = ctx.trace.get(`l${layer}.indices`)!.data!;
  const selected = [...ix.slice(query * TOY.indexTopK, (query + 1) * TOY.indexTopK)].filter((k) => k <= query);
  const routedLayer = TOY.mlpTypes[layer] === "sparse" ? layer : TOY.mlpTypes.findIndex((x) => x === "sparse");
  const scores = ctx.trace.get(`l${routedLayer}.route_scores`)!.data!,
    indices = ctx.trace.get(`l${routedLayer}.route_indices`)!.data!,
    weights = ctx.trace.get(`l${routedLayer}.route_weights`)!.data!;
  const routing: Routing = {
    scores: [...scores.slice(query * TOY.experts, (query + 1) * TOY.experts)],
    chosen: [...indices.slice(query * TOY.topK, (query + 1) * TOY.topK)],
    weights: [...weights.slice(query * TOY.topK, (query + 1) * TOY.topK)],
    chosenGroups: [0],
  };
  return (
    <Section
      id="run"
      title="Run the circuit"
      source={refs("GlmMoeDsaForCausalLM.forward")}
      lede="The clean reference executes in your browser at toy widths with seeded random weights. These are computed activations, not trained checkpoint outputs."
    >
      <Slider label="Toy layer" value={layer} min={0} max={TOY.mlpTypes.length - 1} onChange={onLayer} />
      <Slider label="Attention head" value={head} min={0} max={TOY.heads - 1} onChange={onHead} />
      <Slider label="Query token" value={query} min={0} max={f.demoTokens.value - 1} onChange={onQuery} />
      <Button onClick={() => onSeed(seed === f.seedMax.value ? 1 : seed + 1)}>Draw new random weights</Button>
      <Sentence>
        Seed {n(seed)}, layer {n(layer)}, head {n(head)}, query {n(query)} reads selected past positions{" "}
        {selected.map(n).join(", ")}.{" "}
        {TOY.indexTypes[layer] === "shared"
          ? "This layer reuses earlier selections."
          : "This layer computes its own selection."}
      </Sentence>
      <Figure
        label="Computed sparse attention"
        caption="The highlighted row follows the query control. The future and unselected positions have zero attention probability."
      >
        <Heatmap
          title="Reference attention probabilities"
          values={values}
          marked={query}
          rowLabel="query"
          columnLabel="key"
        />
      </Figure>
      <Prose>
        The same execution produces the router scores below. When the selected toy layer is dense, this figure shows the
        first toy expert layer instead.
      </Prose>
      <Sentence>
        Token {n(query)} in toy expert layer {n(routedLayer)} chooses experts {routing.chosen.map(n).join(", ")}; its
        routed weights sum to {n(routing.weights.reduce((a, b) => a + b, 0))}.
      </Sentence>
      <Figure
        label="Computed expert routing"
        caption="Random weight arithmetic; the shared expert is independent of the routed choices."
      >
        <ToyRouter routing={routing} />
      </Figure>
      <Readout
        label="Toy execution"
        items={[
          { term: "width", value: n(TOY.hidden) },
          { term: "layers", value: n(TOY.mlpTypes.length) },
          { term: "parameters", value: n(ctx.parameterCount()) },
          { term: "logits", value: shape(...ctx.shapeOf("logits")) },
        ]}
      />
      <Boundary>
        The CPU fixture instantiates the pinned upstream eager implementation, perturbs every parameter and the router
        correction buffer, and checks logits and intermediates within numerical tolerance. Random browser weights
        illustrate arithmetic only; no checkpoint weights are bundled.
      </Boundary>
      <Check question="Why can changing random weights change selected keys in a full layer but preserve the selection relation in a shared layer?">
        The full layer reruns the indexer. The shared layer receives its upstream indices regardless of its own
        attention projections.
      </Check>
      <Takeaways
        items={[
          "Sparse attention probabilities are computed after selection and causality, not painted into the heatmap.",
          "Toy experiments retain both dense and expert layers and full versus shared selection.",
        ]}
      />
    </Section>
  );
}
export function Weights() {
  return (
    <Section
      id="weights"
      title="Where the weights live"
      source={refs("GlmMoeDsaForCausalLM")}
      lede="The trainable decoder ledger follows constructor shapes. Routed expert matrices dominate because every expert is stored even when a token uses only a subset."
    >
      <Ledger
        caption="Unquantized decoder trainable parameters"
        unit="parameters"
        rows={[
          {
            label: "Embedding + head",
            detail: "Untied vocabulary matrices",
            value: f.embeddingParams.value,
            lane: "merged",
          },
          {
            label: "Attention",
            detail: "Compress, expand, latent norms and output",
            value: f.attentionParams.value,
            lane: "high",
          },
          {
            label: "Indexers",
            detail: "Own-indexer layers only; includes key LayerNorm bias",
            value: f.indexParams.value,
            lane: "instance",
          },
          { label: "Dense MLP", detail: "Initial dense layers", value: f.denseParams.value, lane: "low" },
          { label: "Experts", detail: "All routed plus shared matrices", value: f.expertParams.value, lane: "low" },
          {
            label: "Routers",
            detail: "Linear weights; correction buffers excluded",
            value: f.routerParams.value,
            lane: "instance",
          },
          {
            label: "Stream norms",
            detail: "Both block norms plus final norm",
            value: f.normParams.value,
            lane: "merged",
          },
        ]}
      />
      <Boundary>{f.parameterNote}</Boundary>
      <Takeaways
        items={[
          "Stored expert count and active expert count answer different questions.",
          "The output head has its own weights; tying is disabled in this checkpoint.",
        ]}
      />
    </Section>
  );
}
export function Serving() {
  const [batch, onBatch] = useUrlNumber("batch", 1, 1, f.batchMax.value),
    [context, onContext] = useUrlNumber("context", f.defaultContext.value, 1, f.maxContext.value);
  const scalars = f.cacheScalars(batch, context);
  return (
    <Section
      id="serving"
      title="What serving changes"
      source={[
        { source: "serving", symbol: "DeepseekV32Model.forward" },
        { source: "serving-attention", symbol: "DeepseekV32Attention.forward" },
      ]}
      lede="The serving implementation fuses projections and residual normalization, and can attend directly in latent space. Payload accounting below includes main latent caches and full-layer index keys."
    >
      <Slider label="Batch" value={batch} min={1} max={f.batchMax.value} onChange={onBatch} />
      <Slider label="Context tokens" value={context} min={1} max={f.maxContext.value} onChange={onContext} />
      <Sentence>
        {n(batch)} sequences of {n(context)} tokens retain {n(scalars)} logical cached scalars across the decoder. At a
        hypothetical BF16 representation that is {n(scalars * f.bytesPerScalar.value)} payload bytes.
      </Sentence>
      <Readout
        label="Cache arithmetic"
        items={[
          { term: "main", value: `${n(batch)} × ${n(context)} × ${n(f.layers.value)} × ${n(f.cacheWidth.value)}` },
          { term: "index", value: `${n(batch)} × ${n(context)} × ${n(f.fullCount.value)} × ${n(f.indexDim.value)}` },
          { term: "logical scalars", value: n(scalars) },
        ]}
      />
      <FactList
        items={[
          [
            "Fused projections",
            "The reference projects q_a and kv_a independently; serving fuses qkv_a. Projection shapes match, while the launch schedule changes.",
          ],
          [
            "Latent decode",
            "The reference expands KV before eager attention. Serving absorbs content-key expansion into the query, runs sparse MQA, then expands the attended latent through W_UV.",
          ],
          [
            "Distributed normalization",
            "Serving fuses all-reduce, residual addition and RMSNorm; the reference expresses addition and normalization separately.",
          ],
          [
            "Index cache",
            "Serving stores FP8 search keys with scale metadata. The reference retains unquantized keys; logical scalar counts do not determine allocator bytes.",
          ],
          [
            "Latent normalization",
            `Reference epsilon ${n(f.latentEps.value)} differs from serving epsilon ${n(f.eps.value)}. This is an arithmetic difference, not a fusion equivalence claim.`,
          ],
        ]}
      />
      <Boundary>
        Payload excludes block padding, scale metadata, temporary top-k buffers and allocator overhead. Serving kernels
        and quantized numerical equivalence are not verified by the CPU fixture; no throughput or GPU performance claim
        is made.
      </Boundary>
      <Check question="Does sharing token selections eliminate a layer’s main KV cache?">
        No. Every attention layer creates its own normalized content and positional key. Only the lightweight indexer
        key cache is absent from shared layers.
      </Check>
      <Takeaways
        items={[
          "Latent decode changes what is materialized, while keeping the main cached representation compact.",
          "Batch and retained context scale the cache linearly; the selection cap limits reads, not retention.",
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
      source={refs("GlmMoeDsaPreTrainedModel")}
      lede="One checkpoint, pinned implementation bytes, and an explicit boundary around what was tested."
    >
      <Table
        caption="Pinned public sources"
        head={["Role", "Repository", "Revision", "File"]}
        rows={sources.sources.map((s) => [
          s.role,
          s.repo,
          s.revision,
          <a
            key={s.id}
            href={
              s.kind === "hf"
                ? `https://huggingface.co/${s.repo}/blob/${s.revision}/${s.path}`
                : `https://github.com/${s.repo}/blob/${s.revision}/${s.path}`
            }
          >
            {s.path.split("/").at(-1)}
          </a>,
        ])}
      />
      <Boundary>
        Scope: text decoder to logits and logical decode-cache accounting. Excluded: tokenizer behavior, sampling,
        training losses, auxiliary MTP execution, quantized numerical execution and hardware performance. No verified
        checkpoint-specific training implementation was found. The CPU fixture uses random perturbed weights, not the
        released tensors.
      </Boundary>
      <Glossary
        terms={[
          ["Residual stream", "A token’s vector carried through the stack and updated by additions."],
          [
            "Latent attention",
            "Compressed queries and key/value content expanded or absorbed into attention projections.",
          ],
          ["RoPE", "Position-dependent rotations on a subset of query and key channels."],
          ["Indexer", "A lightweight scorer selecting key token identities before attention."],
          ["Shared selection", "Reuse of indices across layers, with fresh attention and independent cache."],
          ["MoE", "A mixture of token-selected expert transformations plus an always-on shared transformation."],
          ["Correction buffer", "Stored router offsets used for choices, not for mixture weights."],
          ["SiLU", "The gate activation x multiplied by sigmoid(x)."],
          [
            "FP8",
            "A storage or arithmetic format requiring scales; logical scalar count does not encode those scales.",
          ],
          ["Prefill", "Processing a complete input sequence before incremental decoding."],
          ["MQA", "Attention over a shared latent cache rather than materialized per-head key/value tensors."],
          ["RMSNorm", "Per-token normalization by root mean square with a learned gain."],
          ["LayerNorm", "Per-token centering and variance normalization with gain and bias."],
        ]}
      />
      <Takeaways
        items={[
          "Source links resolve to pinned commits rather than moving branches.",
          "Numerical toy parity establishes the clean arithmetic at tested widths, not quantized checkpoint parity or GPU performance.",
        ]}
      />
    </Section>
  );
}
