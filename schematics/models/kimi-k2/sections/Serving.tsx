import {
  Section,
  Slider,
  Segments,
  Sentence,
  Readout,
  Prose,
  Takeaways,
  Check,
  SourceRef,
  useUrlNumber,
  useUrlChoice,
  n,
  approx,
} from "../../../kit";
import * as f from "../facts";
export function Serving() {
  const [batch, setBatch] = useUrlNumber("batch", 1, 1, f.demo.batchMax);
  const [context, setContext] = useUrlNumber("context", f.demo.defaultContext, 1, f.context.value);
  const [mode, setMode] = useUrlChoice("cache", "latent", ["latent", "expanded"] as const);
  const width = mode === "latent" ? f.latentWidth.value : f.expandedCache.value;
  const scalars = f.cacheScalars(batch, context, mode === "latent");
  return (
    <Section
      id="serving"
      title="What serving changes"
      source={[
        { source: "serving", symbol: "DeepseekV2MLAAttention" },
        { source: "mla-backend", symbol: "FlashAttnMLAImpl.forward_mqa" },
      ]}
      lede="The publisher eager path caches expanded keys and values. The pinned vLLM serving file delegates caching to a wrapper layer that is not pinned here; the pinned MLA backend is handed a combined content-latent and positional-key cache, slices it, and computes attention in that smaller space."
    >
      <Segments<"latent" | "expanded">
        legend="Cache representation"
        value={mode}
        onChange={setMode}
        options={[
          { id: "latent", label: "vLLM latent" },
          { id: "expanded", label: "Publisher eager" },
        ]}
      />
      <Slider label="Batch sequences" value={batch} onChange={setBatch} min={1} max={f.demo.batchMax} />
      <Slider label="Context tokens" value={context} onChange={setContext} min={1} max={f.context.value} />
      <Sentence>
        {n(batch)} sequences of {n(context)} cached tokens hold {n(scalars)} scalars across all layers in the {mode}{" "}
        representation: {n(scalars * f.demo.bytes)} bytes ({approx(scalars * f.demo.bytes)}B) at BF16.
      </Sentence>
      <Readout
        label="Computed values"
        items={[
          {
            term: "Formula",
            value: `${n(batch)} × ${n(context)} × ${n(f.layers.value)} × ${n(width)} × ${n(f.demo.bytes)}`,
            note: "batch × context × layers × width × bytes per scalar",
          },
          {
            term: "Per-token, per-layer width",
            value: n(width),
            note:
              mode === "latent"
                ? `${n(f.kvRank.value)} latent + ${n(f.rope.value)} rotary key`
                : `${n(f.heads.value)} × (${n(f.qHead.value)} key + ${n(f.v.value)} value)`,
          },
          {
            term: "Expanded / latent",
            value: f.cacheRatio.value.toFixed(f.demo.radDigits),
            note: "scalar ratio; no throughput claim",
          },
        ]}
      />
      <Prose>
        <strong>Backend scope.</strong> The publisher FlashAttention alternative pads values to the query-head width
        before caching. That alternative cache is outside this eager-path calculator.{" "}
        <SourceRef source="reference" symbol="DeepseekV3FlashAttention2.forward" />
      </Prose>
      <Prose>
        <strong>Projection fusion.</strong> vLLM combines query and key/value down-projections into a fused qkv_a
        matrix, then passes the modules into its MLA wrapper. Its MLP combines gate and up matrices and fuses SiLU with
        multiplication. <SourceRef source="serving" symbol="DeepseekV2MLAAttention" />{" "}
        <SourceRef source="serving" symbol="DeepseekV2MLP.forward" />
      </Prose>
      <Prose>
        <strong>Latent decode.</strong> The backend splits kv_c_and_k_pe_cache into content latent and positional key.
        It passes the latent as values and an absorbed content query as q_v, avoiding expanded head keys/values at
        decode. This is the specific MLA path, not a promise about every backend.{" "}
        <SourceRef source="mla-backend" symbol="FlashAttnMLAImpl.forward_mqa" />
      </Prose>
      <Prose>
        <strong>Residual fusion.</strong> vLLM carries a residual tensor alongside the sublayer result and folds
        additions into subsequent RMSNorm calls. Its FP16 overflow scaling branches are outside the BF16 diagram shown
        here. <SourceRef source="serving" symbol="DeepseekV2DecoderLayer.forward" />
      </Prose>
      <Prose>
        <strong>Expert dispatch.</strong> FusedMoE routes tokens, executes selected expert matrices and combines them.
        Shared-expert fusion is backend-dependent; the logical shared expert still consumes no router logit. vLLM can
        apply the routed scaling factor to the combined expert output, where the publisher multiplies the routing
        weights by it. <SourceRef source="serving" symbol="DeepseekV2MoE" />
      </Prose>
      <Check question="Does selecting FP8 weights imply the cache uses one byte per scalar?">
        No. Weight storage and cache dtype are separate settings. This calculator explicitly assumes BF16 cache scalars,
        and the pinned FlashAttention MLA backend does not support its FP8 cache mode.
      </Check>
      <Takeaways
        items={[
          "Cache arithmetic here excludes page rounding, allocator metadata, temporary buffers and parallel replication.",
          "A smaller stored representation is not a measured speedup.",
          "Decode-time caching does not change which earlier positions the causal rule allows.",
        ]}
      />
    </Section>
  );
}
