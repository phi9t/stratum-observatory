import {
  Section,
  Slider,
  Sentence,
  Readout,
  Table,
  Prose,
  SourceRef,
  Boundary,
  Takeaways,
  n,
  useUrlNumber,
} from "../../../kit";
import * as f from "../facts";
export function Serving() {
  const [batch, setBatch] = useUrlNumber("batch", 1, 1, f.window.value);
  const [context, setContext] = useUrlNumber("context", f.demoContext.value, 1, f.context.value);
  const c = f.cacheScalars(batch, context);
  return (
    <Section
      id="serving"
      title="Cache owners determine the budget"
      source={[
        { source: "reference", symbol: "Attention" },
        { source: "serving", symbol: "MQALayer" },
      ]}
      lede="Every layer owns a window, but only KV sources own the long compressed cache and index keys. Reuse must not be billed as a fresh cache per layer."
    >
      <Slider label="Batch" min={1} max={f.window.value} value={batch} onChange={setBatch} />
      <Slider label="Context length" min={1} max={f.context.value} value={context} onChange={setContext} />
      <Sentence>
        {n(batch)} sequences at context {n(context)} need {n(c.total)} payload and partial-group scalars across backbone
        owners.
      </Sentence>
      <Readout
        label="Current scenario"
        items={[
          { term: "Window", value: `L · min(S,W) · Dkv` },
          { term: "Compressed", value: "Σowner floor(S/ratio) · Dkv" },
          { term: "Index keys", value: "Σowner floor(S/ratio) · Dindex" },
        ]}
      />
      <Table
        caption="Reference cache payload, before physical packing"
        head={["State", "Scalars", "Ownership"]}
        rows={[
          ["Window KV", n(c.windowKV), "Each backbone layer"],
          ["Compressed KV", n(c.compressed), "KV sources only"],
          ["Index keys", n(c.index), "KV sources only"],
          ["Partial KV / gate scores", n(c.partial), "Pooling sources only"],
          ["Total", n(c.total), "Multiply all terms by batch"],
        ]}
      />
      <Prose>
        Index selections, candidate masks, token-hash history, rotary tables, allocation slack and allocator metadata
        are excluded. These are scalar counts, not a claim that a fake-quantized tensor occupies packed FP4 or FP8
        bytes.
      </Prose>
      <Prose>
        <b>Projection fusion.</b> SGLang can replace separate query-latent and KV projections with a concatenated
        projection: <SourceRef source="serving" symbol="MqaAttentionBase" />.
      </Prose>
      <Prose>
        <b>Cache write fusion.</b> RMSNorm, RoPE and write into the paged window cache share a path that avoids the BF16
        KV intermediate: <SourceRef source="serving" symbol="MQALayer._compute_kv_to_cache" />. This path can quantize
        from FP32 registers; an environment switch uses BF16-rounded source values. Their rounding need not be
        identical.
      </Prose>
      <Prose>
        <b>Depth hand-off.</b> When its previous-sublayer option is set, SGLang carries incoming pre coefficients and
        excludes same-sublayer collapse fusion (where that option is enabled for V4.1 is not in the pinned files):{" "}
        <SourceRef source="serving" symbol="DeepseekV4Model._forward_layers_hc_pre_from_prev" />. Engram invalidates
        precomputed collapsed inputs at its layer boundary.
      </Prose>
      <Prose>
        <b>Training boundary.</b> The pinned torchtitan V4 block consumes its own current pre coefficients and uses a
        separate learned head: <SourceRef source="training" symbol="DeepSeekV4TransformerBlock.forward" />. It is not an
        exact V4.1 training implementation.
      </Prose>
      <Boundary>
        No weights or runtime benchmarks were downloaded or run. Serving comparison is code-level evidence, not GPU
        performance or numerical parity.
      </Boundary>
      <Takeaways
        items={[
          "Cache grows with its actual owners, not with all consumers.",
          "Paged storage and kernel fusion can remove intermediates without changing the source-level topology.",
          "Runtime precision and packing require their own evidence.",
        ]}
      />
    </Section>
  );
}
