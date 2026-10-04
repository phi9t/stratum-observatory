import {
  Section,
  Slider,
  Sentence,
  Readout,
  FactList,
  Prose,
  SourceRef,
  Takeaways,
  Check,
  n,
  useUrlNumber,
} from "../../../kit";
import * as f from "../facts";
export function Serving() {
  const [batch, setBatch] = useUrlNumber("batch", 1, 1, f.MAX_BATCH);
  const [context, setContext] = useUrlNumber("ctx", f.DEMO_CONTEXT, 1, f.positions.value);
  const scalars = f.cacheScalars(batch, context),
    bytes = scalars * f.BF16_BYTES;
  return (
    <Section
      id="serving"
      title="What serving changes"
      source={[
        { source: "serving", symbol: "Qwen3MoeAttention" },
        { source: "serving", symbol: "Qwen3MoeDecoderLayer.forward" },
        { source: "serving", symbol: "Qwen3MoeSparseMoeBlock" },
      ]}
      lede="Decode retains compact keys and values from every layer. vLLM fuses projections and expert dispatch, carries the residual separately, and lets its attention backend own the cache."
    >
      <Slider label="Batch" value={batch} onChange={setBatch} min={1} max={f.MAX_BATCH} />
      <Slider label="Retained context" value={context} onChange={setContext} min={1} max={f.positions.value} />
      <Sentence>
        {n(batch)} {batch === 1 ? "sequence" : "sequences"} retaining {n(context)} tokens keep {n(scalars)} logical KV
        scalars. Assuming BF16 storage, their payload is {n(bytes)} bytes ({(bytes / f.GiB).toFixed(f.DECIMALS)} GiB).
      </Sentence>
      <Readout
        label="Logical cache payload"
        items={[
          {
            term: "Per layer and token",
            value: `${n(f.cacheParts.value)} × ${n(f.kvHeads.value)} × ${n(f.headDim.value)}`,
            note: "keys and values",
          },
          { term: "Layers", value: n(f.layers.value) },
          { term: "Scalars", value: n(scalars), note: `B=${n(batch)}, T=${n(context)}` },
          { term: "Storage assumption", value: `${n(f.BF16_BYTES)} bytes/scalar`, note: "BF16; no allocator overhead" },
        ]}
      />
      <FactList
        items={[
          [
            "Projection packing",
            <>
              Separate checkpoint Q/K/V weights are loaded into <code>QKVParallelLinear</code> and split asymmetrically.
              Expert gate/up matrices are likewise packed.{" "}
              <SourceRef source="serving" symbol="Qwen3MoeModel.load_weights" />
            </>,
          ],
          [
            "Expert fusion",
            <>
              The replicated, unquantized router feeds <code>FusedMoE</code> with{" "}
              <code>renormalize=norm_topk_prob</code>. Tensor-parallel results are reduced after expert execution.{" "}
              <SourceRef source="serving" symbol="Qwen3MoeSparseMoeBlock" />
            </>,
          ],
          [
            "Residual lifetime",
            <>
              The block returns an update and a separate residual; the following RMSNorm performs the add. Final RMSNorm
              completes the last expert addition. <SourceRef source="serving" symbol="Qwen3MoeModel.forward" />
            </>,
          ],
          [
            "KV ownership",
            <>
              Backend <code>Attention</code> owns the cache. KV heads partition when tensor parallelism fits, and
              replicate when ranks outnumber KV heads. The logical payload above counts unique model state, not
              per-device replicated allocations. <SourceRef source="serving" symbol="Qwen3MoeAttention" />
            </>,
          ],
        ]}
      />
      <Prose>
        The serving forward returns hidden states; <code>compute_logits</code> applies the head separately.{" "}
        <SourceRef source="serving" symbol="Qwen3MoeForCausalLM.compute_logits" /> These payload calculations do not
        establish GPU latency or throughput.
      </Prose>
      <Check question="Does doubling batch or retained context change the stored checkpoint weights?">
        No. It doubles the logical cache payload, while the same model matrices are reused. Device replication and
        allocator pages may change actual memory use separately.
      </Check>
      <Takeaways
        items={[
          <>
            Cache grows as B × T × {n(f.layers.value)} × {n(f.cacheParts.value)} × {n(f.kvHeads.value)} ×{" "}
            {n(f.headDim.value)} scalars.
          </>,
          "Fusion changes the physical execution plan. The reference diagrams expose the unfused arithmetic and the logical state it needs.",
        ]}
      />
    </Section>
  );
}
