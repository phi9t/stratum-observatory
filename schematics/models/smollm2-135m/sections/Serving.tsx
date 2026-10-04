import {
  Check,
  FactList,
  Readout,
  Section,
  Sentence,
  Slider,
  SourceRef,
  Takeaways,
  n,
  useUrlNumber,
} from "../../../kit";
import * as f from "../facts";
export function Serving() {
  const [b, setB] = useUrlNumber("batch", 1, 1, f.DISPLAY.batchMax);
  const [s, setS] = useUrlNumber("ctx", f.DISPLAY.context, 1, f.positions.value);
  const scalars = f.cacheScalars(b, s);
  return (
    <Section
      id="serving"
      title="What serving changes"
      lede="A decode step projects the new token and reads prior rotated keys and values. Packed projections and fused residual updates change execution, while preserving the logical circuit."
      source={[
        { source: "serving", symbol: "LlamaAttention" },
        { source: "serving", symbol: "LlamaDecoderLayer" },
      ]}
    >
      <Slider label="Batch" value={b} onChange={setB} min={1} max={f.DISPLAY.batchMax} />
      <Slider label="Context length" value={s} onChange={setS} min={1} max={f.positions.value} />
      <Sentence>
        {n(b)} stream{b === 1 ? "" : "s"} at {n(s)} tokens retain {n(scalars)} scalars, or{" "}
        {n(scalars * f.DISPLAY.bytes)} bytes under a bfloat16 cache assumption.
      </Sentence>
      <Readout
        label="Logical cache budget"
        items={[
          {
            term: "Formula",
            value: `B × S × ${n(f.layers.value)} × ${n(f.kvParts.value)} × ${n(f.kvHeads.value)} × ${n(f.headDim.value)}`,
          },
          {
            term: "Storage",
            value: `${n(f.DISPLAY.bytes)} bytes/scalar`,
            note: "Assumption; backend cache dtype can differ.",
          },
          { term: "Savings", value: `${n(f.groups.value)}×`, note: "Compared with one KV head per query head." },
        ]}
      />
      <FactList
        items={[
          [
            "Packed projections",
            <>
              QKVParallelLinear packs the separate checkpoint Q/K/V tensors. Gate and up are packed by the MLP, which
              uses SiluAndMul. <SourceRef source="serving" symbol="LlamaAttention.forward" />{" "}
              <SourceRef source="serving" symbol="LlamaMLP" />
            </>,
          ],
          [
            "Deferred residual addition",
            <>
              vLLM carries hidden and residual tensors separately and combines addition with RMSNorm. The final norm
              completes the last pending addition. <SourceRef source="serving" symbol="LlamaDecoderLayer.forward" />{" "}
              <SourceRef source="serving" symbol="LlamaModel.forward" />
            </>,
          ],
          [
            "Distributed state",
            <>
              KV heads are partitioned when there are enough for tensor parallel ranks; otherwise replicated. This
              logical budget excludes replication, allocator pages and metadata.{" "}
              <SourceRef source="serving" symbol="LlamaAttention" />
            </>,
          ],
        ]}
      />
      <FactList
        items={[
          [
            "Training implementation",
            <>
              The publisher identifies Nanotron as its pretraining framework. Its Llama attention uses packed QKV
              projections and fused attention. This is an implementation comparison, not proof of the exact historical
              training run or checkpoint numerical parity. <SourceRef source="training" symbol="CausalSelfAttention" />
            </>,
          ],
          [
            "Tied output weights",
            <>
              vLLM ties its output head to the vocabulary embedding and skips a separate head during loading.
              <SourceRef source="serving" symbol="LlamaForCausalLM" />
            </>,
          ],
        ]}
      />
      <Check question="Why can this cache budget differ from memory allocated by a serving engine?">
        It counts logical keys and values only. Tensor-parallel replication, allocator pages, metadata and a different
        cache dtype can change physical allocation without changing the attention circuit.
      </Check>
      <Takeaways
        items={[
          "Queries are recomputed; old keys and values persist.",
          "Logical cache bytes are a capacity calculation, not a measured throughput or memory allocation result.",
        ]}
      />
    </Section>
  );
}
