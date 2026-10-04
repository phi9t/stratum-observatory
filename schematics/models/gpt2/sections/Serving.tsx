import { FactList, Readout, Section, Sentence, Slider, SourceRef, approx, n, useUrlNumber } from "../../../kit";
import * as f from "../facts";

const LAYOUT = { batchMax: 64, startBatch: 1, startContext: 256 };

export function Serving() {
  const [batch, setBatch] = useUrlNumber("batch", LAYOUT.startBatch, 1, LAYOUT.batchMax);
  const [context, setContext] = useUrlNumber("ctx", LAYOUT.startContext, 1, f.positions.value);
  const scalars = f.cacheScalars(batch, context);
  return (
    <Section
      id="serving"
      lede="Generating one token at a time, a server never recomputes the past: it keeps every layer's keys and values and projects only the newest position."
      source={[
        { source: "serving", symbol: "GPT2Attention" },
        { source: "serving", symbol: "GPT2Model" },
      ]}
      title="What serving changes"
    >
      <Slider label="Batch" max={LAYOUT.batchMax} min={1} onChange={setBatch} value={batch} />
      <Slider label="Context length" max={f.positions.value} min={1} onChange={setContext} value={context} />
      <Sentence>
        {n(batch)} sequence{batch === 1 ? "" : "s"} of {n(context)} tokens keep {n(scalars)} cached scalars, about{" "}
        {approx(scalars * f.BYTES_PER_SCALAR)}B at float32.
      </Sentence>
      <Readout
        items={[
          { term: "Per layer", value: `2 × ${n(f.heads.value)} × S × ${n(f.headDim.value)}`, note: "keys and values" },
          { term: "Layers", value: n(f.layers.value) },
          { term: "Scalars", value: n(scalars), note: `batch ${n(batch)}, S = ${n(context)}` },
          {
            term: "Versus weights",
            value: `${(scalars / f.parameterTotal.value).toFixed(2)}×`,
            note: "of the parameter count",
          },
        ]}
        label="Cache size readout"
      />
      <FactList
        items={[
          [
            "Fused projection",
            <>
              vllm loads <code>c_attn</code> as one fused q/k/v layer and chunks its output three ways (
              <SourceRef source="serving" symbol="GPT2Attention.forward" />
              ); the arithmetic is the reference's.
            </>,
          ],
          [
            "Cache ownership",
            <>
              The reference passes a cache object through every forward call; vllm's attention layer owns the cache, so
              its block forward takes only the hidden states.
            </>,
          ],
          [
            "Refused options",
            <>
              vllm asserts that inverse-layer attention scaling is off and has no cross-attention branch (
              <SourceRef source="serving" symbol="GPT2Model" />
              ). This checkpoint uses neither.
            </>,
          ],
        ]}
      />
    </Section>
  );
}
