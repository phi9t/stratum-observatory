import {
  Section,
  Slider,
  Sentence,
  Readout,
  Figure,
  Heatmap,
  Prose,
  Takeaways,
  Check,
  n,
  useUrlNumber,
  Boundary,
} from "../../../kit";
import * as f from "../facts";
import { Ctx } from "../../../kit/ref/tensor";
import { forward, TOY, TOY_IDS, toyHashes } from "../reference/model";
import { useLayer } from "./Stack";
export function Run() {
  const [layer] = useLayer();
  const [seed, setSeed] = useUrlNumber("runseed", f.copies.value, 1, f.window.value);
  const [head, setHead] = useUrlNumber("runhead", 0, 0, f.toySpec.value.h - 1);
  const ctx = new Ctx("value", null, seed, 2);
  const logits = forward(ctx, TOY, TOY_IDS, toyHashes(TOY, TOY_IDS));
  const indices = ctx.trace.get(`layers.${layer}.ffn.gate.indices`)!.data!;
  const values = Array.from({ length: TOY_IDS[0].length }, (_, t) =>
    Array.from({ length: TOY.experts }, (_, e) =>
      Array.from(indices.slice(t * TOY.k, (t + 1) * TOY.k)).includes(e) ? 1 : 0,
    ),
  );
  const probabilities = ctx.trace.get(`layers.${layer}.attn.probabilities`)!;
  const columns = probabilities.shape.at(-1)!;
  const attention = Array.from({ length: TOY_IDS[0].length }, (_, t) =>
    Array.from(probabilities.data!.slice((t * TOY.h + head) * columns, (t * TOY.h + head + 1) * columns)),
  );
  const norms = Array.from({ length: TOY.layers }, (_, i) => {
    const a = ctx.trace.get(`layers.${i}.output`)!.data!;
    return Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  });
  return (
    <Section
      id="run"
      title="Execute the ownership chain"
      source={[{ source: "reference", symbol: "Transformer.forward" }]}
      lede="The clean text backbone runs in your browser at toy width. It preserves every backbone layer and source boundary, with random weights and explicit synthetic lookup IDs."
    >
      <Slider label="Backbone weight seed" value={seed} onChange={setSeed} min={1} max={f.window.value} />
      <Figure
        label="Architecture figure"
        caption="Computed expert choices for each toy token. Changing the layer selection or seed changes the stream reaching the router; these are not checkpoint outputs."
      >
        <Heatmap title="Toy expert dispatch" values={values} rowLabel="token position" columnLabel="toy expert" />
      </Figure>
      <Slider label="Toy attention head" value={head} onChange={setHead} min={0} max={f.toySpec.value.h - 1} />
      <Figure
        label="Computed attention probabilities"
        caption="Rows are queries. Columns start with raw token positions, then compressed positions, and end with the sink. Unreachable positions have zero probability; the sink contributes no value."
      >
        <Heatmap
          title="Reached positions and denominator sink"
          values={attention}
          rowLabel="query position"
          columnLabel="KV position / sink"
        />
      </Figure>
      <Sentence>
        The first {n(f.toyTokens.value)} columns are raw positions; the next {n(columns - f.toyTokens.value - 1)} are
        compressed groups; the last is the sink. Every row sums to unity, including the sink.
      </Sentence>
      <Sentence>
        At layer {n(layer)}, {n(TOY.k)} toy routed experts run per token. The current stream RMS is {n(norms[layer])}.
      </Sentence>
      <Readout
        label="Current scenario"
        items={[
          { term: "Toy backbone layers", value: n(TOY.layers) },
          { term: "Toy language width", value: n(TOY.d) },
          { term: "Toy stored parameters", value: n(ctx.parameterCount()) },
          { term: "Logits shape", value: logits.shape.map(n).join(" × ") },
        ]}
      />
      <Prose>
        The runnable source is <code>reference/model.ts</code>. In shape mode, the same forward pass requests the full
        logical backbone weights and checks its count against the independently derived ledger. Toy lookups substitute
        IDs at the tokenizer boundary; they do not change the gated lookup computation.
      </Prose>
      <Boundary>
        Numerical parity with upstream is unverified. The local Python has CPU-only PyTorch and no TileLang; the
        publisher requires GPU kernels. This run uses ideal floating arithmetic and omits FP8/FP4 rounding, distributed
        execution, decode caches, tokenizer hashing, vision and dSpark.
      </Boundary>
      <Check question="What does a finite toy run establish about the trained checkpoint?">
        It establishes that this clean floating program executes its arithmetic and shapes at toy sizes. It does not
        establish trained behavior, quantized numerical parity, or GPU performance.
      </Check>
      <Takeaways
        items={[
          "The browser computes these choices rather than drawing a synthetic routing pattern.",
          "Source and index ownership boundaries remain active at toy width.",
          "Shape agreement and numerical checkpoint parity are separate evidence.",
        ]}
      />
    </Section>
  );
}
