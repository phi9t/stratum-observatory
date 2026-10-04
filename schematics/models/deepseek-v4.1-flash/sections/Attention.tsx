import {
  Section,
  Sub,
  Prose,
  OpGraph,
  Figure,
  Slider,
  Sentence,
  Readout,
  Takeaways,
  Check,
  useUrlNumber,
  n,
} from "../../../kit";
import { RotaryDials } from "../widgets";
import * as f from "../facts";
import { queryGraph, windowGraph, attendGraph, outputGraph, graph, stream, latent, query } from "../graphs";
import { useLayer } from "./Stack";
export function Attention() {
  const [layer] = useLayer();
  const [pos, setPos] = useUrlNumber("ropeposition", 0, 0, f.context.value - 1);
  const compressed = !!f.ratios.value[layer];
  return (
    <Section
      id="attention"
      title="Shared KV, independent queries"
      source={[{ source: "reference", symbol: "Attention.forward" }]}
      lede="Window and compressed paths meet in a single attention normalization. A learned sink can absorb probability without contributing a value."
    >
      <Sentence>
        Layer {n(layer)} {compressed ? "reads both window and compressed state." : "reads its causal window only."}
      </Sentence>
      <OpGraph
        id="attn-topology"
        title="Attention topology"
        {...graph(
          [
            {
              id: "input",
              label: "attention input",
              width: f.hidden.value,
              output: stream,
              what: "Collapsed and normalized stream enters both projection paths.",
              formula: "x = normalized collapsed stream",
              col: 0,
              row: 0,
            },
            {
              id: "queries",
              label: "query path",
              width: f.qWidth.value,
              output: query,
              what: "Low-rank query projection and RoPE.",
              formula: "q = RoPE(Wq_b RMSNorm(Wq_a x))",
              col: 1,
              row: 0,
            },
            {
              id: "kvpath",
              label: compressed ? "window + compressed" : "window KV",
              width: f.headDim.value,
              output: latent,
              what: compressed
                ? "Combine local window with owner-built compressed state."
                : "Construct and cache the window KV.",
              formula: "kv = reachable shared KV",
              col: 1,
              row: 1,
              input: stream,
            },
            {
              id: "attend",
              label: "sparse attention",
              width: f.qWidth.value,
              output: query,
              what: "Queries attend over the reached KV set with a denominator sink.",
              formula: "o = attention(q,kv,sink)",
              col: 2,
              row: 0,
            },
            {
              id: "output",
              label: "inverse + output",
              width: f.hidden.value,
              output: stream,
              what: "Undo tail rotation, project per group then mix groups.",
              formula: "y = Wo_b grouped_Wo_a inverse_RoPE(o)",
              col: 3,
              row: 0,
            },
          ],
          "Attention.forward",
        )}
        edges={[
          { from: "input", to: "queries" },
          { from: "input", to: "kvpath" },
          { from: "queries", to: "attend" },
          { from: "kvpath", to: "attend" },
          { from: "attend", to: "output" },
        ]}
      />
      <Sub>Build queries</Sub>
      <Prose>
        The normalized latent also feeds the indexer at source layers. No extra per-head query norm appears in this
        publisher implementation.
      </Prose>
      <OpGraph id="query" title="Query projection" {...queryGraph()} />
      <Sub>Build the local window</Sub>
      <Prose>Keys and values are the same vector. Quantization follows RoPE; the cache retains the whole vector.</Prose>
      <OpGraph id="window" title="Window KV path" {...windowGraph()} />
      <Sub>Read reached positions</Sub>
      <Prose>
        Window indices enforce causality. Compressed indices admit only completed groups. Their scores are normalized
        together.
      </Prose>
      <OpGraph id="attend" title="Attention arithmetic" {...attendGraph()} />
      <Sub>Return to language width</Sub>
      <Prose>Each output group projects only its own heads. The final projection combines all group latents.</Prose>
      <OpGraph id="output" title="Grouped output projection" {...outputGraph()} />
      <Sub>Position enters only the tail</Sub>
      <Slider label="Rotary position" min={0} max={f.context.value - 1} value={pos} onChange={setPos} />
      <Figure
        label="Architecture figure"
        caption="The selected layer’s actual frequency rule: base frequencies for window-only layers, and the YaRN blend for compressed layers. Angles are computed before display."
      >
        <RotaryDials position={pos} compressed={compressed} />
      </Figure>
      <Sentence>
        Position {n(pos)} rotates the final {n(f.ropeDim.value)} channels. The attention output applies the inverse
        rotation.
      </Sentence>
      <Readout
        label="Current scenario"
        items={[
          { term: "Window base", value: n(f.ropeBase.value) },
          { term: "Compressed base", value: n(f.compressedBase.value) },
          { term: "YaRN factor", value: n(f.factor.value) },
        ]}
      />
      <Check question="If all reached positions lose probability to the sink, is the missing mass multiplied by another learned value?">
        No. The sink enters the denominator only. The weighted KV output can therefore have total attention mass below
        unity.
      </Check>
      <Takeaways
        items={[
          "The sink has no value vector.",
          "Compressed layers change frequencies for their window branch as well as their compressed branch.",
          "Inverse RoPE occurs before grouped output projection.",
        ]}
      />
    </Section>
  );
}
