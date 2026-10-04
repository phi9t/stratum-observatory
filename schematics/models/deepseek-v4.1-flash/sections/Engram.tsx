import {
  Section,
  OpGraph,
  Prose,
  Table,
  Takeaways,
  Check,
  n,
  Slider,
  Sentence,
  Readout,
  useUrlNumber,
  Figure,
  Heatmap,
} from "../../../kit";
import * as f from "../facts";
import { engramGraph } from "../graphs";
import { Ctx } from "../../../kit/ref/tensor";
import { Engram as runEngram, TOY } from "../reference/model";
export function Engram() {
  const [hash, setHash] = useUrlNumber("lookup", 0, 0, TOY.engramRows[0] - 1);
  const ctx = new Ctx("value", null, 1, 2);
  const input = {
    shape: [1, 1, TOY.hc, TOY.d],
    data: Float32Array.from({ length: TOY.hc * TOY.d }, (_, i) => Math.sin(i)),
  };
  const output = runEngram(ctx, TOY, "engram", input, [Array(TOY.hashColumns).fill(hash)], TOY.engramRows[0]);
  const changes = Array.from({ length: TOY.hc }, (_, h) =>
    Array.from({ length: TOY.d }, (_, d) => Math.abs(output.data![h * TOY.d + d] - input.data[h * TOY.d + d])),
  );
  const peak = Math.max(...changes.flat(), 1);
  return (
    <Section
      id="engram"
      title="A lookup added before mixing"
      source={[
        { source: "reference", symbol: "Engram.forward" },
        { source: "engram", symbol: "NgramHashState.forward" },
      ]}
      lede="Token n-grams retrieve hash-table rows. A gate compares the lookup key with each residual copy before adding a shared value."
    >
      <Prose>
        Engram runs at layers {f.engramLayers.value.map(n).join(", ")}. The hash path normalizes token strings, maps
        compressed token IDs, XORs multiplied look-back IDs, takes each prime modulus, then adds bucket offsets. Images
        stop look-back and receive no Engram contribution.
      </Prose>
      <Table
        caption="Hash representation"
        head={["Quantity", "Derived shape or count"]}
        rows={[
          ["Maximum n-gram", n(f.ngram.value)],
          ["Hash columns", n(f.hashColumns.value)],
          ["Concatenated lookup width", n(f.hashWidth.value)],
          ["Table rows by layer", f.engramRows.value.map(n).join(", ")],
          ["Key / value projection width", n((f.copies.value + 1) * f.hidden.value)],
        ]}
      />
      <OpGraph id="engram-ops" title="Engram forward" {...engramGraph()} />
      <Slider label="Toy lookup row" min={0} max={TOY.engramRows[0] - 1} value={hash} onChange={setHash} />
      <Figure
        label="Architecture figure"
        caption="Absolute per-channel change divided by the peak. Floating arithmetic with random weights and explicit lookup IDs; these IDs are not publisher tokenizer hashes."
      >
        <Heatmap
          title="Gated lookup contribution"
          values={changes.map((row) => row.map((v) => v / peak))}
          rowLabel="residual copy"
          columnLabel="toy channel"
        />
      </Figure>
      <Sentence>Lookup row {n(hash)} changes every copy through its own normalized-key gate.</Sentence>
      <Readout
        label="Current scenario"
        items={[
          { term: "Peak absolute update", value: n(Math.max(...changes.flat())) },
          { term: "Checkpoint lookup width", value: n(f.hashWidth.value) },
        ]}
      />
      <Check question="Does Engram normalize across all residual copies at once?">
        No. Its similarity gate normalizes each copy and key over channels separately. Hyper-connection coefficient
        generation later normalizes the flattened copy-and-channel stream.
      </Check>
      <Takeaways
        items={[
          "Hash lookups are input-dependent retrieval, not another attention cache.",
          "Every residual copy has a separate key and gate, but the value is shared.",
          "The clean browser run requires explicit lookup IDs; it does not pretend to implement int64 token hashing.",
        ]}
      />
    </Section>
  );
}
