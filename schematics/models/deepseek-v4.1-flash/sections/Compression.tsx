import {
  Section,
  Sub,
  Prose,
  OpGraph,
  Sentence,
  Slider,
  Readout,
  Takeaways,
  Check,
  n,
  useUrlNumber,
  Figure,
  TensorScale,
  Table,
  Heatmap,
} from "../../../kit";
import { candidateMask } from "../reference/ops";
import * as f from "../facts";
import { compressedCacheGraph, compressionGraph, indexQueryGraph, indexKeyGraph, selectionGraph } from "../graphs";
import { useLayer } from "./Stack";
export function Compression() {
  const [layer] = useLayer();
  const [context, setContext] = useUrlNumber("compresscontext", f.demoContext.value, 1, f.context.value);
  const [visible, setVisible] = useUrlNumber("candidatevisible", f.toyTokens.value, 1, f.toyTokens.value);
  const candidateScores = Array.from({ length: f.toyTokens.value }, (_, i) => f.toyTokens.value - i);
  const candidate = candidateMask(candidateScores, visible, 1, f.toySpec.value.candidateBlock);
  const source = f.owner(layer, f.kvSources.value) ?? f.kvSources.value[0],
    ratio = f.ratios.value[source];
  const index = f.owner(layer, f.indexSources.value) ?? f.indexSources.value[0];
  const complete = Math.floor(context / ratio);
  return (
    <Section
      id="compression"
      title="Compress at four layers, re-index at eight, reuse in between"
      source={[
        { source: "reference", symbol: "Compressor" },
        { source: "reference", symbol: "Indexer" },
      ]}
      lede="Completed token groups become shared KV vectors. A smaller side attention selects which compressed positions the main attention may read."
    >
      <Sentence>
        {f.ratios.value[layer]
          ? `Selected layer ${n(layer)} reads KV from ${n(source)} and indices from ${n(index)}.`
          : `Selected layer ${n(layer)} is window-only. The source path below is the first compressed bundle, layer ${n(source)}.`}
      </Sentence>
      <Sub>The compressed KV owner</Sub>
      <Prose>
        The ratio determines whether learned pooling is present. The indexer consumes the normalized latent before its
        tail is rotated and rounded to FP4 for the main cache.
      </Prose>
      <OpGraph id="compress" title="KV compressor" {...compressionGraph(source)} />
      <Slider label="Tokens processed" min={1} max={f.context.value} value={context} onChange={setContext} />
      <Figure
        label="Architecture figure"
        caption="Equal channel width; horizontal lengths show how many vectors exist. The window is bounded; the compressed cache grows with complete groups."
      >
        <TensorScale
          title="Completed compressed groups"
          pxPerToken={Math.min(2, 180 / context)}
          pxPerWidth={0.1}
          footnote="Raw tokens versus complete compressed groups; partial groups are excluded."
          stages={[
            {
              label: "Raw token span",
              tensors: [{ key: "raw", name: "raw KV", tokens: context, width: f.headDim.value, lane: "high" }],
            },
            {
              label: "Compressed owner",
              tensors: [
                {
                  key: "compressed",
                  from: "raw",
                  name: "compressed KV",
                  tokens: complete,
                  width: f.headDim.value,
                  lane: "instance",
                },
              ],
            },
            {
              label: "Local window",
              tensors: [
                {
                  key: "window",
                  name: "window KV",
                  tokens: Math.min(context, f.window.value),
                  width: f.headDim.value,
                  lane: "instance",
                },
              ],
            },
          ]}
        />
      </Figure>
      <Sentence>
        {n(context)} processed tokens yield {n(complete)} complete groups at ratio {n(ratio)}. {n(context % ratio)}{" "}
        tokens wait in partial-group state.
      </Sentence>
      <Readout
        label="Current scenario"
        items={[
          { term: "Compressed payload", value: `${n(complete * f.headDim.value)} scalars` },
          { term: "Maximum selected", value: n(Math.min(complete, f.indexTopK.value)) },
          { term: "Pooling reduction", value: `${n(ratio)}× fewer vectors` },
        ]}
      />
      <Sub>Indexer keys: owned with KV</Sub>
      <Prose>
        Only KV source layers build these keys. A later index source reuses them while rebuilding queries and head
        weights.
      </Prose>
      <OpGraph id="index-key" title="Indexer key path" {...indexKeyGraph()} />
      <Sub>After index keys: publish main KV</Sub>
      <Prose>
        The main compressed cache has its own quantization format. Its write follows index-key construction, so keys
        never read the main cache’s already rotated storage.
      </Prose>
      <OpGraph id="compressed-cache" title="Compressed KV cache publication" {...compressedCacheGraph(source)} />
      <Sub>Indexer queries and weights</Sub>
      <Prose>
        The head-weight projection reads the original normalized attention input, while the query projection reads its
        low-rank latent. These are parallel inputs.
      </Prose>
      <OpGraph
        id="index-query"
        title="Indexer query path"
        {...indexQueryGraph()}
        edges={[
          { from: "iq", to: "iqrope" },
          { from: "iqrope", to: "iqquant" },
        ]}
      />
      <Sub>Reachability and selection</Sub>
      <Prose>
        The candidate source keeps high-scoring blocks and pins the newest reachable block. Later indexers restrict
        their selection to that mask; reuse layers read the published result.
      </Prose>
      <OpGraph id="index-select" title="Indexer selection" {...selectionGraph(index)} />
      <Sub>The candidate source’s extra step</Sub>
      <OpGraph id="candidate-select" title="Candidate source selection" {...selectionGraph(f.candidateSource.value)} />
      <Slider label="Toy completed positions" min={1} max={f.toyTokens.value} value={visible} onChange={setVisible} />
      <Figure
        label="Newest-block candidate rule"
        caption="Toy descending scores favor the oldest block. With a single block budget, the actual candidate rule overrides them to keep the newest reachable block. Future slots stay excluded."
      >
        <Heatmap
          title="Candidate admission"
          values={[candidateScores.map((v) => v / f.toyTokens.value), candidate.map((v) => (v ? 1 : 0))]}
          rowLabel="score / admitted"
          columnLabel="compressed position"
        />
      </Figure>
      <Sentence>
        At {n(visible)} complete positions, the newest reachable toy block owns positions{" "}
        {candidate
          .map((v, i) => (v ? n(i) : null))
          .filter((v) => v !== null)
          .join(", ")}
        . The checkpoint budget is {n(f.candidateTopK.value)} blocks of {n(f.candidateBlock.value)} positions.
      </Sentence>
      <Table
        caption="Independent state ownership"
        head={["State", "Source IDs", "Consumer rule"]}
        rows={[
          ["Compressed KV + index keys", f.kvSources.value.map(n).join(", "), "Read latest KV owner"],
          ["Selected positions", f.indexSources.value.map(n).join(", "), "Read latest index owner"],
          ["Candidate blocks", n(f.candidateSource.value), "Later indexers mask to these blocks"],
        ]}
      />
      <Check question="Can the indexer score a partially filled compression group?">
        No. The compressor publishes only complete groups, and the indexer masks positions using the completed-group
        count for each query.
      </Check>
      <Takeaways
        items={[
          "KV ownership and index ownership are different axes.",
          "Channel-wise pooling preserves vector width while reducing position count.",
          "Reused state crosses depth; ring and partial-group state cross decoding time.",
        ]}
      />
    </Section>
  );
}
