import {
  Section,
  OpGraph,
  Sub,
  Prose,
  Figure,
  Heatmap,
  Slider,
  Sentence,
  Readout,
  Takeaways,
  Check,
  n,
  useUrlNumber,
} from "../../../kit";
import * as f from "../facts";
import { boundaryGraph, mixingGraph } from "../graphs";
import { useLayer } from "./Stack";
import { Ctx } from "../../../kit/ref/tensor";
import { forward, TOY, TOY_IDS, toyHashes } from "../reference/model";
export function Block() {
  const [layer] = useLayer();
  const [seed, setSeed] = useUrlNumber("hcseed", 1, 1, f.window.value);
  const ctx = new Ctx("value", null, seed, 2);
  forward(ctx, TOY, TOY_IDS, toyHashes(TOY, TOY_IDS));
  const matrix = ctx.trace.get(`layers.${layer}.attn.comb`)!.data!;
  const rows = Array.from({ length: TOY.hc }, (_, i) => Array.from(matrix.slice(i * TOY.hc, (i + 1) * TOY.hc)));
  return (
    <Section
      id="block"
      title="A delayed collapse, a current residual mix"
      source={[{ source: "reference", symbol: "Block.forward" }]}
      lede="A sublayer builds three coefficient sets. Its expansion and residual matrix apply now; its collapse weights are consumed by the next sublayer."
    >
      <Sentence>
        Layer {n(layer)}: {f.layerDescription(layer)}
      </Sentence>
      <Sub>Attention boundary</Sub>
      <Prose>
        The attention path reads incoming collapse coefficients. Its post operation also reads the original residual
        copies and the current attention coefficients.
      </Prose>
      <OpGraph id="block-attn" title="Attention boundary" {...boundaryGraph(layer, "attn")} />
      <Sub>Expert boundary</Sub>
      <Prose>
        The FFN collapse uses the attention pre coefficients. The FFN post operation mixes its saved residual and
        publishes the next block’s collapse coefficients.
      </Prose>
      <OpGraph id="block-ffn" title="FFN boundary" {...boundaryGraph(layer, "ffn")} />
      <Sub>Where coefficients come from</Sub>
      <Prose>
        This side computation runs on the saved residual before each sublayer. The matrix is normalized over copy axes,
        not channel axes.
      </Prose>
      <OpGraph id="mixes" title="Hyper-connection coefficients" {...mixingGraph()} />
      <Slider label="Mixing weight seed" value={seed} onChange={setSeed} min={1} max={f.window.value} />
      <Figure
        label="Architecture figure"
        caption="Computed from the clean floating reference at toy width, with random weights. Rows are source copies; columns are destination copies."
      >
        <Heatmap title="Residual mixing matrix" values={rows} rowLabel="source copy" columnLabel="destination copy" />
      </Figure>
      <Sentence>
        Layer {n(layer)}, seed {n(seed)}: each copy can contribute to every destination; this is not an identity skip.
      </Sentence>
      <Readout
        label="Current scenario"
        items={[
          { term: "Iterations", value: n(f.iterations.value) },
          { term: "Matrix shape", value: `${n(f.copies.value)} × ${n(f.copies.value)}` },
        ]}
      />
      <Check question="Can the attention input be collapsed using the pre weights just computed by that attention sublayer?">
        No. The publisher uses the incoming pre weights; its newly computed attention pre is handed to the FFN.
        Replacing this hand-off changes the forward pass.
      </Check>
      <Takeaways
        items={[
          "Residual mixing coefficients depend on the whole flattened stream.",
          "The FFN’s pre coefficients survive the layer boundary.",
          "Sinkhorn makes row and column sums approximately unit, rather than replacing the matrix with identity.",
        ]}
      />
    </Section>
  );
}
