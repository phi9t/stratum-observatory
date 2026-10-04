import {
  Section,
  Sub,
  OpGraph,
  Prose,
  Figure,
  Slider,
  Sentence,
  Readout,
  Takeaways,
  Check,
  n,
  useUrlNumber,
} from "../../../kit";
import { ToyRouter } from "../widgets";
import * as f from "../facts";
import { routerGraph, expertGraph, graph, stream } from "../graphs";
import { Ctx } from "../../../tnsr/tensor";
import { forward, TOY, TOY_IDS, toyHashes } from "../reference/model";
import { useLayer } from "./Stack";
export function Experts() {
  const [layer] = useLayer();
  const [seed, setSeed] = useUrlNumber("routerseed", f.copies.value, 1, f.window.value);
  const [token, setToken] = useUrlNumber("routertoken", 0, 0, f.toyTokens.value - 1);
  const ctx = new Ctx("value", null, seed, 2);
  forward(ctx, TOY, TOY_IDS, toyHashes(TOY, TOY_IDS));
  const prefix = `layers.${layer}.ffn.gate`;
  const scores = Array.from(
    ctx.trace.get(prefix + ".scores")!.data!.slice(token * TOY.experts, (token + 1) * TOY.experts),
  );
  const chosen = Array.from(ctx.trace.get(prefix + ".indices")!.data!.slice(token * TOY.k, (token + 1) * TOY.k));
  const weights = Array.from(ctx.trace.get(prefix + ".weights")!.data!.slice(token * TOY.k, (token + 1) * TOY.k));
  return (
    <Section
      id="experts"
      title="Select with bias, weight without it"
      source={[
        { source: "reference", symbol: "Gate.forward" },
        { source: "reference", symbol: "MoE.forward" },
      ]}
      lede="The router chooses a small set of experts, while a shared expert processes every token. Correction bias changes selection, not the gathered weights."
    >
      <OpGraph
        id="moe-topology"
        title="Routed and shared paths"
        {...graph(
          [
            {
              id: "input",
              label: "normalized input",
              width: f.hidden.value,
              output: stream,
              what: "Same input reaches router, routed FFNs and shared FFN.",
              formula: "x = FFN RMSNorm output",
              col: 0,
            },
            {
              id: "routing",
              label: "router + dispatch",
              width: f.topK.value,
              output: `top-${n(f.topK.value)}`,
              what: "Score and select experts; dispatch token rows.",
              formula: "ids, weights = Gate(x)",
              col: 1,
            },
            {
              id: "routed",
              label: "routed FFNs",
              width: f.hidden.value,
              output: stream,
              what: "Only selected experts run, weighted before their down projections.",
              formula: "routed = Σe Expert_e(x,w_e)",
              col: 2,
            },
            {
              id: "shared",
              label: "shared FFN",
              width: f.hidden.value,
              output: stream,
              what: "Every token runs this expert without a router logit or routing weight.",
              formula: "shared = Expert_shared(x)",
              col: 2,
              row: 1,
              input: stream,
            },
            {
              id: "sum",
              label: "accumulate + add",
              width: f.hidden.value,
              output: stream,
              what: "Reduce routed partial sums, add shared output, restore token shape.",
              formula: "y = all_reduce(routed) + shared",
              col: 3,
            },
          ],
          "MoE.forward",
          "DeepseekV4DecoderLayer",
        )}
        edges={[
          { from: "input", to: "routing" },
          { from: "input", to: "routed" },
          { from: "routing", to: "routed", kind: "state" },
          { from: "input", to: "shared" },
          { from: "routed", to: "sum" },
          { from: "shared", to: "sum" },
        ]}
      />
      <Sub>The router</Sub>
      <Prose>
        Text tokens use the text selection bias; image spans have a separate correction bias. The publisher does no
        expert-group pruning.
      </Prose>
      <OpGraph id="router" title="Router arithmetic" {...routerGraph()} />
      <Sub>Inside an expert</Sub>
      <Prose>
        Gate and up projections run in parallel. The shared expert has the same FFN arithmetic, without the
        routing-weight operation.
      </Prose>
      <OpGraph id="expert" title="Clamped SwiGLU expert" {...expertGraph()} />
      <Sub>A computed routing example</Sub>
      <Slider label="Routing weight seed" value={seed} onChange={setSeed} min={1} max={f.window.value} />
      <Slider label="Token position" value={token} onChange={setToken} min={0} max={f.toyTokens.value - 1} />
      <Figure
        label="Architecture figure"
        caption="Computed router scores from the toy backbone’s random weights and stream. Toy expert counts are reduced; the selection-bias and weight rules are unchanged. The shared expert has no router score."
      >
        <ToyRouter scores={scores} chosen={chosen} weights={weights} />
      </Figure>
      <Sentence>
        Token {n(token)} at layer {n(layer)} selects toy experts {chosen.map(n).join(", ")}; their weights sum to{" "}
        {n(weights.reduce((a, b) => a + b, 0))}.
      </Sentence>
      <Readout
        label="Current scenario"
        items={[
          { term: "Checkpoint routed", value: `${n(f.topK.value)} / ${n(f.experts.value)}` },
          { term: "Shared", value: n(f.shared.value) },
          { term: "Active routed fraction", value: n(f.activeFraction.value) },
        ]}
      />
      <Check question="Could a large correction bias give an expert a large output weight even when its raw score is small?">
        It can cause selection, but its output weight is gathered from the raw score, then normalized against the other
        selected scores. The bias never enters that weight.
      </Check>
      <Takeaways
        items={[
          `Each token activates ${n(f.topK.value)} of ${n(f.experts.value)} routed experts plus the always-on shared expert.`,
          "Gate clipping is upper-only; up clipping is symmetric.",
          "The routing weight multiplies the activated hidden vector before w2.",
        ]}
      />
    </Section>
  );
}
