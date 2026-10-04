import {
  Section,
  Figure,
  Heatmap,
  Slider,
  Button,
  Sentence,
  Readout,
  Prose,
  Sub,
  Takeaways,
  Check,
  ref,
  useUrlNumber,
  n,
  shape,
} from "../../../kit";
import * as f from "../facts";
import { forward } from "../reference/model";
import fixture from "../reference/fixture.json";
const LAYOUT = { pitch: 56, bar: 32, left: 24, base: 158, height: 100, shared: 510 };
export function Run() {
  const [seed, setSeed] = useUrlNumber("run-seed", 1, 0, f.demo.seedMax);
  const [layer, setLayer] = useUrlNumber("run-layer", f.toy.firstDense, 0, f.toy.layers - 1);
  const [head, setHead] = useUrlNumber("run-head", 0, 0, f.toy.heads - 1);
  const [token, setToken] = useUrlNumber("run-token", 0, 0, fixture.input_ids[0].length - 1);
  const ctx = new ref.Ctx("value", null, seed, f.demo.gain);
  const logits = forward(ctx, f.toy, fixture.input_ids);
  const seq = fixture.input_ids[0].length;
  const weights = ctx.trace.get(`h${layer}.attn.weights`)!;
  const values = Array.from({ length: seq }, (_, r) =>
    Array.from(weights.data!.subarray((head * seq + r) * seq, (head * seq + r + 1) * seq)),
  );
  const latent = ctx.trace.get(`h${layer}.attn.latent`)!;
  const latentRow = Array.from(latent.data!.subarray(token * f.toy.kvRank, (token + 1) * f.toy.kvRank));
  const routerLayer = Math.max(f.toy.firstDense, layer);
  const scores = ctx.trace.get(`h${routerLayer}.mlp.scores`)!.data!;
  const indices = ctx.trace.get(`h${routerLayer}.mlp.indices`)!.data!;
  const rw = ctx.trace.get(`h${routerLayer}.mlp.weights`)!.data!;
  const routing = {
    scores: Array.from(scores.subarray(token * f.toy.experts, (token + 1) * f.toy.experts)),
    chosen: Array.from(indices.subarray(token * f.toy.topK, (token + 1) * f.toy.topK)),
    weights: Array.from(rw.subarray(token * f.toy.topK, (token + 1) * f.toy.topK)),
    chosenGroups: [0],
  };
  return (
    <Section
      id="run"
      title="Run the forward pass"
      source={[
        { source: "reference", symbol: "DeepseekV3Attention.forward" },
        { source: "reference", symbol: "DeepseekV3MoE.forward" },
      ]}
      lede="A small decoder runs in your browser with random weights. It preserves latent attention, YaRN and corrected expert routing; these are computed tensors, not trained-model outputs."
    >
      <Slider label="Toy layer" value={layer} onChange={setLayer} min={0} max={f.toy.layers - 1} />
      <Slider label="Attention head" value={head} onChange={setHead} min={0} max={f.toy.heads - 1} />
      <Slider label="Token position" value={token} onChange={setToken} min={0} max={seq - 1} />
      <Button onClick={() => setSeed((seed + 1) % (f.demo.seedMax + 1))}>Draw new weights</Button>
      <Sentence>
        Seed {n(seed)}, layer {n(layer)}, head {n(head)}, token {n(token)}: its compressed latent is [
        {latentRow.map((v) => v.toFixed(f.demo.radDigits)).join(", ")}]. Changing the seed recomputes the whole decoder.
      </Sentence>
      <Sub>Attention reconstructed from the latent</Sub>
      <Figure
        label="Computed toy attention"
        caption="The triangular zeros come from the causal mask. The latent below is the content half of the latent-plus-positional-key cache that the pinned serving backend is given in place of expanded head keys and values; the layer that writes that cache is not pinned here."
      >
        <Heatmap title="Computed attention weights" values={values} rowLabel="query" columnLabel="key" marked={token} />
      </Figure>
      <Readout
        label="Computed values"
        items={[
          {
            term: "Content latent",
            value: latentRow.map((v) => v.toFixed(f.demo.radDigits)).join(" "),
            note: `${n(f.toy.kvRank)} scalars at the selected token`,
          },
          {
            term: "Toy cache representation",
            value: n(f.toy.kvRank + f.toy.rope),
            note: `versus ${n(f.toy.heads * (f.toy.nope + f.toy.rope + f.toy.v))} expanded scalars`,
          },
          { term: "Logits", value: shape(...logits.shape), note: "full random-weight decoder output" },
        ]}
      />
      <Sub>The same run chooses experts</Sub>
      <Prose>
        {layer < f.toy.firstDense
          ? `Selected layer ${n(layer)} is dense, so the router shown is from expert layer ${n(routerLayer)}.`
          : `The router below is from the selected expert layer ${n(routerLayer)}.`}{" "}
        Only the first batch row is visualized; the parity fixture also tests another batch row.
      </Prose>
      <Figure
        label="Computed toy routing"
        caption="These scores, selected experts and mixture weights come from the decoder's actual normalized stream. The shared expert remains separate."
      >
        <svg
          className="ms-fig"
          role="img"
          aria-label="Computed toy expert scores and shared path"
          width={700}
          height={218}
          viewBox="0 0 700 218"
        >
          <text className="ms-fig__column" x={LAYOUT.left} y={20}>
            computed routed scores
          </text>
          {routing.scores.map((score, index) => {
            const h = (score / Math.max(...routing.scores)) * LAYOUT.height;
            return (
              <g key={index}>
                <rect
                  className="ms-router__bar"
                  data-chosen={routing.chosen.includes(index) ? "true" : undefined}
                  x={LAYOUT.left + index * LAYOUT.pitch}
                  y={LAYOUT.base - h}
                  width={LAYOUT.bar}
                  height={h}
                />
                <text
                  className="ms-fig__note"
                  textAnchor="middle"
                  x={LAYOUT.left + index * LAYOUT.pitch + LAYOUT.bar / 2}
                  y={LAYOUT.base + 22}
                >
                  {n(index)}
                </text>
                <text
                  className="ms-node__line"
                  textAnchor="middle"
                  x={LAYOUT.left + index * LAYOUT.pitch + LAYOUT.bar / 2}
                  y={LAYOUT.base + 40}
                >
                  {score.toFixed(f.demo.radDigits)}
                </text>
              </g>
            );
          })}
          <text className="ms-fig__column" x={LAYOUT.shared} y={20}>
            shared · always on
          </text>
          <rect
            className="ms-router__bar"
            data-shared="true"
            x={LAYOUT.shared}
            y={LAYOUT.base - LAYOUT.height}
            width={LAYOUT.bar}
            height={LAYOUT.height}
          />
          <text className="ms-fig__note" x={LAYOUT.shared} y={LAYOUT.base + 22}>
            no router logit
          </text>
        </svg>
      </Figure>
      <Sentence>
        Token {n(token)} in expert layer {n(routerLayer)} goes to experts {routing.chosen.map(n).join(", ")} with total
        routed weight {routing.weights.reduce((a, b) => a + b, 0).toFixed(f.demo.radDigits)}.
      </Sentence>
      <Readout
        label="Computed values"
        items={[
          {
            term: "Toy decoder",
            value: `${n(f.toy.layers)} × ${n(f.toy.hidden)}`,
            note: `layers × stream width; ${n(f.toy.heads)} heads`,
          },
          { term: "Parameters", value: n(ctx.parameterCount()), note: "counted by the same clean forward" },
          {
            term: "Chosen expert weights",
            value: routing.weights.map((w) => w.toFixed(f.demo.radDigits)).join(" "),
            note: "corrected selection, original-score weights",
          },
        ]}
      />
      <Prose>
        The toy uses {n(f.toy.firstDense)} dense layers and {n(f.toy.layers - f.toy.firstDense)} expert layers to
        exercise repeated blocks. The shipped checkpoint has its own layer boundary. CPU tests load perturbed weights
        from the pinned publisher modules, compare logits and intermediates, and require every parameter name to match.
      </Prose>
      <Check question="Do the random router's favorite experts tell us which experts the trained checkpoint prefers?">
        No. The random experiment demonstrates the selection and mixture rule only. Its weights and token vectors do not
        come from the checkpoint.
      </Check>
      <Takeaways
        items={[
          "Both the latent and the routing figure are computed by the same decoder execution.",
          "Changing head changes the attention view; changing token changes the displayed latent and expert assignment.",
          "Numerical agreement with a toy upstream run verifies arithmetic, not FP8 checkpoint or GPU parity.",
        ]}
      />
    </Section>
  );
}
