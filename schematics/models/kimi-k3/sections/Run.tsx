import {
  Section,
  Figure,
  Heatmap,
  Table,
  Slider,
  Button,
  Sentence,
  Readout,
  Prose,
  Takeaways,
  Check,
  useUrlNumber,
  ref,
  n,
  shape,
} from "../../../kit";
import * as f from "../facts";
import { forward } from "../reference/model";
export function Run() {
  const [seed, setSeed] = useUrlNumber("run-seed", 1, 0, f.DEMO.seedMax);
  const [layer, setLayer] = useUrlNumber("toy-layer", 0, 0, f.TOY.layers - 1);
  const [token, setToken] = useUrlNumber("toy-token", 0, 0, f.DEMO.tokens - 1);
  const ctx = new ref.Ctx("value", null, seed);
  const logits = forward(ctx, f.TOY, [Array.from({ length: f.DEMO.tokens }, (_, i) => i)]);
  const kda = f.TOY.kda.includes(layer + 1);
  const state = ctx.trace.get(`h${layer}.kda.state`);
  const attention = ctx.trace.get(`h${layer}.mla.weights`);
  const depth = ctx.trace.get(`h${layer}.mlp-res.depth-weights`)!;
  const routing = ctx.trace.get(`language_model.model.layers.${layer}.block_sparse_moe.gate.ids`);
  const scores = ctx.trace.get(`language_model.model.layers.${layer}.block_sparse_moe.gate.scores`);
  const recurrence = ctx.trace.get(`h${layer}.kda.output`);
  const matrix = kda
    ? Array.from({ length: f.TOY.dim }, (_, i) =>
        Array.from({ length: f.TOY.dim }, (_, j) => state!.data![i * f.TOY.dim + j]),
      )
    : Array.from({ length: f.DEMO.tokens }, (_, i) =>
        Array.from({ length: f.DEMO.tokens }, (_, j) => attention!.data![i * f.DEMO.tokens + j]),
      );
  const magnitude = Math.max(...matrix.flat().map(Math.abs));
  const depthN = depth.shape[2];
  return (
    <Section
      id="run"
      title="Run the scalar reference"
      source={[
        { source: "reference", symbol: "KimiDecoderLayer._forward_attn_residual" },
        { source: "kda-math", symbol: "naive_recurrent_kda" },
      ]}
      lede="The text decoder executes in your browser at toy width with seeded random weights. These computed tensors teach the mechanisms; they are not trained-checkpoint outputs."
    >
      <Slider label="Toy layer" min={0} max={f.TOY.layers - 1} value={layer} onChange={setLayer} />
      <Slider label="Inspect token" min={0} max={f.DEMO.tokens - 1} value={token} onChange={setToken} />
      <Button onClick={() => setSeed((seed + 1) % (f.DEMO.seedMax + 1))}>Draw new weights</Button>
      <Sentence>
        Seed {n(seed)}, toy layer {n(layer)} uses {kda ? "KDA recurrent memory" : "causal full attention"}.{" "}
        {routing
          ? `Token ${n(token)} selects experts ${Array.from(
              routing.data!.slice(token * f.TOY.topK, (token + 1) * f.TOY.topK),
            )
              .map(n)
              .join(", ")}.`
          : "This dense prefix layer has no router."}
      </Sentence>
      <Figure
        label="Computed token-memory tensor"
        caption={
          kda
            ? "Magnitude of final recurrent memory in the first toy head, scaled by its current maximum absolute entry. Exact signed entries appear below; these colors are not probabilities."
            : "Attention probabilities of the first toy head; upper-triangle entries are exactly zero from the causal mask."
        }
      >
        <Heatmap
          title={kda ? "KDA final state" : "Full-attention probabilities"}
          values={kda ? matrix.map((row) => row.map((v) => (magnitude ? Math.abs(v) / magnitude : 0))) : matrix}
          rowLabel={kda ? "key channel" : "query"}
          columnLabel={kda ? "value channel" : "key"}
          marked={kda ? undefined : token}
        />
      </Figure>
      {kda && (
        <Table
          caption="Signed recurrent state (first head)"
          head={["Key channel", ...matrix[0].map((_, i) => `Value ${n(i)}`)]}
          rows={matrix.map((row, i) => [n(i), ...row.map(n)])}
        />
      )}
      <Prose>
        The next heatmap is the feed-forward depth mixture at every token, from that same execution. Columns are
        completed blocks followed by the live prefix. It is independent of the token attention softmax above.
      </Prose>
      <Figure
        label="Computed depth mixture"
        caption="Each row sums to unity across depth candidates, not across token positions."
      >
        <Heatmap
          title="Depth probabilities"
          values={Array.from({ length: f.DEMO.tokens }, (_, i) =>
            Array.from(depth.data!.slice(i * depthN, (i + 1) * depthN)),
          )}
          rowLabel="token"
          columnLabel="depth"
          marked={token}
        />
      </Figure>
      <Readout
        label="Computed values"
        items={[
          { term: "Toy dimensions", value: `${n(f.TOY.layers)} layers × ${n(f.TOY.hidden)} width` },
          { term: "Logical toy parameters", value: n(ctx.parameterCount()) },
          { term: "Logits shape", value: shape(...logits.shape) },
          { term: "Token logit", value: n(logits.data![token * f.TOY.vocab]) },
          {
            term: "First expert score",
            value: scores ? n(scores.data![token * f.TOY.experts]) : "dense layer: no router",
          },
          {
            term: kda ? "Recurrence output" : "Attention row sum",
            value: n(
              kda ? recurrence!.data![token * f.TOY.heads * f.TOY.dim] : matrix[token].reduce((a, b) => a + b, 0),
            ),
          },
        ]}
      />
      <Prose>
        CPU component fixtures check the publisher's router, latent MoE, full-attention module and depth mixture, plus
        FLA's naive recurrence and gate. Whole-decoder logits and fused GPU KDA numerical parity are unverified: this
        environment lacks FLA and FlashAttention and has CPU-only PyTorch. The browser recurrence follows the separately
        pinned naive equations.
      </Prose>
      <Check question="Why is the depth heatmap not an attention map over previous tokens?">
        Its columns identify depth candidates at the same token. Sequence communication occurs inside KDA or full
        attention, not this mixture.
      </Check>
      <Takeaways
        items={[
          "Changing the random seed changes computed state and routing, with all state preserved in the URL.",
          "The toy keeps both attention kinds, latent routed/shared experts, SiTU and depth block boundaries active.",
          "A component numerical check does not establish whole-model checkpoint parity.",
        ]}
      />
    </Section>
  );
}
