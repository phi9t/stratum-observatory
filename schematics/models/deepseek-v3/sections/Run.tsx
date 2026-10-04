import { ToyRouter } from "./ToyRouter";
import { useMemo } from "react";
import {
  Section,
  Sub,
  Prose,
  Figure,
  TensorScale,
  Table,
  Sentence,
  Readout,
  Slider,
  Button,
  Check,
  Takeaways,
  n,
  shape,
  useUrlNumber,
} from "../../../kit";
import * as f from "../facts";
import * as ref from "../../../tnsr/tensor";
import { forward, TOY } from "../reference/model";

export function Run() {
  const [seed, setSeed] = useUrlNumber("run-seed", f.TOY.seed, 0, f.DEMO.maxSeed);
  const [layer, setLayer] = useUrlNumber("run-layer", f.TOY.firstDense, f.TOY.firstDense, f.TOY.layers - 1);
  const [head, setHead] = useUrlNumber("run-head", 0, 0, f.TOY.heads - 1);
  const [token, setToken] = useUrlNumber("run-token", 0, 0, f.TOY.tokens - 1);
  const { ctx, logits } = useMemo(() => {
    const ctx = new ref.Ctx("value", null, seed, f.TOY.gain);
    return { ctx, logits: forward(ctx, TOY, f.TOY.ids) };
  }, [seed]);
  const get = (op: string) => ctx.trace.get(`h${layer}.${op}`)!;
  const latent = get("kn"),
    keys = get("joink"),
    values = get("kvsplit");
  const ids = get("topk"),
    weights = get("routeweights"),
    groups = get("groups"),
    scores = get("sigmoid");
  const row = (t: ref.Tensor, start: number, width: number) => Array.from(t.data!.slice(start, start + width));
  const vector = (v: readonly number[]) => <code>[{v.map((x) => x.toFixed(f.DEMO.precision)).join(", ")}]</code>;
  const routes = Array.from({ length: f.TOY.tokens }, (_, t) => ({
    scores: row(scores, t * f.TOY.experts, f.TOY.experts),
    chosen: row(ids, t * f.TOY.topK, f.TOY.topK),
    chosenGroups: row(groups, t * f.TOY.topGroups, f.TOY.topGroups),
    weights: row(weights, t * f.TOY.topK, f.TOY.topK),
  }));
  const chosen = routes[token];
  const keyWidth = f.TOY.nope + f.TOY.rotary;
  return (
    <Section
      id="run"
      title="Run it"
      source={[
        { source: "reference", symbol: "DeepseekV3Attention.forward" },
        { source: "reference", symbol: "MoEGate.forward" },
      ]}
      lede="Run the clean reference in your browser. Random toy weights expose how a cached latent rebuilds head-specific keys and values, and how the grouped router sends each token to its experts."
    >
      <Prose>
        These are computed values from the complete toy forward pass, not learned checkpoint predictions. Reduced widths
        preserve both layer kinds, low-rank queries and KV, partial rotary positions, group selection and the separate
        shared expert. The same code passes a pinned upstream numerical fixture and counts the full-width main model.
      </Prose>
      <Slider label="Toy MoE layer" value={layer} min={f.TOY.firstDense} max={f.TOY.layers - 1} onChange={setLayer} />
      <Slider label="Toy head" value={head} min={0} max={f.TOY.heads - 1} onChange={setHead} />
      <Slider label="Toy token position" value={token} min={0} max={f.TOY.tokens - 1} onChange={setToken} />
      <Button onClick={() => setSeed((s) => (s + 1) % (f.DEMO.maxSeed + 1))}>Draw new weights</Button>
      <Sub>Keep a latent; rebuild keys and values</Sub>
      <Sentence>
        Seed {n(seed)}, layer {n(layer)}, head {n(head)}: position {n(token)} keeps a normalized latent of width{" "}
        {n(f.TOY.kvRank)} plus a shared rotated positional key of width {n(f.TOY.rotary)}. Expansion produces different
        content keys and values for each head.
      </Sentence>
      <Figure
        label="Toy compressed state versus expanded tensors"
        caption="To scale by scalars per token. The positional key bypasses the latent expansion and is shared across heads; the eager forward used below builds full keys and values. The figure identifies what compressed serving can keep, without claiming this full-sequence program implements persistent decode caching."
      >
        <TensorScale
          title="Toy latent and expanded key/value sizes"
          pxPerToken={24}
          pxPerWidth={6}
          footnote="Height is scalars per token; equal sequence lengths."
          stages={[
            {
              label: "compressed state",
              tensors: [
                { key: "c", name: "latent", tokens: f.TOY.tokens, width: f.TOY.kvRank, lane: "instance" },
                { key: "pe", name: "shared position", tokens: f.TOY.tokens, width: f.TOY.rotary, lane: "instance" },
              ],
            },
            {
              label: "rebuilt for all heads",
              tensors: [
                { key: "k", name: "full keys", tokens: f.TOY.tokens, width: f.TOY.heads * keyWidth, lane: "high" },
                {
                  key: "v",
                  name: "values",
                  from: "c",
                  tokens: f.TOY.tokens,
                  width: f.TOY.heads * f.TOY.valueDim,
                  lane: "high",
                },
              ],
            },
          ]}
        />
      </Figure>
      <Table
        caption={`Actual values at position ${n(token)}; displayed rounding only`}
        head={["Tensor", "Computed vector"]}
        rows={[
          ["Normalized cached latent", vector(row(latent, token * f.TOY.kvRank, f.TOY.kvRank))],
          ["Rotated shared positional key", vector(row(get("kpe"), token * f.TOY.rotary, f.TOY.rotary))],
          [`Rebuilt key · head ${n(head)}`, vector(row(keys, (head * f.TOY.tokens + token) * keyWidth, keyWidth))],
          [
            `Rebuilt value · head ${n(head)}`,
            vector(row(values, (head * f.TOY.tokens + token) * f.TOY.valueDim, f.TOY.valueDim)),
          ],
        ]}
      />
      <Prose>
        KV B expands the normalized latent into each head’s content key and value. The full key then appends the shared
        rotated slice. Move the head control: the latent and shared slice stay fixed while the rebuilt content changes.
        At checkpoint widths, compressed state keeps {n(f.kvAWidth.value)} rather than {n(f.eagerCacheWidth.value)}{" "}
        scalars per token per layer.
      </Prose>
      <Sub>Route the toy sequence</Sub>
      <Sentence>
        Position {n(token)} selects groups {chosen.chosenGroups.map(n).join(", ")} and experts{" "}
        {chosen.chosen.map(n).join(", ")}; its mixture weights sum to{" "}
        {chosen.weights.reduce((a, b) => a + b, 0).toFixed(f.DEMO.precision)}. The shared expert runs independently.
      </Sentence>
      <Figure
        label="Computed reference router"
        caption="The reference router reads the actual normalized toy stream. Bars show original sigmoid scores; correction bias affects group and expert selection only. Selected groups sum their strongest corrected scores, then expert top-k runs only inside the surviving groups."
      >
        <ToyRouter routing={chosen} />
      </Figure>
      <Table
        caption="Every token’s computed routes; experts and weights are paired in increasing expert id order"
        head={["Position / input id", "Groups kept", "Experts → mixture weights"]}
        current={token}
        rows={routes.map((r, t) => [
          <code>
            {n(t)} / {n(f.TOY.ids[0][t])}
          </code>,
          r.chosenGroups.map(n).join(", "),
          <code>{r.chosen.map((e, i) => `${n(e)} → ${r.weights[i].toFixed(f.DEMO.precision)}`).join("; ")}</code>,
        ])}
      />
      <Readout
        label="Executable toy instance"
        items={[
          {
            term: "Toy layers × width",
            value: `${n(f.TOY.layers)} × ${n(f.TOY.hidden)}`,
            note: `${n(f.TOY.firstDense)} dense, remaining MoE`,
          },
          { term: "Parameters", value: n(ctx.parameterCount()) },
          { term: "Logits", value: shape(...logits.shape) },
          { term: "Reference", value: "reference/model.ts", note: "same upstream parameter names" },
        ]}
      />
      <Check question="If changing heads changes the reconstructed keys, must the serving cache keep a separate latent for each head?">
        No. The same latent feeds a learned expansion with different output rows for each head. Head-specific keys are
        reconstructed from shared compressed state; only their rotary slice is shared directly.
      </Check>
      <Check question="Can an expert with a large original sigmoid score still be excluded?">
        Yes. Corrected group scores decide which groups survive before expert selection. An expert in an excluded group
        cannot receive this token, regardless of its original score; original scores are used only to weight the
        selected ids.
      </Check>
      <Takeaways
        items={[
          "The cached latent is a representation from which head-specific content keys and values can be rebuilt, not a shortened full key.",
          "Routing depends on both the token stream and correction bias; bias controls selection without entering mixture weights.",
          "Random toy runs prove how the arithmetic behaves, not the quality or speed of the trained checkpoint.",
        ]}
      />
    </Section>
  );
}
