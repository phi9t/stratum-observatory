import { Section, Prose, Takeaways, Check, useUrlNumber, n, shape } from "../../../kit";
import * as f from "../facts";
import { Graph, type Step } from "./graph";
export function Residual() {
  const [i] = useUrlNumber("layer", 0, 0, f.layers.value - 1);
  const candidates = f.candidateCount(i) + 1;
  const steps: Step[] = [
    {
      id: "candidates",
      label: "bank + live prefix",
      width: f.hidden.value,
      kind: "state",
      lane: "skip",
      what: "Stack completed raw blocks with the current additive prefix at each token.",
      formula: "V = concat_depth(bank, prefix)",
      output: shape("B×S", candidates, f.hidden.value),
    },
    {
      id: "rms",
      label: "candidate RMS",
      width: f.hidden.value,
      lane: "skip",
      what: "Normalize each candidate for scoring only.",
      formula: `K = V / sqrt(mean_channels(V²) + ${n(f.eps.value)})`,
      output: shape("B×S", candidates, f.hidden.value),
    },
    {
      id: "score-weight",
      col: 1,
      row: 1,
      inputs: [],
      label: "gain × projection",
      width: f.hidden.value,
      lane: "skip",
      what: "Combine learned RMSNorm gain with the scalar score projection.",
      formula: "w_score = norm.weight × proj.weight",
      params: [
        ["norm.weight", shape(f.hidden.value)],
        ["proj.weight", shape(1, f.hidden.value)],
      ],
      output: shape(f.hidden.value),
    },
    {
      id: "scores",
      col: 2,
      inputs: [
        ["K", shape("B×S", candidates, f.hidden.value)],
        ["w_score", shape(f.hidden.value)],
      ],
      label: "depth scores",
      width: candidates,
      lane: "skip",
      what: "Score every candidate independently at each token.",
      formula: "scores = sum_channels(K × w_score)",
      input: shape("B×S", candidates, f.hidden.value),
      output: shape("B×S", candidates),
    },
    {
      id: "softmax",
      col: 3,
      label: "softmax over depth",
      width: candidates,
      lane: "skip",
      what: "Normalize scores over candidates, never across token positions.",
      formula: "P = softmax_depth(scores)",
      output: shape("B×S", candidates),
    },
    {
      id: "mix",
      col: 4,
      inputs: [
        ["P", shape("B×S", candidates)],
        ["V", shape("B×S", candidates, f.hidden.value)],
      ],
      label: "raw-value mixture",
      width: f.hidden.value,
      lane: "skip",
      what: "Use the depth probabilities to combine unnormalized candidate values.",
      formula: "mixed = P V",
      output: shape("B", "S", f.hidden.value),
      note: "V is raw. Scoring normalization does not replace the candidate values.",
    },
  ];
  return (
    <Section
      id="residual"
      title="Attention across depth"
      source={[
        { source: "reference", symbol: "_apply_attn_res" },
        { source: "serving", symbol: "KimiK3DecoderLayer._forward_attn_residual" },
      ]}
      lede="Every token chooses a mixture of completed depth blocks and its live prefix. This softmax runs across depth candidates; token positions never read each other here."
    >
      <Prose>
        The selected layer's attention-side input has {n(candidates)} possible{" "}
        {candidates === 1 ? "candidate" : "candidates"} before a boundary write. At the initial layer the empty-bank
        mixture is bypassed. Its feed-forward side includes the newly snapshotted embedding block; final output
        aggregation uses the complete bank.
      </Prose>
      <Graph
        id="depth-residual"
        title="Depth candidate mixture"
        steps={steps}
        source="_apply_attn_res"
        serving="KimiK3DecoderLayer._forward_attn_residual"
        edges={[
          { from: "candidates", to: "rms" },
          { from: "rms", to: "scores" },
          { from: "score-weight", to: "scores" },
          { from: "scores", to: "softmax" },
          { from: "softmax", to: "mix" },
          { from: "candidates", to: "mix", kind: "skip" },
        ]}
      />
      <Check question="Would normalizing the values before their weighted sum preserve this forward pass?">
        No. Normalization is used to compute scores; the mixture then reads the raw bank and prefix. Normalizing those
        values would change the amplitude of the residual contributions.
      </Check>
      <Takeaways
        items={[
          `Depth boundaries occur every ${n(f.blockSize.value)} decoder layers, and the complete bank holds ${n(f.bankBlocks.value)} snapshots.`,
          `Depth memory is local to one forward's token rows; recurrent and KV caches persist across decode calls.`,
          `SGLang precomputes gain × projection and can fuse mixture, norm and snapshot; the conceptual depth axis stays unchanged.`,
        ]}
      />
    </Section>
  );
}
