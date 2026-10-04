import { Prose, Section, LayerMap, Sentence, Table, Takeaways, Check, useUrlNumber, n, shape } from "../../../kit";
import * as f from "../facts";
import { Graph, type Step } from "./graph";
export function Stack() {
  const [i, setLayer] = useUrlNumber("layer", 0, 0, f.layers.value - 1);
  const write = i % f.blockSize.value === 0;
  const steps: Step[] = [
    {
      id: "mix-a",
      label: i === 0 ? "input prefix" : "depth mixture",
      width: f.hidden.value,
      what:
        i === 0
          ? "The initial empty bank bypasses the attention-side mixture."
          : "Select a weighted mixture of raw depth candidates for this token.",
      formula: i === 0 ? "x = embedding" : "x = Σ softmax(score(candidate)) × candidate",
      lane: "skip",
    },
    {
      id: "snapshot",
      label: write ? "snapshot + reset" : "retain prefix",
      width: f.hidden.value,
      what: write
        ? "Store the old prefix in the completed bank and reset the additive prefix before attention."
        : "Keep the current additive prefix; no new block is written.",
      formula: write ? "bank ← bank ∪ {prefix}; prefix ← ∅" : "prefix unchanged",
      lane: "skip",
    },
    {
      id: "norm-a",
      label: "RMSNorm",
      width: f.hidden.value,
      what: "Normalize the mixture before attention.",
      formula: "x / sqrt(mean(x²) + ε) × gain",
      params: [["input_layernorm.weight", shape(f.hidden.value)]],
    },
    {
      id: "attention",
      label: f.isKda(i) ? "KDA" : "full attention",
      width: f.hidden.value,
      what: "Read token history using the attention kind of this absolute layer.",
      formula: f.isKda(i) ? "o = KDA(x)" : "o = gated causal MLA(x)",
    },
    {
      id: "extend-a",
      label: write ? "start prefix" : "extend prefix",
      width: f.hidden.value,
      what: "Attention starts a fresh prefix at a block boundary or extends the current prefix.",
      formula: write ? "prefix = o" : "prefix += o",
      lane: "skip",
    },
    {
      id: "mix-f",
      params: [["post_attention_layernorm.weight", shape(f.hidden.value)]],
      label: "depth mix + norm",
      width: f.hidden.value,
      what: "Mix the updated prefix with the completed bank, then normalize for the feed-forward branch.",
      formula: "x = RMSNorm(depth_mix(bank,prefix))",
      lane: "skip",
    },
    {
      id: "ffn",
      label: f.isMoe(i) ? "latent MoE" : "dense SiTU MLP",
      width: f.hidden.value,
      what: "Apply tokenwise feed-forward arithmetic to the mixture.",
      formula: f.isMoe(i)
        ? "y = expand(norm(Σ w_e expert_e(compress(x)))) + shared(x)"
        : "y = down(SiTU(gate(x),up(x)))",
      lane: "low",
    },
    {
      id: "extend-f",
      label: "extend prefix",
      width: f.hidden.value,
      what: "Return the additive prefix and completed bank to the next layer.",
      formula: "prefix += y",
      lane: "skip",
    },
  ];
  return (
    <Section
      id="stack"
      title="The stack and one block"
      source={[
        { source: "reference", symbol: "KimiDecoderLayer._forward_attn_residual" },
        { source: "configuration", symbol: "KimiLinearConfig.is_kda_layer" },
      ]}
      lede="Select an absolute layer. The attention kind, feed-forward kind and depth-bank boundary all follow the checkpoint's explicit rules."
    >
      <Prose>
        The bar beneath each operation compares its output channel width with the widest operation in that graph. The
        inspector gives exact tensor axes; a state matrix is not a stream vector.
      </Prose>
      <LayerMap
        label="Decoder layers (publisher zero-based indices)"
        count={f.layers.value}
        kinds={[
          { id: "kda-dense", label: "KDA · dense", lane: "merged" },
          { id: "kda-moe", label: "KDA · MoE", lane: "high" },
          { id: "mla-moe", label: "Full · MoE", lane: "low" },
        ]}
        kindOf={f.kindOf}
        selected={i}
        onSelect={setLayer}
      />
      <Sentence>
        Publisher index {n(i)} (absolute layer {n(i + 1)}) uses {f.isKda(i) ? "KDA" : "full attention"} and{" "}
        {f.isMoe(i) ? "latent MoE" : "dense FFN"}. It enters with {n(f.candidateCount(i))} completed depth blocks;{" "}
        {write ? "it snapshots and resets the prefix." : "it extends the current prefix."}
      </Sentence>
      <Graph
        id="block"
        title={`Layer ${n(i)} anatomy`}
        steps={steps}
        source="KimiDecoderLayer._forward_attn_residual"
        serving="KimiK3DecoderLayer._forward_attn_residual"
      />
      <Table
        caption="Layer-dependent rules"
        head={["Rule", "Exact checkpoint behavior"]}
        rows={[
          [
            "Attention",
            `Membership of (index + 1) in kda_layers; ${n(f.kdaCount.value)} recurrent, ${n(f.fullCount.value)} full.`,
          ],
          ["Feed-forward", `index ≥ ${n(f.denseLayers.value)} and index % ${n(f.moeFreq.value)} = 0 selects MoE.`],
          ["Depth boundary", `index % ${n(f.blockSize.value)} = 0 writes a completed block before attention.`],
        ]}
      />
      <Check question="At a depth boundary, is the old prefix still added to the new attention output?">
        No. It has been snapshotted into the bank; the attention output starts a new prefix. The old block can still
        influence the weighted sublayer input.
      </Check>
      <Takeaways
        items={[
          "Layer lists are one-based; the publisher loop is zero-based.",
          "The final adjacent full-attention layers are real, not an alternating-template artifact.",
          "Depth mixture and additive prefix are different tensors.",
        ]}
      />
    </Section>
  );
}
