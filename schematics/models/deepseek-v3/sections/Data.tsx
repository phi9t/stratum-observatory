import { Figure, Section, TensorScale, Table, Takeaways, n, shape, Check } from "../../../kit";
import * as f from "../facts";
export function Data() {
  return (
    <Section
      id="data"
      title="Data representation"
      source={[
        { source: "reference", symbol: "DeepseekV3Attention" },
        { source: "publisher", symbol: "MLA" },
      ]}
      lede="Attention first compresses the stream, then expands it into per-head queries, keys and values. The compressed KV representation is also sufficient persistent state for an absorbed decode implementation."
    >
      <Figure
        label="KV representation widths"
        caption={`The eager reference caches ${n(f.eagerCacheWidth.value)} scalars per token per layer. Absorbed decode keeps ${n(f.kvAWidth.value)}: a ${n(f.cacheReduction.value)}× reduction before precision and allocator overhead.`}
      >
        <TensorScale
          title="Expanded cache versus compressed latent"
          pxPerToken={6}
          pxPerWidth={0.004}
          footnote="Equal token counts; height shows stored width."
          stages={[
            {
              label: "residual",
              tensors: [{ key: "x", name: "hidden", tokens: f.DEMO.tokens, width: f.hidden.value, lane: "merged" }],
            },
            {
              label: "projection output",
              tensors: [
                {
                  key: "latent",
                  from: "x",
                  name: "latent + rotary key",
                  tokens: f.DEMO.tokens,
                  width: f.kvAWidth.value,
                  lane: "instance",
                },
              ],
            },
            {
              label: "eager cache",
              tensors: [
                {
                  key: "expanded",
                  from: "latent",
                  name: "expanded K + V",
                  tokens: f.DEMO.tokens,
                  width: f.eagerCacheWidth.value,
                  lane: "high",
                },
              ],
            },
          ]}
        />
      </Figure>
      <Table
        caption="Axes and persistent state"
        head={["Tensor", "Shape", "Meaning"]}
        rows={[
          ["Token ids", shape("B", "S"), `Vocabulary indices; BOS ${n(f.bos.value)}, EOS ${n(f.eos.value)}.`],
          ["Positions", shape(1, "S"), "Absolute positions offset by the existing cache length."],
          ["Query", shape("B", f.heads.value, "S", f.qDim.value), "Per-head nonpositional and rotary channels."],
          [
            "Expanded K / V",
            `${shape("B", f.heads.value, "T", f.qDim.value)} / ${shape("B", f.heads.value, "T", f.valueDim.value)}`,
            "Eager cache returns all previous plus current positions.",
          ],
          [
            "Compressed state",
            shape("B", "T", f.kvAWidth.value),
            "Normalized KV latent plus rotated shared key; no persistent query.",
          ],
          ["Logits", shape("B", "S", f.vocab.value), "Unnormalized next-token scores, not sampled tokens."],
        ]}
      />
      <Check question="Can compressed decode discard the positional key and cache only the normalized latent?">
        No. The rotary key is made by separate projection channels and rotated at its own absolute position. It cannot
        be recovered from the normalized latent alone.
      </Check>
      <Takeaways
        items={[
          "B is batch, S is the current query sequence, and T includes the cached history.",
          "Compression describes how KV is represented; it does not remove causal attention to earlier positions.",
        ]}
      />
    </Section>
  );
}
