import { Section, Table, Boundary, Glossary, Sub, Prose, Takeaways, sourceUrl, useSources } from "../../../kit";
export function Sources() {
  const { file } = useSources();
  return (
    <Section
      id="sources"
      title="Sources and boundaries"
      source={[{ source: "reference", symbol: "DeepseekV3ForCausalLM" }]}
      lede="This page describes one checkpoint at one pinned revision. Source links open exact implementation lines; experiments use random or perturbed toy weights."
    >
      <Table
        caption={`Exact checkpoint: ${file.checkpoint.repo} @ ${file.checkpoint.revision}`}
        head={["Role", "Repository", "Commit", "File"]}
        rows={file.sources.map((s) => [
          s.role,
          s.repo,
          <code key="c">{s.revision}</code>,
          <a key="f" href={sourceUrl(s)} target="_blank" rel="noreferrer">
            {s.path}
          </a>,
        ])}
      />
      <Boundary>
        Scope: the eager evaluation decoder from token ids to logits, with decode cache representations described
        separately. Excluded: tokenization and chat templates, sampling, training loss, distributed collectives,
        speculative decoding and GPU performance. The checkpoint is FP8; the clean reference uses float32 arithmetic and
        does not emulate quantized execution. Numerical parity is verified at toy widths against unchanged publisher
        module definitions, not checkpoint weights. No weights were downloaded.
      </Boundary>
      <Prose>
        No exact public Kimi-K2 training forward implementation was established during source selection. The publisher
        inference gate explicitly requires evaluation mode; a generic training recipe would not be evidence for this
        checkpoint.
      </Prose>
      <Sub>Terms in order of use</Sub>
      <Glossary
        terms={[
          ["Decoder", "A stack that scores the next token using only visible preceding and current positions."],
          ["Residual stream", "The hidden tensor carried between blocks; each sublayer adds an update to it."],
          ["Pre-norm", "Normalization before a sublayer reads the stream."],
          ["RMSNorm", "Scaling by inverse root mean square and a learned gain, without subtracting a mean."],
          ["Latent", "A learned lower-dimensional content representation used to reconstruct keys and values."],
          ["Attention head", "A separate query/key comparison and value mixture within a layer."],
          ["Rotary position", "Rotation of a query/key slice by an angle determined by its absolute position."],
          ["YaRN", "A frequency interpolation scheme that also adjusts the attention score scale."],
          ["Causal mask", "An additive mask that forbids reading later sequence positions."],
          ["Cache", "State from preceding decode steps retained instead of recomputed."],
          ["Mixture of experts", "Selected token-specific feed-forward transforms combined with router weights."],
          ["Correction bias", "Learned score offset used for expert selection only."],
          ["Shared expert", "Always-on feed-forward path separate from the routed bank."],
          ["SiLU", "The gate activation x multiplied by sigmoid(x)."],
          ["Logits", "Unnormalized scores over vocabulary entries."],
          ["BF16", "The cache arithmetic's two-byte floating-point scalar format."],
          ["FP8", "The checkpoint's eight-bit weight format, with separate block scales."],
          ["Parity fixture", "A saved upstream toy execution used to test the clean arithmetic."],
          [
            "Absorbed query",
            "Content query transformed into latent coordinates so attention need not expand cached keys.",
          ],
          [
            "Fused operation",
            "Multiple arithmetic steps implemented by a combined kernel or projection without an extra logical branch.",
          ],
        ]}
      />
      <Takeaways
        items={[
          "Pinned implementation evidence establishes the circuit, not learned behavior or measured performance.",
          "The original Instruct checkpoint has no sparse-attention indexer and no multi-token-prediction (next-n) layers in this decoder.",
        ]}
      />
    </Section>
  );
}
