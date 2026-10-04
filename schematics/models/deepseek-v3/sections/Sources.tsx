import { Section, Table, Sub, Glossary, Boundary, Takeaways, sourceUrl, useSources } from "../../../kit";
export function Sources() {
  const { file } = useSources();
  return (
    <Section
      id="sources"
      title="Sources and boundaries"
      source={[{ source: "reference", symbol: "DeepseekV3ForCausalLM" }]}
      lede="This is one checkpoint's main inference architecture, read from the code shipped with it and compared with independently maintained serving and training implementations."
    >
      <Table
        caption={`${file.checkpoint.repo} @ ${file.checkpoint.revision}`}
        head={["Role", "Repository", "Commit", "File"]}
        rows={file.sources.map((s) => [
          s.role,
          s.repo,
          <code>{s.revision}</code>,
          <a href={sourceUrl(s)} target="_blank" rel="noreferrer">
            {s.path}
          </a>,
        ])}
      />
      <Sub>Terms used on this page</Sub>
      <Glossary
        terms={[
          ["Residual stream", "The token vectors passed through the stack, with branch outputs added to them."],
          [
            "Pre-norm",
            "Normalizing a branch input before its computation rather than normalizing the residual addition.",
          ],
          ["RMSNorm", "A learned rescaling based on mean square, without subtracting the mean."],
          ["Head", "An independent query/key comparison and value average within a layer."],
          ["Latent / low rank", "An intermediate of reduced width that is expanded by a learned linear map."],
          ["MLA", "Multi-head latent attention: query and KV projection paths use compressed intermediates."],
          ["Causal mask", "A constraint that each query reads itself and earlier valid tokens only."],
          [
            "KV cache",
            "Persistent state for historical tokens, in expanded or compressed form depending on implementation.",
          ],
          ["RoPE", "Rotary positional embedding: rotate query/key channel pairs by position-dependent phases."],
          ["YaRN", "Frequency blending and scale adjustments configured for this checkpoint's rotary scheme."],
          ["MoE", "A mixture of experts: token-selected feed-forward transforms combined with route weights."],
          ["Correction bias", "An expert preference added for selection, excluded from mixture weights."],
          ["Shared expert", "An always-on feed-forward path added separately from the routed mixture."],
          [
            "SiLU / SwiGLU",
            "SiLU activates a gate projection; its product with a separate up projection is projected down.",
          ],
          ["Absorption", "Reordering linear maps around attention so history can remain compressed."],
          [
            "MTP",
            "Multi-token prediction: auxiliary modules stored with the checkpoint but absent from the reference's main forward.",
          ],
          ["FP8 / BF16", "Storage and arithmetic formats; counting scalars does not determine their byte size."],
          ["Logits", "Unnormalized vocabulary scores before sampling or probability normalization."],
        ]}
      />
      <Boundary>
        The drawings follow eager evaluation of the publisher reference. FlashAttention and compressed decode are
        identified as implementation alternatives. The auxiliary MTP tensors are reconciled in storage accounting but
        their training/speculative execution is outside the main circuit. Tokenizer internals, sampling, training
        losses, distributed collectives and quantization kernels are not drawn. No weights, learned router biases or
        model predictions were loaded; synthetic routing fixtures do not establish checkpoint parity. Configured context
        is not tested capability. No GPU memory or speed benchmark is claimed.
      </Boundary>
      <Takeaways
        items={[
          "Pinned links identify the exact definitions behind the drawings; later versions may differ.",
          "Source inspection and arithmetic checks establish this page's structural claims, not measured model quality or runtime performance.",
        ]}
      />
    </Section>
  );
}
