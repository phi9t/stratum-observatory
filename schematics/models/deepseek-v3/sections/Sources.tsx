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
          ["Top-k", "Select the highest-scoring expert ids after restricting candidates to the surviving groups."],
          ["Dense FFN", "A feed-forward transform used for every token rather than chosen from a routed bank."],
          [
            "Renormalization",
            "Divide the chosen original sigmoid scores by their sum before applying the route scale.",
          ],
          [
            "FP8 block scale",
            "A stored scale for a block of quantized matrix entries, counted separately from main model parameters.",
          ],
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
          [
            "noaux_tc",
            "The configured inference router: corrected grouped selection followed by unbiased mixture weights.",
          ],
          [
            "Parity fixture",
            "Toy inputs, perturbed upstream parameters and saved outputs used to test the clean reference independently.",
          ],
          [
            "Eager / FlashAttention",
            "Explicit attention tensor operations versus a fused attention kernel; their cache forms can differ.",
          ],
          [
            "Group selection",
            "A restriction to the groups with the strongest sums of corrected expert scores before individual expert selection.",
          ],
          ["FP16 / float32", "Floating-point arithmetic formats with different precision and representable ranges."],
          [
            "Collectives / sharding",
            "Cross-device communication and partitioning of tensors in distributed implementations.",
          ],
        ]}
      />
      <Boundary>
        The drawings follow eager evaluation of the publisher reference. FlashAttention and compressed decode are
        identified as implementation alternatives. The auxiliary MTP tensors are reconciled in storage accounting but
        their training/speculative execution is outside the main circuit. Tokenizer internals, sampling, training
        losses, distributed collectives and quantization kernels are not drawn. No weights, learned router biases or
        model predictions were loaded. The clean toy reference matches a pinned upstream CPU fixture; this does not
        establish trained checkpoint parity. Configured context is not tested capability. No GPU memory or speed
        benchmark is claimed.
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
