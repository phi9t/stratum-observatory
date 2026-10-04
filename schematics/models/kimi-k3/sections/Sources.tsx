import { Section, Prose, Table, Boundary, Glossary, Takeaways, sourceUrl, useSources } from "../../../kit";
export function Sources() {
  const { file } = useSources();
  return (
    <Section
      id="sources"
      title="Sources and boundaries"
      source={[{ source: "reference", symbol: "KimiLinearForCausalLM" }]}
      lede="Every implementation and config is pinned. This page follows the publisher's text arithmetic and records serving differences rather than assuming framework equivalence."
    >
      <Prose>
        Checkpoint: {file.checkpoint.repo} at{" "}
        <code style={{ overflowWrap: "anywhere" }}>{file.checkpoint.revision}</code>.
      </Prose>
      <Table
        caption={`${file.checkpoint.repo} @ ${file.checkpoint.revision}`}
        head={["Role", "Repository", "Commit", "File"]}
        rows={file.sources.map((s) => [
          s.role,
          s.repo,
          <code key="sha">{s.revision}</code>,
          <a key="path" href={sourceUrl(s)} target="_blank" rel="noreferrer">
            {s.path}
          </a>,
        ])}
      />
      <Boundary>
        The clean reference covers unpadded text prefill to logits with logical weights. The exact checkpoint's vision
        interface is drawn and vision parameters are counted, but vision internals, media preprocessing, tokenization,
        sampling, training loss, beam handling, distributed execution and packed MXFP4 numerical arithmetic are
        excluded. No compatible training source was found in the checkpoint; publisher MoE explicitly rejects training
        mode. Whole-model numerical parity is blocked by CPU-only PyTorch, missing FLA and missing FlashAttention.
        Component parity has narrower scope.
      </Boundary>
      <Glossary
        terms={[
          ["Residual prefix", "The raw additive sum within the current depth block."],
          ["Depth bank", "Completed additive blocks used as candidates for a same-token weighted depth mixture."],
          ["RMSNorm", "Channelwise normalization by root mean square, with a learned gain."],
          ["KDA", "Kimi Delta Attention: causal convolution followed by a decayed matrix-state delta update."],
          ["MLA", "Multi-head latent attention: narrow query/KV paths expanded or absorbed into head-local attention."],
          ["NoPE", "No positional embedding transform in the text full-attention branch."],
          ["SiTU", "Saturating tanh and sigmoid gate multiplied by a separately saturated value projection."],
          ["MoE", "Mixture of experts: tokenwise selection of routed feed-forward networks."],
          [
            "Latent MoE",
            "Experts operating in a compressed representation, with aggregation normalized before expansion.",
          ],
          ["Shared expert", "An always-on full-stream feed-forward branch, separate from router selection."],
          ["Correction bias", "Learned expert-selection adjustment that is excluded from mixture weights."],
          ["KV cache", "Persistent token-attention keys/values or their absorbed latent representation."],
          ["Absorption", "Moving KV expansion weights into query/output algebra so expanded KV need not be stored."],
          [
            "MXFP4",
            "Packed floating-point expert-weight storage; logical parameter count is separate from its byte size.",
          ],
          ["2D rotary positions", "Rotations applied to vision query/key channels from spatial coordinates."],
          ["TP / EP", "Tensor parallelism shards operations; expert parallelism distributes selected experts."],
        ]}
      />
      <Takeaways
        items={[
          "One pinned checkpoint defines this page; toy weights demonstrate mechanics only.",
          "Missing whole-model numerical evidence remains unverified.",
          "Sources link to upstream; no upstream implementation text is bundled into the page.",
        ]}
      />
    </Section>
  );
}
