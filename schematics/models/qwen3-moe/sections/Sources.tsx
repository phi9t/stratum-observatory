import {
  Section,
  Table,
  Glossary,
  Boundary,
  Takeaways,
  SourceRef,
  Prose,
  Sub,
  sourceUrl,
  useSources,
} from "../../../kit";
export function Sources() {
  const { file } = useSources();
  return (
    <Section
      id="sources"
      title="Sources and boundaries"
      source={[{ source: "reference", symbol: "Qwen3MoeForCausalLM" }]}
      lede="One checkpoint, pinned implementation bytes, and an independently executed CPU parity fixture. Links open the exact source symbols used for the diagrams."
    >
      <Table
        caption={`Checkpoint ${file.checkpoint.repo} @ ${file.checkpoint.revision}`}
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
      <Sub>Training support has a specific scope</Sub>
      <Prose>
        The pinned Megatron bridge explicitly registers this architecture, maps separate Q/K/V and expert gate/up
        tensors, and configures RMSNorm, QK norms, grouped expert GEMM and an all-to-all dispatcher. It is a
        conversion/provider bridge; its imported training implementation and distributed kernels were not executed in
        this run. <SourceRef source="training" symbol="Qwen3MoEBridge.provider_bridge" />
      </Prose>
      <Sub>Terms used on this page</Sub>
      <Glossary
        terms={[
          ["Residual stream", "The vector at each position that layers read and update by addition."],
          ["Pre-norm", "Normalize the input to a sublayer while its residual bypass stays unnormalized."],
          ["MoE", "A bank of feed-forward experts with a router selecting a weighted subset for each token."],
          ["Router", "A learned projection whose expert scores determine per-token choices and mixture weights."],
          ["Logits", "Unnormalized scores, over the vocabulary at the output or over experts at the router."],
          ["GQA", "Grouped-query attention: several query heads share one key head and one value head."],
          ["Head", "An independent slice of attention channels with its own query and attention weights."],
          ["KV cache", "Retained keys and values, stored per layer so earlier tokens need not be projected again."],
          ["RoPE", "Rotary positional embedding: position-dependent rotations of query and key channel pairs."],
          ["RMSNorm", "Rescale using mean squared magnitude and a learned gain, without mean subtraction."],
          ["SiLU", "The elementwise function x times sigmoid(x), used to gate an expert's up projection."],
          ["Top-k", "Select the largest expert probabilities and retain their ids and values."],
          ["Renormalization", "Divide selected probabilities by their selected sum to make a unit-sum mixture."],
          [
            "Causal mask",
            "Restrict a query to its own position and earlier positions, also respecting padding upstream.",
          ],
          ["Prefill", "Process an input sequence together to build hidden states and optionally the cache."],
          ["Decode", "Process new tokens while reusing retained keys and values from earlier positions."],
          ["BF16", "A floating-point storage format; cache calculations assume this precision but do not measure it."],
          ["GiB", "A binary unit of bytes used for the explicitly stated logical cache payload."],
          ["Tensor parallelism", "Partition or replicate parameter and activation slices across serving ranks."],
          ["Pipeline parallelism", "Assign consecutive layer ranges to different serving ranks."],
          ["GEMM", "Matrix multiplication used for projections; grouping organizes expert computations."],
          ["All-to-all", "A distributed token dispatcher exchanging expert-bound tokens between ranks."],
          [
            "Weight tying",
            "Reusing a learned matrix for multiple roles; disabled for this checkpoint's embedding and head.",
          ],
        ]}
      />
      <Boundary>
        This page covers the exact text decoder architecture through logits. Tokenization, sampling, training losses and
        recipes, task-specific heads, quantization, distributed kernel internals and checkpoint-quality claims are
        excluded. The clean executable is unpadded prefill with float32 toy tensors; cache behavior is source-inspected
        and drawn, not numerically parity-tested. No trained weights were downloaded; GPU and BF16 numerical parity,
        training parity and performance are unverified.
      </Boundary>
      <Takeaways
        items={[
          "Pinned source inspection and toy numerical parity establish a scoped implementation correspondence, not trained-model quality or serving performance.",
          "Random experiments are labeled and parameter counts are derived; missing runtime evidence is not replaced by a benchmark claim.",
        ]}
      />
    </Section>
  );
}
