import { Section, Table, Glossary, Sub, Boundary, Takeaways, sourceUrl, useSources } from "../../../kit";
export function Sources() {
  const { file } = useSources();
  return (
    <Section
      id="sources"
      title="Sources and verification boundaries"
      source={[{ source: "reference", symbol: "Transformer" }]}
      lede="The page follows the publisher’s inference implementation for this exact checkpoint. Serving and training references carry explicit support limits."
    >
      <Table
        caption={`${file.checkpoint.repo} @ ${file.checkpoint.revision}`}
        head={["Role", "Repository", "Commit", "File"]}
        rows={file.sources.map((s) => [
          s.role,
          s.repo,
          <code key="commit">{s.revision}</code>,
          <a key="url" href={sourceUrl(s)} target="_blank" rel="noreferrer">
            {s.path}
          </a>,
        ])}
      />
      <Sub>Terms introduced on this page</Sub>
      <Glossary
        terms={[
          [
            "Residual copies",
            "Parallel channel vectors carried through the backbone, mixed at each sublayer boundary.",
          ],
          [
            "Collapse / expansion",
            "Weighted reduction to a sublayer input, then distribution of its output to residual copies.",
          ],
          ["Hyper-connections", "Input-dependent collapse, expansion and residual-matrix coefficients."],
          ["Sinkhorn", "Alternating row and column normalization of a positive matrix."],
          ["SwiGLU", "An activated gate multiplied by a separate up projection, then projected down."],
          ["MoE", "Selected routed experts plus an independent always-on shared expert."],
          ["KV source", "A layer that constructs compressed KV and index keys for subsequent consumers."],
          ["Index source", "A layer that recomputes the selected compressed positions."],
          ["Candidate block", "A coarse group admitted before later fine-grained index selection."],
          ["Ring cache", "A bounded window buffer whose write slot wraps with decoding position."],
          ["RoPE", "Position-dependent rotation of paired channels; this model rotates only the tail."],
          ["YaRN", "A frequency blend that preserves shorter wavelengths and stretches longer ones."],
          ["Attention sink", "A learned denominator contribution with no value vector."],
          ["Engram", "Hashed token n-gram retrieval gated into residual copies."],
          ["dSpark", "The checkpoint’s auxiliary block-draft forward path."],
          ["Logical parameter", "A scalar weight before quantized packing and scale metadata."],
          ["Fake quantization", "Rounding to a low-precision grid while retaining an activation tensor."],
          ["Parity", "Agreement with independently executed upstream outputs, not a test of a program against itself."],
        ]}
      />
      <Boundary>
        Executable scope: text-only backbone prefill to logits; Engram hash IDs are supplied explicitly. Excluded from
        that implementation: tokenizer hashes, vision, draft heads, distributed collectives, sampling, caches and
        quantized rounding. Vision and draft structure are drawn but unverified numerically. Logical parameter
        reconciliation includes the separately counted vision and draft paths. No benchmark or trained-model behavior is
        claimed.
      </Boundary>
      <Takeaways
        items={[
          "Source pins identify the implementation being described.",
          "The training reference is a family comparison with different arithmetic, not exact checkpoint support.",
          "A passed software gate cannot repair missing numerical parity or quantized execution evidence.",
        ]}
      />
    </Section>
  );
}
