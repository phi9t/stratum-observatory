import { Takeaways, Boundary, Glossary, Section, sourceUrl, Sub, Table, useSources } from "../../../kit";

export function Sources() {
  const { file } = useSources();
  return (
    <Section
      id="sources"
      lede="The page is drawn from these files at these commits. Helpers they call that are not pinned here (the serving layer\u2019s fused norm and activation, the cache container) are described from their call sites only."
      source={[{ source: "reference", symbol: "LlamaForCausalLM" }]}
      title="Sources and boundaries"
    >
      <Table
        caption={`Pinned sources for ${file.checkpoint.repo}`}
        head={["Role", "Repository", "Commit", "File"]}
        rows={file.sources.map((source) => [
          source.role,
          source.repo,
          <code key="c">{source.revision}</code>,
          <a href={sourceUrl(source)} key="f" rel="noreferrer" target="_blank">
            {source.path}
          </a>,
        ])}
      />
      <Sub>Checkpoint identity</Sub>
      <Boundary>
        {file.checkpoint.repo} @ <code>{file.checkpoint.revision}</code>. The reference is the pinned Transformers Llama
        implementation; this is the base checkpoint, not an Instruct variant.
      </Boundary>
      <Sub>Terms used on this page</Sub>
      <Glossary
        terms={[
          ["Residual stream", "The one tensor that runs through the whole model. Blocks read from it and add to it."],
          [
            "Pre-norm",
            "Normalising the input of each sublayer instead of its output, leaving the stream itself unnormalised until the end.",
          ],
          [
            "Head",
            "One of several independent attention computations that share a layer, each working on its own slice of the width.",
          ],
          ["Causal mask", "The rule that a position may read itself and earlier positions only."],
          [
            "KV cache",
            "Keys and values of positions already processed, kept so that generating the next token does not recompute them.",
          ],
          ["Grouped-query attention", "Several query heads read the same key/value head."],
          ["RMSNorm", "Scale a vector by its root mean square and a learned gain without centering it."],
          [
            "Scaled dot product",
            "Sum coordinate products of a query and key, divided by the square root of head width.",
          ],
          [
            "Seeded weight gain",
            "The multiplier on synthetic initialization spread, distinct from the learned coordinate gains in RMSNorm.",
          ],
          ["Tied weights", "The input embedding matrix is reused to compute output vocabulary scores."],
          ["RoPE", "Rotate query and key coordinate pairs by position-dependent angles."],
          ["SiLU gate", "A smooth activation multiplied by an independent content projection; no token routing."],
          ["Prefill", "Process the prompt positions together before generating new tokens."],
          ["Tensor parallelism", "Partition a module across ranks; KV heads may also be replicated."],
        ]}
      />
      <Boundary>
        This page covers the language-model forward pass of one checkpoint. It does not cover tokenisation, sampling,
        training losses, classification or question-answering heads, quantized kernels, padding experiments or GPU
        performance. The clean reference executes unpadded prefill, not cached decoding.
      </Boundary>
      <Takeaways
        items={[
          "Sources are pinned; no checkpoint weights are bundled.",
          "Random toy weights demonstrate arithmetic, not trained model behavior.",
        ]}
      />
    </Section>
  );
}
