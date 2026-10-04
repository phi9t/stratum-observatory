import { Boundary, Glossary, Section, sourceUrl, Sub, Table, useSources } from "../../../kit";

export function Sources() {
  const { file } = useSources();
  return (
    <Section
      id="sources"
      lede="Everything above is read from these files at these commits."
      source={[{ source: "reference", symbol: "GPT2LMHeadModel" }]}
      title="Sources and boundaries"
    >
      <Table
        caption={`Pinned sources for ${file.checkpoint.repo} @ ${file.checkpoint.revision.slice(0, 12)}`}
        head={["Role", "Repository", "Commit", "File"]}
        rows={file.sources.map((source) => [
          source.role,
          source.repo,
          <code key="c">{source.revision.slice(0, 12)}</code>,
          <a href={sourceUrl(source)} key="f" rel="noreferrer" target="_blank">
            {source.path}
          </a>,
        ])}
      />
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
          ["Weight tying", "Using one matrix for two jobs; here the token embedding is also the output projection."],
        ]}
      />
      <Boundary>
        This page covers the language-model forward pass of one checkpoint. It does not cover tokenisation, sampling,
        training losses, the classification and multiple-choice heads that the reference file also defines, or the
        cross-attention branch this checkpoint does not enable.
      </Boundary>
    </Section>
  );
}
