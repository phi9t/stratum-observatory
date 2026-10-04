import { Boundary, Section, Table, sourceUrl, useSources } from "../../../kit";

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
          <a href={sourceUrl(source)} key="f" rel="noreferrer" target="_blank">{source.path}</a>,
        ])}
      />
      <Boundary>
        This page covers the language-model forward pass of one checkpoint. It does not cover tokenisation, sampling, training losses, the
        classification and multiple-choice heads that the reference file also defines, or the cross-attention branch this checkpoint does not enable.
      </Boundary>
    </Section>
  );
}
