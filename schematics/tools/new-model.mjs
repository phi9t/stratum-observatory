#!/usr/bin/env node
// usage: node tools/new-model.mjs <slug> <hf-repo>
//
// Scaffolds models/<slug>/ per the model folder contract in README.md.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import { MODELS_ROOT, fail, parseArgs, requireSlug } from "./lib/common.mjs";

const USAGE = "npm run new -- <slug> <hf-repo>";
const { positional } = parseArgs(process.argv.slice(2));
const slug = requireSlug(positional[0], USAGE);
const repo = positional[1];
if (!repo || positional.length !== 2) {
  fail(`usage: ${USAGE}`);
}
if (slug === "gallery") {
  fail('"gallery" is reserved for kit/gallery');
}
if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
  fail(`hf-repo "${repo}" must look like org/Name`);
}
const root = join(MODELS_ROOT, slug);
if (existsSync(root)) {
  fail(`${root} already exists; refusing to overwrite it`);
}

const name = repo.split("/")[1];
const json = (value) => `${JSON.stringify(value, null, 2)}\n`;
const text = (value) => JSON.stringify(value);

const files = {
  "index.html": `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="light dark" />
    <title>${name.replace(/[&<>]/g, "")} · model schematic</title>
  </head>
  <body class="ms-body">
    <div id="root"></div>
    <script type="module" src="./main.tsx"></script>
  </body>
</html>
`,

  "main.tsx": `import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "../../kit/styles/schematic.css";
import { SourcesProvider, type SourcesFile, type SymbolIndex } from "../../kit/lib/sources";
import { Page } from "./Page";
import sources from "./sources.json";
import symbols from "./symbols.json";

const root = document.getElementById("root");
if (root === null) {
  throw new Error("index.html has no #root");
}
createRoot(root).render(
  <StrictMode>
    <SourcesProvider sources={sources as SourcesFile} symbols={symbols as unknown as SymbolIndex}>
      <Page />
    </SourcesProvider>
  </StrictMode>,
);
`,

  "Page.tsx": `// The page: <Schematic> composed of one <Section> per file in ./sections.
// Pattern to follow: models/gpt2/Page.tsx.
import { Schematic, type SectionSpec } from "../../kit";

const sections: readonly SectionSpec[] = [
  // { id: "overview", label: "Overview" },
];

/** Section ids, in page order. coverage.json maps reference classes onto these. */
export const sectionIds: string[] = sections.map((section) => section.id);

export function Page() {
  return (
    <Schematic
      authority={<>Checkpoint ${repo}</>}
      eyebrow="Model schematic"
      name=${text(name)}
      sections={sections}
      spec={[]}
      tagline=""
    >
      {null}
    </Schematic>
  );
}
`,

  "facts.ts": `// Every number the page shows. Section files take numbers only from here.
//
//   cfg(path, value)        dotted path into the pinned config.json, value as written
//   code(value, locator)    read from source: { source: "<source id>", symbol: "Class.method", contains: "literal" }
//   derive(fn, deps)        computed from other facts; never typed in
import { cfg, derive, type Fact } from "../../kit/lib/facts";

// Placeholder: replace the value with what config.json holds, then add the rest.
export const modelType = cfg("model_type", "");

/** Every Fact, for the tests and the facts gate. */
export const facts: Record<string, Fact> = { modelType };

/** Parameter count derived from the facts above; checked against the hub's safetensors.total. */
export const parameterTotal = derive(() => 0, [modelType]);

// If parameterTotal legitimately differs from safetensors.total, say why:
// export const parameterNote = "…";
`,

  "sources.json": json({
    checkpoint: { repo, revision: "main" },
    sources: [
      { id: "config", kind: "hf", repo, revision: "main", path: "config.json", role: "config" },
    ],
  }),

  "symbols.json": json({}),

  "coverage.json": json({
    source: "reference",
    classes: {},
    serving: { source: "serving", notes: "" },
  }),

  "sections/.gitkeep": "",

  "tests/facts.test.ts": `import { describe, expect, it } from "vitest";

import {
  verifyCoverage,
  verifyFacts,
  verifyParameters,
  verifyPins,
} from "../../../kit/lib/verify";
import * as model from "../facts";
import { sectionIds } from "../Page";

const SLUG = ${text(slug)};

describe(\`\${SLUG} against its pinned sources\`, () => {
  it("pins every source to a commit and has it cached", () => {
    expect(verifyPins(SLUG)).toEqual([]);
  });

  it("states only facts the pinned config and code hold", () => {
    expect(verifyFacts(SLUG, model.facts)).toEqual([]);
  });

  it("maps every reference class to a section or excludes it with a reason", () => {
    expect(verifyCoverage(SLUG, sectionIds)).toEqual([]);
  });

  it("derives the parameter total the hub reports", () => {
    const note = (model as { parameterNote?: string }).parameterNote;
    const result = verifyParameters(SLUG, model.parameterTotal, note);
    expect(result.status, result.detail).not.toBe("fail");
  });
});
`,
};

for (const [relative, content] of Object.entries(files)) {
  const file = join(root, relative);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, content);
}

console.log(`created ${root}`);
for (const relative of Object.keys(files)) {
  console.log(`  ${relative}`);
}
console.log(`
next:
  1. add the reference and serving sources to models/${slug}/sources.json
  2. npm run fetch -- ${slug}
  3. npm run dev -- ${slug}
  4. npm run check -- ${slug}`);
