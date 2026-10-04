// usage: npm run index   — builds every model page, then dist/index.html linking them.
// The index is static HTML styled by the kit's stylesheet; it has no script.
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const slugs = readdirSync(join(root, "models"))
  .filter((slug) => existsSync(join(root, "models", slug, "sources.json")))
  .sort();

const escape = (text) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const rows = [];
for (const slug of slugs) {
  if (!process.argv.includes("--no-build")) {
    const built = spawnSync("node", ["tools/build.mjs", slug], { cwd: root, stdio: "inherit" });
    if (built.status !== 0) {
      console.error(`index: build failed for ${slug}`);
      process.exit(1);
    }
  }
  const sources = JSON.parse(readFileSync(join(root, "models", slug, "sources.json"), "utf8"));
  const page = readFileSync(join(root, "models", slug, "Page.tsx"), "utf8");
  const name = /name=(?:"([^"]+)"|\{"([^"]+)"\})/.exec(page)?.slice(1).find(Boolean) ?? slug;
  const tagline = /tagline=(?:"([^"]+)"|\{"([^"]+)"\})/s.exec(page)?.slice(1).find(Boolean) ?? "";
  const reference = existsSync(join(root, "models", slug, "reference", "model.ts"));
  const fixture = existsSync(join(root, "models", slug, "reference", "fixture.json"));
  const component = existsSync(join(root, "models", slug, "reference", "component-fixture.json"));
  rows.push({ slug, name, tagline, repo: sources.checkpoint.repo, revision: sources.checkpoint.revision, reference, fixture, component });
}

const css = readFileSync(join(root, "kit", "styles", "schematic.css"), "utf8");
const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="color-scheme" content="light dark" />
<title>Model schematics</title>
<style>${css}
.ms-index { display: grid; gap: 0; margin-block: 1.5rem 0; list-style: none; }
.ms-index li { display: grid; grid-template-columns: 16rem minmax(0, 1fr) auto; gap: 0.25rem 2rem; padding: 1.1rem 0; border-top: 1px solid var(--ms-rule); align-items: baseline; }
.ms .ms-index a { color: var(--ms-ink); font-family: var(--ms-serif); font-size: 1.5rem; text-decoration: none; }
.ms .ms-index a:hover { text-decoration: underline; }
.ms-index p { color: var(--ms-ink-2); max-width: 44rem; }
.ms-index code { color: var(--ms-ink-3); font-size: 0.75rem; }
@media (max-width: 760px) { .ms-index li { grid-template-columns: minmax(0, 1fr); } }
</style>
</head>
<body class="ms-body">
<div class="ms" data-standalone="true">
<header class="ms-hero">
<div class="ms-hero__eyebrow">Model schematics</div>
<h1 class="ms-hero__title">Circuits, drawn from code</h1>
<div class="ms-hero__tagline">One page per checkpoint: the whole model, any layer, every operation, with shapes and formulas traced to pinned source and a clean reference you can run in the browser.</div>
</header>
<section class="ms-section" aria-label="Models">
<ol class="ms-index">
${rows
  .map(
    (row) => `<li>
<a href="./${escape(row.slug)}.html">${escape(row.name)}</a>
<p>${escape(row.tagline)}</p>
<code>${escape(row.repo)} @ ${escape(row.revision.slice(0, 12))}<br />${
      row.fixture ? "clean reference · upstream parity fixture" : row.component ? "clean reference · component parity only" : row.reference ? "clean reference · shape-checked" : "no clean reference yet"
    }</code>
</li>`,
  )
  .join("\n")}
</ol>
</section>
</div>
</body>
</html>
`;
mkdirSync(join(root, "dist"), { recursive: true });
writeFileSync(join(root, "dist", "index.html"), html);
console.log(`index: ${rows.length} models -> dist/index.html`);
