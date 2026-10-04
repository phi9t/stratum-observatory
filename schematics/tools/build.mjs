#!/usr/bin/env node
// usage: node tools/build.mjs <slug> [--dev]
//
// Builds models/<slug>/ (or kit/gallery for `gallery`) into ONE self-contained
// file, dist/<slug>.html: CSS and JS inlined, assets as data URIs, nothing
// loaded from anywhere else. `--dev` starts the Vite dev server instead.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import react from "@vitejs/plugin-react";
import { build, createServer } from "vite";

import { DIST_ROOT, PACKAGE_ROOT, fail, pageRoot, parseArgs, requireSlug } from "./lib/common.mjs";

const USAGE = "npm run build -- <slug>   |   npm run dev -- <slug>";
const { positional, values } = parseArgs(process.argv.slice(2), { flags: ["dev"] });
const slug = requireSlug(positional[0], USAGE);
const root = pageRoot(slug);
if (!existsSync(join(root, "index.html"))) {
  fail(`${join(root, "index.html")} does not exist`);
}

const shared = {
  configFile: false,
  root,
  base: "./",
  cacheDir: join(PACKAGE_ROOT, "node_modules", ".vite"),
  plugins: [react()],
};

if (values.dev) {
  const server = await createServer({
    ...shared,
    server: { host: "127.0.0.1", fs: { allow: [PACKAGE_ROOT] } },
  });
  await server.listen();
  const url = server.resolvedUrls?.local[0] ?? `http://127.0.0.1:${server.config.server.port}/`;
  console.log(`dev server for ${slug}: ${url}`);
} else {
  const outDir = mkdtempSync(join(tmpdir(), `schematics-build-${slug}-`));
  try {
    await build({
      ...shared,
      logLevel: "warn",
      build: {
        outDir,
        emptyOutDir: true,
        cssCodeSplit: false,
        modulePreload: false,
        assetsInlineLimit: () => true,
        reportCompressedSize: false,
        sourcemap: false,
        rolldownOptions: { output: { codeSplitting: false } },
      },
    });
    const html = inline(outDir);
    mkdirSync(DIST_ROOT, { recursive: true });
    const target = join(DIST_ROOT, `${slug}.html`);
    writeFileSync(target, html);
    console.log(`built ${target} (${Buffer.byteLength(html)} bytes)`);
  } catch (error) {
    console.error(`error: build of ${slug} failed: ${error?.message ?? String(error)}`);
    process.exitCode = 1;
  } finally {
    rmSync(outDir, { recursive: true, force: true });
  }
}

/** Replaces the emitted <link rel=stylesheet> and <script src> with their contents. */
function inline(outDir) {
  const read = (reference) => {
    if (/^(?:[a-z]+:)?\/\//i.test(reference) || reference.startsWith("data:")) {
      throw new Error(`index.html loads an external resource: ${reference}`);
    }
    const file = join(outDir, reference.replace(/^\.?\//, "").split(/[?#]/)[0]);
    if (!existsSync(file)) {
      throw new Error(`index.html references ${reference}, which the build did not emit`);
    }
    return readFileSync(file, "utf8");
  };
  const attribute = (tag, name) => new RegExp(`\\s${name}=["']?([^"'\\s>]+)`, "i").exec(tag)?.[1];

  let html = readFileSync(join(outDir, "index.html"), "utf8");
  html = html.replace(/<link\b[^>]*>/gi, (tag) => {
    const rel = attribute(tag, "rel") ?? "";
    if (rel === "modulepreload") {
      return "";
    }
    if (rel !== "stylesheet") {
      throw new Error(`index.html keeps a <link rel="${rel}">; a single-file page cannot have one`);
    }
    const css = read(attribute(tag, "href") ?? "").replace(/<\/style/gi, "<\\/style");
    return `<style>${css.trim()}</style>`;
  });

  const scripts = [];
  html = html.replace(/<script\b[^>]*\bsrc=[^>]*>\s*<\/script>/gi, (tag) => {
    scripts.push(read(attribute(tag, "src") ?? ""));
    return scripts.length === 1 ? "<!--schematics:script-->" : "";
  });
  if (scripts.length === 0) {
    throw new Error("index.html has no module script (expected <script type=\"module\" src=\"./main.tsx\">)");
  }
  const chunks = readdirSync(outDir, { recursive: true }).filter((file) => /\.m?js$/.test(String(file)));
  if (chunks.length !== scripts.length) {
    throw new Error(`build emitted ${chunks.length} script chunks but index.html loads ${scripts.length}`);
  }
  // An inline module script is deferred like an external one, so it may stay in <head>.
  // "</script" and "<!--" would end or derail the inline script; inside JS they only
  // occur in strings and regular expressions, where \x3C is the same character.
  const js = scripts
    .join("\n")
    .trim()
    .replace(/<(?=\/script|!--)/gi, "\\x3C");
  if (/<link\b/i.test(html) || /<script\b[^>]*\ssrc=/i.test(html)) {
    throw new Error("inlining left a <link> or <script src> in the page");
  }
  return html.replace("<!--schematics:script-->", () => `<script type="module">${js}</script>`);
}
