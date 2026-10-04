#!/usr/bin/env node
// usage: node tools/shoot.mjs <slug> [--views desktop-light,desktop-dark,mobile-light] [--sels ".ms-hero,#ms-overview"]
//
// Screenshots dist/<slug>.html section by section into runs/shots/<slug>/<view>-<name>.png.
// Default selectors: .ms-hero plus every section.ms-section on the page.
// env: CHROME overrides the Chrome binary.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { VIEWS, launchChrome, serveDirectory } from "./lib/chrome.mjs";
import { DIST_ROOT, RUNS_ROOT, fail, parseArgs, requireSlug } from "./lib/common.mjs";

const { positional, values } = parseArgs(process.argv.slice(2), { options: ["views", "sels"] });
const slug = requireSlug(positional[0], "npm run shoot -- <slug> [--views a,b] [--sels x,y]");
const views = (values.views ?? "desktop-light,desktop-dark,mobile-light").split(",").filter(Boolean);
for (const view of views) {
  if (!VIEWS[view]) {
    fail(`unknown view "${view}" (known: ${Object.keys(VIEWS).join(", ")})`);
  }
}
if (!existsSync(join(DIST_ROOT, `${slug}.html`))) {
  fail(`dist/${slug}.html does not exist (npm run build -- ${slug})`);
}

const outDir = join(RUNS_ROOT, "shots", slug);
mkdirSync(outDir, { recursive: true });
const MARGIN = 8;

let server;
let browser;
let missing = 0;
try {
  server = await serveDirectory(DIST_ROOT);
  browser = await launchChrome();
  const { page } = browser;
  for (const view of views) {
    await page.setView(view);
    await page.open(`${server.base}${slug}.html`);
    const selectors = values.sels
      ? values.sels.split(",").filter(Boolean)
      : await page.evaluate(
          `[".ms-hero", ...[...document.querySelectorAll("section.ms-section")].filter((s) => s.id).map((s) => "#" + CSS.escape(s.id))]`,
        );
    for (const selector of selectors) {
      const box = await page.evaluate(`(() => {
        const element = document.querySelector(${JSON.stringify(selector)});
        if (!element) return null;
        const rect = element.getBoundingClientRect();
        return { x: rect.left + scrollX, y: rect.top + scrollY, width: rect.width, height: rect.height };
      })()`);
      if (!box || box.width === 0 || box.height === 0) {
        console.error(`${view}: nothing to shoot for ${selector}`);
        missing += 1;
        continue;
      }
      const x = Math.max(0, box.x - MARGIN);
      const png = await page.screenshot({
        x,
        y: Math.max(0, box.y),
        width: box.x - x + box.width + MARGIN,
        height: box.height,
      });
      const name = selector.replace(/^[#.]/, "").replace(/[^A-Za-z0-9_-]+/g, "-");
      const file = join(outDir, `${view}-${name}.png`);
      writeFileSync(file, png);
      console.log(file);
    }
  }
  if (page.errors.length > 0) {
    console.error(`page errors:\n  ${[...new Set(page.errors)].join("\n  ")}`);
  }
} catch (error) {
  console.error(`error: ${error.message}`);
  process.exitCode = 1;
} finally {
  await browser?.close();
  await server?.close();
}
if (missing > 0) {
  process.exitCode = 1;
}
