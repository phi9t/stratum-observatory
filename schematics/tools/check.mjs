#!/usr/bin/env node
// usage: node tools/check.mjs <slug>
//
// Runs every gate of the README table, in order, never stopping early. Writes
// runs/check/<slug>/report.json and exits 1 if any gate failed. A gate that
// cannot run is `skipped` with the reason, never `pass`.
// env: CHROME overrides the Chrome binary used by the layout and play gates.
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

import react from "@vitejs/plugin-react";
import ts from "typescript";
import { createServer } from "vite";

import { findChrome, launchChrome, serveDirectory } from "./lib/chrome.mjs";
import { DIST_ROOT, MODELS_ROOT, PACKAGE_ROOT, RUNS_ROOT, fail, pageRoot, parseArgs, requireSlug } from "./lib/common.mjs";

const { positional } = parseArgs(process.argv.slice(2));
const slug = requireSlug(positional[0], "npm run check -- <slug>");
const root = pageRoot(slug);
if (!existsSync(root)) {
  fail(`${root} does not exist`);
}
const isGallery = slug === "gallery";
const distFile = join(DIST_ROOT, `${slug}.html`);
const LAYOUT_VIEWS = ["desktop-light", "desktop-dark", "mobile-light"];
const REPORT_LIMIT = 10;
const PLAY_SAMPLE = 12;

const gates = [];
const record = (id, status, details = []) => {
  gates.push({ id, status, details: details.map(String) });
};
const fromFailures = (id, failures) => record(id, failures.length === 0 ? "pass" : "fail", failures);
const message = (error) => error?.message ?? String(error);

/** Runs one gate; an exception inside the gate is a failure of that gate, not of the run. */
async function gate(id, body) {
  try {
    await body();
  } catch (error) {
    record(id, "fail", [`gate crashed: ${message(error)}`]);
  }
}

// --- modules under test, loaded through Vite so .ts/.tsx need no build step ---

let vite;
const loaded = {};
async function load(name, path) {
  if (!(name in loaded)) {
    try {
      vite ??= await createServer({
        configFile: false,
        root: PACKAGE_ROOT,
        logLevel: "silent",
        appType: "custom",
        plugins: [react()],
        server: { middlewareMode: true, hmr: false, ws: false, watch: null },
        optimizeDeps: { noDiscovery: true, include: [] },
      });
      loaded[name] = { module: await vite.ssrLoadModule(path) };
    } catch (error) {
      loaded[name] = { error: `cannot load ${path}: ${message(error).split("\n")[0]}` };
    }
  }
  return loaded[name];
}
const modelPath = (file) => `/${relative(PACKAGE_ROOT, join(MODELS_ROOT, slug, file)).split("\\").join("/")}`;
const loadVerify = () => load("verify", "/kit/lib/verify.ts");
const loadFacts = () => load("facts", modelPath("facts.ts"));
const loadPage = () => load("page", modelPath("Page.tsx"));

const NO_SOURCES = ["the gallery has no sources.json; this gate applies to models/<slug> only"];

// --- pins ---

await gate("pins", async () => {
  if (isGallery) {
    return record("pins", "skipped", NO_SOURCES);
  }
  const verify = await loadVerify();
  if (verify.error) {
    return record("pins", "fail", [verify.error]);
  }
  fromFailures("pins", verify.module.verifyPins(slug));
});

// --- typecheck ---

await gate("typecheck", () => {
  const tsc = join(PACKAGE_ROOT, "node_modules", "typescript", "bin", "tsc");
  if (!existsSync(tsc)) {
    return record("typecheck", "skipped", ["typescript is not installed (npm ci)"]);
  }
  const result = spawnSync(process.execPath, [tsc, "--noEmit", "-p", ".", "--pretty", "false"], {
    cwd: PACKAGE_ROOT,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.status === 0) {
    return record("typecheck", "pass");
  }
  const lines = `${result.stdout}${result.stderr}`.split("\n").filter((line) => line.trim() !== "");
  const errors = lines.filter((line) => /error TS\d+/.test(line));
  const shown = (errors.length > 0 ? errors : lines).slice(0, 30);
  if (errors.length > shown.length) {
    shown.push(`… ${errors.length - shown.length} more errors`);
  }
  record("typecheck", "fail", shown.length > 0 ? shown : [`tsc exited with ${result.status ?? result.signal}`]);
});

// --- literals ---

function literalViolations(file) {
  const source = ts.createSourceFile(file, readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const violations = [];
  const isLayoutName = (name) =>
    /^[A-Z][A-Z0-9_]*$/.test(name) && (name.endsWith("_PX") || name.endsWith("_GEOMETRY") || name.startsWith("LAYOUT"));
  const exempt = (node) => {
    // (c) an array index: xs[3]
    if (ts.isElementAccessExpression(node.parent) && node.parent.argumentExpression === node) {
      return true;
    }
    // (d) grid placement of an op: { col: 3, row: 1 }
    if (
      ts.isPropertyAssignment(node.parent) &&
      node.parent.initializer === node &&
      ts.isIdentifier(node.parent.name) &&
      (node.parent.name.text === "col" || node.parent.name.text === "row")
    ) {
      return true;
    }
    for (let current = node.parent; current; current = current.parent) {
      // (a) geometry inside a JSX attribute: x={12}, viewBox, style={{ width: 300 }}
      if (ts.isJsxAttribute(current)) {
        return true;
      }
      // (b) a top-level layout constant: const NODE_WIDTH_PX = 96
      if (
        ts.isVariableDeclaration(current) &&
        ts.isIdentifier(current.name) &&
        isLayoutName(current.name.text) &&
        ts.isVariableDeclarationList(current.parent) &&
        (current.parent.flags & ts.NodeFlags.Const) !== 0 &&
        ts.isVariableStatement(current.parent.parent) &&
        ts.isSourceFile(current.parent.parent.parent)
      ) {
        return true;
      }
    }
    return false;
  };
  const visit = (node) => {
    if (ts.isNumericLiteral(node) || ts.isBigIntLiteral(node)) {
      const value = Number(node.text.replace(/n$/, ""));
      if (!(value === 0 || value === 1 || value === 2) && !exempt(node)) {
        const { line } = source.getLineAndCharacterOfPosition(node.getStart(source));
        violations.push(`${relative(PACKAGE_ROOT, file)}:${line + 1}: numeric literal ${node.getText(source)} (take it from facts.ts)`);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return violations;
}

await gate("literals", () => {
  const directory = join(root, "sections");
  const files = existsSync(directory)
    ? readdirSync(directory)
        .filter((name) => name.endsWith(".tsx"))
        .sort()
    : [];
  if (files.length === 0) {
    return record("literals", "skipped", [`no section files in ${relative(PACKAGE_ROOT, directory)}/`]);
  }
  fromFailures("literals", files.flatMap((name) => literalViolations(join(directory, name))));
});

// --- format ---

await gate("format", () => {
  // Formatting is mechanical: `npm run format -- <slug>` fixes every finding.
  const result = spawnSync("npx", ["prettier", "--check", `${relative(PACKAGE_ROOT, root)}/**/*.{ts,tsx}`], {
    cwd: PACKAGE_ROOT,
    encoding: "utf8",
  });
  if (result.status === 0) {
    return record("format", "pass", []);
  }
  const unformatted = `${result.stdout}\n${result.stderr}`
    .split("\n")
    .filter((line) => line.startsWith("[warn] ") && /\.tsx?$/.test(line))
    .map((line) => `${line.replace("[warn] ", "")}: not formatted; run npm run format -- ${slug}`);
  record("format", "fail", unformatted.length > 0 ? unformatted : [result.stderr.trim() || "prettier --check failed"]);
});

// --- tests ---

await gate("tests", () => {
  if (isGallery) {
    return record("tests", "skipped", NO_SOURCES);
  }
  const result = spawnSync("npx", ["vitest", "run", relative(PACKAGE_ROOT, root)], { cwd: PACKAGE_ROOT, encoding: "utf8" });
  const output = `${result.stdout}\n${result.stderr}`;
  const summary = output.split("\n").find((line) => /^\s*Tests\s/.test(line))?.trim() ?? "";
  const hasReference = existsSync(join(root, "reference", "model.ts"));
  const note = hasReference ? "includes the clean reference's parity test" : "no reference/model.ts: no clean reference implementation";
  if (result.status === 0) {
    return record("tests", "pass", [`${summary}; ${note}`]);
  }
  const failed = output
    .split("\n")
    .filter((line) => /FAIL|AssertionError|Error:/.test(line))
    .slice(0, 8)
    .map((line) => line.trim());
  record("tests", "fail", failed.length > 0 ? failed : [summary || "vitest failed"]);
});

// --- facts, coverage, parameters ---

await gate("facts", async () => {
  if (isGallery) {
    return record("facts", "skipped", NO_SOURCES);
  }
  const [verify, facts] = [await loadVerify(), await loadFacts()];
  if (verify.error || facts.error) {
    return record("facts", "fail", [verify.error, facts.error].filter(Boolean));
  }
  if (!facts.module.facts || typeof facts.module.facts !== "object") {
    return record("facts", "fail", ["facts.ts does not export `facts`"]);
  }
  fromFailures("facts", verify.module.verifyFacts(slug, facts.module.facts));
});

await gate("coverage", async () => {
  if (isGallery) {
    return record("coverage", "skipped", NO_SOURCES);
  }
  const [verify, page] = [await loadVerify(), await loadPage()];
  if (verify.error || page.error) {
    return record("coverage", "fail", [verify.error, page.error].filter(Boolean));
  }
  if (!Array.isArray(page.module.sectionIds)) {
    return record("coverage", "fail", ["Page.tsx does not export `sectionIds: string[]`"]);
  }
  fromFailures("coverage", verify.module.verifyCoverage(slug, page.module.sectionIds));
});

await gate("parameters", async () => {
  if (isGallery) {
    return record("parameters", "skipped", NO_SOURCES);
  }
  const [verify, facts] = [await loadVerify(), await loadFacts()];
  if (verify.error || facts.error) {
    return record("parameters", "fail", [verify.error, facts.error].filter(Boolean));
  }
  if (facts.module.parameterTotal === undefined) {
    return record("parameters", "fail", ["facts.ts does not export `parameterTotal`"]);
  }
  const result = verify.module.verifyParameters(slug, facts.module.parameterTotal, facts.module.parameterNote);
  record("parameters", result.status, [result.detail]);
});

await vite?.close();

// --- build ---

const EXTERNAL = [
  [/<script\b[^>]*\ssrc\s*=[^>]*>/i, "script loaded by src"],
  [/<link\b[^>]*\shref\s*=\s*["']?(?:https?:)?\/\/[^>]*>/i, "external <link>"],
  [/@import\s+(?:url\(\s*)?["']?(?:https?:)?\/\/[^\s"');]*/i, "external @import"],
  [/\ssrc\s*=\s*["'](?:https?:)?\/\/[^"']*/i, "external src"],
  [/url\(\s*["']?(?:https?:)?\/\/[^)"']*/i, "external url()"],
];

let built = false;
await gate("build", () => {
  const hashes = [];
  for (const attempt of [1, 2]) {
    const result = spawnSync(process.execPath, [join(PACKAGE_ROOT, "tools", "build.mjs"), slug], {
      cwd: PACKAGE_ROOT,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    });
    if (result.status !== 0 || !existsSync(distFile)) {
      const output = `${result.stderr}\n${result.stdout}`.split("\n").filter((line) => line.trim() !== "");
      return record("build", "fail", [`build ${attempt} failed`, ...output.slice(0, 20)]);
    }
    hashes.push(createHash("sha256").update(readFileSync(distFile)).digest("hex"));
  }
  built = true;
  const failures = [];
  if (hashes[0] !== hashes[1]) {
    failures.push(`two builds differ: sha256 ${hashes[0]} vs ${hashes[1]}`);
  }
  const html = readFileSync(distFile, "utf8");
  for (const [pattern, label] of EXTERNAL) {
    const match = pattern.exec(html);
    if (match) {
      failures.push(`${label}: ${match[0].slice(0, 160)}`);
    }
  }
  if (failures.length > 0) {
    return record("build", "fail", failures);
  }
  record("build", "pass", [`dist/${slug}.html ${Buffer.byteLength(html)} bytes sha256 ${hashes[0]}`]);
});

// --- layout and play: in-page probes (serialised into the page; keep them self-contained) ---

function layoutProbe(limit) {
  const describe = (element) => {
    const parts = [];
    for (let current = element; current && current.nodeType === 1 && parts.length < 4; current = current.parentElement) {
      const classes = typeof current.className === "string" ? current.className : current.className.baseVal;
      const first = classes.trim().split(/\s+/).filter(Boolean)[0];
      parts.unshift(`${current.localName}${current.id ? `#${current.id}` : ""}${first ? `.${first}` : ""}`);
      if (current.id) break;
    }
    return parts.join(" > ");
  };
  const ownText = (element) =>
    [...element.childNodes]
      .filter((node) => node.nodeType === 3)
      .map((node) => node.textContent)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  const clip = (text) => (text.length > 60 ? `${text.slice(0, 57)}…` : text);

  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const parsed = new Map();
  const rgba = (color) => {
    if (!parsed.has(color)) {
      context.clearRect(0, 0, 1, 1);
      context.fillStyle = "rgba(0, 0, 0, 0)";
      context.fillStyle = color;
      context.fillRect(0, 0, 1, 1);
      const [r, g, b, a] = context.getImageData(0, 0, 1, 1).data;
      parsed.set(color, [r, g, b, a / 255]);
    }
    return parsed.get(color);
  };
  const over = (top, bottom) => top.slice(0, 3).map((channel, index) => channel * top[3] + bottom[index] * (1 - top[3]));
  const luminance = (rgb) => {
    const [r, g, b] = rgb.map((channel) => {
      const unit = channel / 255;
      return unit <= 0.03928 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const supportsDark = /dark/.test(getComputedStyle(document.documentElement).colorScheme);
  const canvasColor = dark && supportsDark ? [18, 18, 18] : [255, 255, 255];

  const result = {
    overflow: document.documentElement.scrollWidth > innerWidth
      ? `page scrolls sideways: scrollWidth ${document.documentElement.scrollWidth} > innerWidth ${innerWidth}`
      : null,
    small: [],
    smallCount: 0,
    svg: [],
    svgCount: 0,
    contrast: [],
    contrastCount: 0,
    textElements: 0,
  };

  for (const element of document.body.querySelectorAll("*")) {
    if (["script", "style", "noscript", "template"].includes(element.localName)) continue;
    const text = ownText(element);
    if (text === "") continue;
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    if (rect.width === 0 || rect.height === 0 || style.visibility === "hidden" || style.display === "none") continue;
    result.textElements += 1;

    const fontSize = parseFloat(style.fontSize);
    if (fontSize < 11) {
      result.smallCount += 1;
      if (result.small.length < limit) {
        result.small.push(`text under 11px (${fontSize}px): ${describe(element)} "${clip(text)}"`);
      }
    }

    if (!(element instanceof HTMLElement)) continue;
    let translucent = false;
    let painted = false;
    const layers = [];
    for (let current = element; current; current = current.parentElement) {
      const currentStyle = getComputedStyle(current);
      if (parseFloat(currentStyle.opacity) < 1) {
        translucent = true;
        break;
      }
      const background = rgba(currentStyle.backgroundColor);
      if (currentStyle.backgroundImage !== "none" && layers.every((layer) => layer[3] < 1)) {
        painted = true; // a gradient or image behind the text: no single background colour to compare with
        break;
      }
      if (background[3] > 0) {
        layers.push(background);
        if (background[3] === 1) break;
      }
    }
    if (translucent || painted) continue;
    let backdrop = canvasColor;
    for (const layer of layers.reverse()) {
      backdrop = over(layer, backdrop);
    }
    const ink = over(rgba(style.color), backdrop);
    const [lighter, darker] = [luminance(ink), luminance(backdrop)].sort((a, b) => b - a);
    const ratio = (lighter + 0.05) / (darker + 0.05);
    if (ratio < 4.5) {
      result.contrastCount += 1;
      if (result.contrast.length < limit) {
        const hex = (rgb) => `#${rgb.map((channel) => Math.round(channel).toString(16).padStart(2, "0")).join("")}`;
        result.contrast.push(
          `contrast ${ratio.toFixed(2)}:1 (${hex(ink)} on ${hex(backdrop)}): ${describe(element)} "${clip(text)}"`,
        );
      }
    }
  }

  for (const graph of document.querySelectorAll("svg[data-bands]")) {
    const bands = Number(graph.getAttribute("data-bands"));
    const rows = Number(graph.getAttribute("data-rows"));
    if (bands > 1 && rows > 1) {
      result.svgCount += 1;
      if (result.svg.length < limit) {
        result.svg.push(
          `op graph "${graph.getAttribute("data-graph")}" wraps onto ${bands} bands and uses ${rows} rows: wires tangle. Split it into two graphs at a seam (each with a heading), or shorten each op's sub line so it fits one band`,
        );
      }
    }
  }
  for (const node of document.querySelectorAll("svg g.ms-node")) {
    const rect = node.querySelector(":scope > rect");
    if (!rect) continue;
    let boxWidth;
    try {
      boxWidth = rect.getBBox().width;
    } catch {
      continue;
    }
    if (boxWidth === 0) continue;
    for (const text of node.querySelectorAll(":scope > text")) {
      let width;
      try {
        width = text.getBBox().width;
      } catch {
        continue;
      }
      if (width > boxWidth - 4) {
        result.svgCount += 1;
        if (result.svg.length < limit) {
          result.svg.push(
            `SVG text wider than its box (${width.toFixed(1)} > ${boxWidth.toFixed(1)} - 4): ${describe(node)} "${clip(text.textContent.trim())}"`,
          );
        }
      }
    }
  }
  return result;
}

/** Installs window.__msPlay: the control list, a state hash and an actuator. */
function playProbe() {
  const scope = document.querySelector(".ms");
  if (!scope) return null;
  const label = (element) => {
    const name =
      element.getAttribute("aria-label") ||
      element.getAttribute("data-op") ||
      element.getAttribute("name") ||
      element.getAttribute("value") ||
      (element.textContent || "").replace(/\s+/g, " ").trim() ||
      element.id;
    const text = `${element.localName}${element.id ? `#${element.id}` : ""} "${name ?? ""}"`;
    return text.length > 70 ? `${text.slice(0, 67)}…"` : text;
  };
  const kindOf = (element) => {
    if (element.hasAttribute("data-op")) return "op";
    if (element.localName === "select") return "select";
    if (element.localName === "button") return "button";
    return element.type === "range" ? "range" : "radio";
  };
  const controls = [
    ...scope.querySelectorAll("input[type=range], input[type=radio], select, button, [data-op]"),
  ]
    .filter((element) => !(element.localName === "button" && element.closest(".ms-nav")))
    .filter((element) => {
      const rect = element.getBoundingClientRect();
      return !element.disabled && element.getAttribute("aria-disabled") !== "true" && rect.width > 0 && rect.height > 0;
    })
    .map((element) => ({ element, kind: kindOf(element), label: label(element) }));

  const hash = () => {
    // Playable state only: the section nav tracks scroll position, not state.
    const copy = scope.cloneNode(true);
    // Form state lives in properties; write it into the copy's attributes so that a
    // clicked radio and a radio rendered checked after a reload serialise the same.
    const live = scope.querySelectorAll("input, option, textarea");
    copy.querySelectorAll("input, option, textarea").forEach((field, index) => {
      const source = live[index];
      // Remove, then append: attribute order must not depend on how the state was reached.
      const put = (name, value) => {
        field.removeAttribute(name);
        if (value !== null) field.setAttribute(name, value);
      };
      if (source.localName === "option") {
        put("selected", source.selected ? "" : null);
      } else if (source.type === "radio" || source.type === "checkbox") {
        put("checked", source.checked ? "" : null);
      } else {
        put("value", source.value);
      }
    });
    for (const nav of copy.querySelectorAll(".ms-nav")) nav.remove();
    // Attribute order depends on the order attributes were toggled; sort it away.
    for (const element of copy.querySelectorAll("*")) {
      const attributes = [...element.attributes].map((attribute) => [attribute.name, attribute.value]);
      attributes.sort((x, y) => (x[0] < y[0] ? -1 : 1));
      for (const [name] of attributes) element.removeAttribute(name);
      for (const [name, value] of attributes) element.setAttribute(name, value);
    }
    const text = copy.innerHTML;
    window.__msPlayText = text;
    let high = 0xdeadbeef;
    let low = 0x41c6ce57;
    for (let index = 0; index < text.length; index += 1) {
      const code = text.charCodeAt(index);
      high = Math.imul(high ^ code, 2654435761);
      low = Math.imul(low ^ code, 1597334677);
    }
    high = Math.imul(high ^ (high >>> 16), 2246822507) ^ Math.imul(low ^ (low >>> 13), 3266489909);
    low = Math.imul(low ^ (low >>> 16), 2246822507) ^ Math.imul(high ^ (high >>> 13), 3266489909);
    return `${(low >>> 0).toString(16).padStart(8, "0")}${(high >>> 0).toString(16).padStart(8, "0")}:${text.length}`;
  };

  const fire = (element, type) => element.dispatchEvent(new Event(type, { bubbles: true }));
  const actuate = (index) => {
    const { element, kind } = controls[index];
    if (!element.isConnected) return "gone";
    if (element.disabled) return "disabled";
    if (kind === "range") {
      const min = element.min === "" ? 0 : Number(element.min);
      const max = element.max === "" ? 100 : Number(element.max);
      const current = Number(element.value);
      const step = Number(element.step) > 0 ? Number(element.step) : 1;
      // Prefer an interior value: at an end stop, "next"/"previous" buttons have nowhere to go.
      const middle = min + Math.round((max - min) / 2 / step) * step;
      const target = [middle, max, min].find((candidate) => candidate !== current);
      if (target === undefined) return "fixed";
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(element, String(target));
      fire(element, "input");
      fire(element, "change");
    } else if (kind === "select") {
      const options = [...element.options].filter((option) => !option.disabled && option.value !== element.value);
      if (options.length === 0) return "fixed";
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set.call(element, options[0].value);
      fire(element, "input");
      fire(element, "change");
    } else if (kind === "radio" && element.checked) {
      return "checked";
    } else if (typeof element.click === "function") {
      element.click();
    } else {
      element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
    }
    return "done";
  };

  window.__msPlay = { hash, actuate };
  return controls.map(({ kind, label: text, element }) => ({
    kind,
    label: text,
    checked: kind === "radio" && element.checked,
  }));
}

const call = (probe, ...args) => `(${probe.toString()})(${args.map((arg) => JSON.stringify(arg)).join(", ")})`;

async function withChrome(ids, body) {
  const skipAll = (reason) => ids.forEach((id) => record(id, "skipped", [reason]));
  if (!built) {
    return skipAll(`dist/${slug}.html is not available because the build gate failed`);
  }
  if (!findChrome()) {
    return skipAll(`Chrome not found${process.env.CHROME ? ` at CHROME=${process.env.CHROME}` : ""} (set CHROME to the binary)`);
  }
  let server;
  let browser;
  try {
    server = await serveDirectory(DIST_ROOT);
    browser = await launchChrome();
  } catch (error) {
    await server?.close();
    return skipAll(`Chrome could not start: ${message(error)}`);
  }
  try {
    await body(browser.page, `${server.base}${slug}.html`);
  } finally {
    await browser.close();
    await server.close();
  }
}

await withChrome(["layout", "play"], async (page, url) => {
  await gate("layout", async () => {
    const failures = [];
    let textElements = 0;
    for (const view of LAYOUT_VIEWS) {
      await page.setView(view);
      await page.open(url);
      const probe = await page.evaluate(call(layoutProbe, REPORT_LIMIT));
      textElements += probe.textElements;
      if (!(await page.evaluate(`document.querySelector(".ms") !== null`))) {
        failures.push(`${view}: the page rendered no .ms root`);
      }
      if (probe.overflow) {
        failures.push(`${view}: ${probe.overflow}`);
      }
      for (const [items, count] of [
        [probe.small, probe.smallCount],
        [probe.svg, probe.svgCount],
        [probe.contrast, probe.contrastCount],
      ]) {
        failures.push(...items.map((item) => `${view}: ${item}`));
        if (count > items.length) {
          failures.push(`${view}: … ${count - items.length} more of the same kind`);
        }
      }
    }
    failures.push(...[...new Set(page.errors)].map((error) => `page error: ${error}`));
    if (failures.length > 0) {
      return record("layout", "fail", failures);
    }
    record("layout", "pass", [`${LAYOUT_VIEWS.join(", ")}: ${textElements} text elements checked`]);
  });

  await gate("play", async () => {
    await page.setView("desktop-light");
    await page.open(url.split("#")[0]);
    const controls = await page.evaluate(call(playProbe));
    if (controls === null) {
      return record("play", "fail", ["the page rendered no .ms root"]);
    }
    const failures = [];
    const kinds = [...new Set(controls.map((control) => control.kind))];
    const census = kinds.map((kind) => `${controls.filter((control) => control.kind === kind).length} ${kind}`).join(", ");
    if (controls.length < 3 || kinds.length < 2) {
      failures.push(
        `needs at least 3 controls of at least 2 kinds (range, radio, select, button, data-op); found ${controls.length}${census ? ` (${census})` : ""}`,
      );
    }

    // Sample round-robin over kinds, in page order; a checked radio cannot change anything.
    const queues = kinds.map((kind) =>
      controls.map((control, index) => ({ ...control, index })).filter((control) => control.kind === kind && !control.checked),
    );
    const sample = [];
    while (sample.length < PLAY_SAMPLE && queues.some((queue) => queue.length > 0)) {
      for (const queue of queues) {
        if (queue.length > 0 && sample.length < PLAY_SAMPLE) {
          sample.push(queue.shift());
        }
      }
    }

    const attempt = async (control) => {
      const before = await page.evaluate(`window.__msPlay.hash()`);
      const outcome = await page.evaluate(`window.__msPlay.actuate(${control.index})`);
      if (outcome !== "done") {
        return outcome;
      }
      await page.settle(150);
      return (await page.evaluate(`window.__msPlay.hash()`)) === before ? "inert" : "changed";
    };
    let changed = 0;
    const inert = [];
    for (const control of sample) {
      const outcome = await attempt(control);
      if (outcome === "changed") changed += 1;
      if (outcome === "inert" || outcome === "fixed") inert.push(control);
    }
    // A reset or "previous" control is inert in the initial state: try once more now that state moved.
    for (const control of inert) {
      const outcome = await attempt(control);
      if (outcome === "changed") {
        changed += 1;
      } else if (outcome === "inert" || outcome === "fixed") {
        failures.push(`control changes nothing on the page: ${control.kind} ${control.label}`);
      }
    }

    await page.settle(500);
    // A reset sampled last legitimately empties the hash: move state away from the defaults again.
    const emptyHash = `(location.hash === "" || location.hash === "#")`;
    for (const control of sample) {
      if (!(await page.evaluate(emptyHash))) break;
      await attempt(control);
      await page.settle(300);
    }
    const before = await page.evaluate(`({ hash: location.hash, state: window.__msPlay.hash(), text: window.__msPlayText })`);
    if (before.hash === "" || before.hash === "#") {
      failures.push("location.hash is empty after interaction: playable state is not written to the URL");
    }
    await page.reload();
    const after = await page.evaluate(`(${playProbe.toString()})() === null ? null : window.__msPlay.hash()`);
    if (after !== before.state) {
      const reloaded = (await page.evaluate(`window.__msPlayText`)) ?? "";
      let at = 0;
      while (at < before.text.length && before.text[at] === reloaded[at]) at += 1;
      const around = (text) => JSON.stringify(text.slice(Math.max(0, at - 80), at + 80));
      failures.push(
        `state does not survive a reload: .ms HTML differs after reloading ${before.hash || "(no hash)"}`,
        `  before reload: ${around(before.text)}`,
        `  after reload:  ${around(reloaded)}`,
        "  state that is not in the URL hash (useUrlNumber / useUrlChoice) is lost on reload",
      );
    }
    if (failures.length > 0) {
      return record("play", "fail", failures);
    }
    record("play", "pass", [`${controls.length} controls (${census}); ${changed} of ${sample.length} sampled changed the page; reload kept ${before.hash}`]);
  });
});

// --- report ---

const ok = gates.every((entry) => entry.status !== "fail");
const reportDir = join(RUNS_ROOT, "check", slug);
mkdirSync(reportDir, { recursive: true });
const reportFile = join(reportDir, "report.json");
writeFileSync(reportFile, `${JSON.stringify({ slug, gates, ok }, null, 2)}\n`);

const SHOWN = 12;
console.log(`\ncheck ${slug}`);
for (const entry of gates) {
  console.log(`  ${entry.id.padEnd(11)} ${entry.status}`);
  for (const detail of entry.details.slice(0, SHOWN)) {
    console.log(`      ${detail.length > 200 ? `${detail.slice(0, 199)}…` : detail}`);
  }
  if (entry.details.length > SHOWN) {
    console.log(`      … ${entry.details.length - SHOWN} more in the report`);
  }
}
const tally = (status) => gates.filter((entry) => entry.status === status).length;
console.log(`\n${ok ? "ok" : "FAILED"}: ${tally("pass")} pass, ${tally("fail")} fail, ${tally("skipped")} skipped  ->  ${relative(PACKAGE_ROOT, reportFile)}`);
process.exit(ok ? 0 : 1);
