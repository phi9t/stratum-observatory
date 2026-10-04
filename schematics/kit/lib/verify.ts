// Node-only checks behind the `pins`, `facts`, `coverage` and `parameters` gates.
// Used by each model's tests/facts.test.ts and by tools/check.mjs. Never import
// this from page code: it reads the filesystem.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

import type { Fact } from "./facts";
import { indexSymbols, isCodePath, sliceLines, type FileSymbols } from "./symbols";

export type SourceKind = "hf" | "github";

export type Source = {
  id: string;
  kind: SourceKind;
  repo: string;
  revision: string;
  path: string;
  role: string;
};

export type Sources = {
  checkpoint: { repo: string; revision: string };
  sources: Source[];
};

export type CoverageEntry = { section?: string; excluded?: string };

export type Coverage = {
  source?: string;
  classes?: Record<string, CoverageEntry>;
  serving?: { source?: string; notes?: string };
};

export type ParameterResult = {
  status: "pass" | "fail" | "skipped";
  detail: string;
};

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const HEX40 = /^[0-9a-f]{40}$/;

/** `models/` root; `SCHEMATICS_MODELS` overrides it (tests). */
export function modelsRoot(): string {
  const override = process.env.SCHEMATICS_MODELS;
  return override ? resolve(override) : join(PACKAGE_ROOT, "models");
}

/** `.sources-cache/` root; `SCHEMATICS_CACHE` overrides it (tests). */
export function cacheRoot(): string {
  const override = process.env.SCHEMATICS_CACHE;
  return override ? resolve(override) : join(PACKAGE_ROOT, ".sources-cache");
}

function readJson(file: string): unknown {
  return JSON.parse(readFileSync(file, "utf8"));
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function loadSources(slug: string): Sources {
  const file = join(modelsRoot(), slug, "sources.json");
  const parsed = readJson(file) as Partial<Sources> | null;
  if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.sources)) {
    throw new Error(`${file}: expected { checkpoint, sources: [...] }`);
  }
  return {
    checkpoint: parsed.checkpoint ?? { repo: "", revision: "" },
    sources: parsed.sources,
  };
}

/** Where `fetch` puts the file of one source: `.sources-cache/<slug>/<id>/<path>`. */
export function cachePath(slug: string, sourceId: string): string {
  const source = loadSources(slug).sources.find((candidate) => candidate.id === sourceId);
  if (!source) {
    throw new Error(`sources.json of ${slug} has no source with id "${sourceId}"`);
  }
  return join(cacheRoot(), slug, source.id, source.path);
}

export function verifyPins(slug: string): string[] {
  let sources: Sources;
  try {
    sources = loadSources(slug);
  } catch (error) {
    return [`sources.json: ${describe(error)}`];
  }
  const failures: string[] = [];

  if (!sources.checkpoint.repo) {
    failures.push("checkpoint: repo is missing");
  }
  if (!HEX40.test(String(sources.checkpoint.revision ?? ""))) {
    failures.push(`checkpoint: revision "${String(sources.checkpoint.revision)}" is not a 40-hex commit (run fetch)`);
  }

  const seen = new Set<string>();
  for (const source of sources.sources) {
    const label = `source "${String(source.id)}"`;
    if (!source.id || !source.repo || !source.path) {
      failures.push(`${label}: id, repo and path are required`);
      continue;
    }
    if (seen.has(source.id)) {
      failures.push(`${label}: duplicate id`);
    }
    seen.add(source.id);
    if (source.kind !== "hf" && source.kind !== "github") {
      failures.push(`${label}: kind "${String(source.kind)}" is not "hf" or "github"`);
    }
    if (!HEX40.test(String(source.revision ?? ""))) {
      failures.push(`${label}: revision "${String(source.revision)}" is not a 40-hex commit (run fetch)`);
    }
    const cached = join(cacheRoot(), slug, source.id, source.path);
    if (!existsSync(cached)) {
      failures.push(`${label}: cached file missing: ${cached} (run fetch)`);
    }
  }

  failures.push(...verifySymbolIndex(slug, sources));

  const count = (role: string): number => sources.sources.filter((source) => source.role === role).length;
  if (count("config") !== 1) {
    failures.push(`roles: expected exactly one "config" source, found ${count("config")}`);
  }
  if (count("reference") !== 1) {
    failures.push(`roles: expected exactly one "reference" source, found ${count("reference")}`);
  }
  if (count("serving") < 1) {
    failures.push('roles: expected at least one "serving" source, found 0');
  }
  return failures;
}

// ---------------------------------------------------------------------------
// Symbol lookup

/** symbols.json must index every cached code source, and match what the cache holds. */
function verifySymbolIndex(slug: string, sources: Sources): string[] {
  const codeSources = sources.sources.filter((source) => source.id && source.path && isCodePath(source.path));
  if (codeSources.length === 0) {
    return [];
  }
  const file = join(modelsRoot(), slug, "symbols.json");
  let index: Record<string, FileSymbols>;
  try {
    index = readJson(file) as Record<string, FileSymbols>;
  } catch (error) {
    return [`symbols.json: ${describe(error)} (run fetch)`];
  }
  const failures: string[] = [];
  for (const source of codeSources) {
    const cached = join(cacheRoot(), slug, source.id, source.path);
    if (!existsSync(cached)) {
      continue; // already reported as a missing cached file
    }
    const entry = index?.[source.id];
    if (!entry || typeof entry !== "object") {
      failures.push(`symbols.json: no entry for source "${source.id}" (run fetch)`);
    } else if (!isDeepStrictEqual(entry, indexSymbols(readFileSync(cached, "utf8")))) {
      failures.push(`symbols.json: entry for source "${source.id}" is stale (run fetch)`);
    }
  }
  return failures;
}

/** Names of the classes defined at column 0 (Python `class Name(...)` / `class Name:`). */
export function topLevelClasses(text: string): string[] {
  const names: string[] = [];
  for (const match of text.matchAll(/^class\s+([A-Za-z_]\w*)\s*[(:]/gm)) {
    if (!names.includes(match[1])) {
      names.push(match[1]);
    }
  }
  return names;
}

function lookup(root: unknown, path: string): { found: boolean; value?: unknown } {
  let current: unknown = root;
  for (const part of path.split(".")) {
    if (current === null || typeof current !== "object" || !(part in (current as object))) {
      return { found: false };
    }
    current = (current as Record<string, unknown>)[part];
  }
  return { found: true, value: current };
}

// ---------------------------------------------------------------------------
// Facts

export function verifyFacts(slug: string, facts: Record<string, Fact>): string[] {
  let sources: Sources;
  try {
    sources = loadSources(slug);
  } catch (error) {
    return [`sources.json: ${describe(error)}`];
  }
  const failures: string[] = [];
  const entries = Object.entries(facts ?? {});
  if (entries.length === 0) {
    failures.push("facts: the `facts` export is empty");
  }

  let config: { loaded: boolean; value?: unknown; error?: string } | undefined;
  const loadConfig = (): { loaded: boolean; value?: unknown; error?: string } => {
    if (!config) {
      const source = sources.sources.find((candidate) => candidate.role === "config");
      if (!source) {
        config = { loaded: false, error: 'no source with role "config"' };
      } else {
        const file = join(cacheRoot(), slug, source.id, source.path);
        try {
          config = { loaded: true, value: readJson(file) };
        } catch (error) {
          config = { loaded: false, error: `cannot read ${file}: ${describe(error)}` };
        }
      }
    }
    return config;
  };

  const texts = new Map<string, { text?: string; symbols?: FileSymbols; error?: string }>();
  const loadText = (sourceId: string): { text?: string; symbols?: FileSymbols; error?: string } => {
    let entry = texts.get(sourceId);
    if (!entry) {
      const source = sources.sources.find((candidate) => candidate.id === sourceId);
      if (!source) {
        entry = { error: `sources.json has no source "${sourceId}"` };
      } else {
        const file = join(cacheRoot(), slug, source.id, source.path);
        try {
          const text = readFileSync(file, "utf8");
          entry = { text, symbols: indexSymbols(text) };
        } catch (error) {
          entry = { error: `cannot read ${file}: ${describe(error)}` };
        }
      }
      texts.set(sourceId, entry);
    }
    return entry;
  };

  for (const [name, fact] of entries) {
    if (!fact || typeof fact !== "object") {
      failures.push(`${name}: not a Fact`);
      continue;
    }
    if (fact.kind === "cfg") {
      if (!fact.path) {
        failures.push(`${name}: cfg fact has no path`);
        continue;
      }
      const loaded = loadConfig();
      if (!loaded.loaded) {
        failures.push(`${name}: ${loaded.error ?? "config unavailable"}`);
        continue;
      }
      const hit = lookup(loaded.value, fact.path);
      if (!hit.found) {
        failures.push(`${name}: config.json has no "${fact.path}"`);
      } else if (!isDeepStrictEqual(hit.value, fact.value)) {
        failures.push(
          `${name}: cfg("${fact.path}") is ${JSON.stringify(fact.value)} but config.json holds ${JSON.stringify(hit.value)}`,
        );
      }
    } else if (fact.kind === "code") {
      const locator = fact.locator;
      if (!locator || !locator.source || !locator.symbol) {
        failures.push(`${name}: code fact needs locator.source and locator.symbol`);
        continue;
      }
      const loaded = loadText(locator.source);
      if (loaded.text === undefined) {
        failures.push(`${name}: ${loaded.error ?? "source unavailable"}`);
        continue;
      }
      const symbols = loaded.symbols as FileSymbols;
      const range = Object.hasOwn(symbols, locator.symbol) ? symbols[locator.symbol] : undefined;
      if (!range) {
        const qualified = Object.keys(symbols).filter((key) => key.endsWith(`.${locator.symbol}`));
        failures.push(
          qualified.length > 0
            ? `${name}: "${locator.symbol}" is defined more than once in source "${locator.source}"; use one of ${qualified.join(", ")}`
            : `${name}: "${locator.symbol}" is not defined in source "${locator.source}"`,
        );
      } else if (locator.contains !== undefined && !sliceLines(loaded.text, range).includes(locator.contains)) {
        failures.push(
          `${name}: ${JSON.stringify(locator.contains)} does not appear inside "${locator.symbol}" (lines ${String(range[0])}-${String(range[1])}) in source "${locator.source}"`,
        );
      }
    } else if (fact.kind === "derived") {
      if (!Array.isArray(fact.deps) || fact.deps.length === 0) {
        failures.push(`${name}: derived fact names no deps`);
      }
    } else {
      failures.push(`${name}: unknown fact kind "${String((fact as { kind?: unknown }).kind)}"`);
    }
  }
  return failures;
}

// ---------------------------------------------------------------------------
// Coverage

export function verifyCoverage(slug: string, sectionIds: string[]): string[] {
  const failures: string[] = [];
  let sources: Sources;
  let coverage: Coverage;
  try {
    sources = loadSources(slug);
  } catch (error) {
    return [`sources.json: ${describe(error)}`];
  }
  const coverageFile = join(modelsRoot(), slug, "coverage.json");
  try {
    coverage = readJson(coverageFile) as Coverage;
  } catch (error) {
    return [`coverage.json: ${describe(error)}`];
  }
  if (!Array.isArray(sectionIds)) {
    return ["Page.tsx: `sectionIds` export is not an array"];
  }

  const source = coverage.source
    ? sources.sources.find((candidate) => candidate.id === coverage.source)
    : sources.sources.find((candidate) => candidate.role === "reference");
  if (!source) {
    failures.push(
      coverage.source
        ? `coverage.json: "source" names "${coverage.source}", which is not in sources.json`
        : 'sources.json has no source with role "reference"',
    );
  } else {
    const file = join(cacheRoot(), slug, source.id, source.path);
    let text: string | undefined;
    try {
      text = readFileSync(file, "utf8");
    } catch (error) {
      failures.push(`cannot read reference source ${file}: ${describe(error)}`);
    }
    if (text !== undefined) {
      const defined = topLevelClasses(text);
      const mapped = coverage.classes ?? {};
      for (const name of defined) {
        const entry = mapped[name];
        if (!entry || typeof entry !== "object") {
          failures.push(`class ${name}: not in coverage.json`);
          continue;
        }
        const hasSection = typeof entry.section === "string" && entry.section !== "";
        const hasExcluded = entry.excluded !== undefined;
        if (hasSection && hasExcluded) {
          failures.push(`class ${name}: give either "section" or "excluded", not both`);
        } else if (hasSection) {
          if (!sectionIds.includes(entry.section as string)) {
            failures.push(`class ${name}: section "${entry.section}" is not in Page.tsx sectionIds`);
          }
        } else if (hasExcluded) {
          if (typeof entry.excluded !== "string" || entry.excluded.trim() === "") {
            failures.push(`class ${name}: "excluded" needs a non-empty reason`);
          }
        } else {
          failures.push(`class ${name}: entry needs "section" or "excluded"`);
        }
      }
      for (const name of Object.keys(mapped)) {
        if (!defined.includes(name)) {
          failures.push(`class ${name}: listed in coverage.json but not defined in ${source.path}`);
        }
      }
    }
  }

  const serving = coverage.serving;
  if (!serving || typeof serving.notes !== "string" || serving.notes.trim() === "") {
    failures.push('coverage.json: "serving.notes" must say how serving differs, or "none found"');
  }
  if (serving?.source && !sources.sources.some((candidate) => candidate.id === serving.source)) {
    failures.push(`coverage.json: serving.source "${serving.source}" is not in sources.json`);
  }
  return failures;
}

// ---------------------------------------------------------------------------
// Parameters

export function verifyParameters(
  slug: string,
  parameterTotal: number | Fact<number>,
  parameterNote?: string,
): ParameterResult {
  const claimed = typeof parameterTotal === "object" && parameterTotal !== null ? parameterTotal.value : parameterTotal;
  if (typeof claimed !== "number" || !Number.isFinite(claimed)) {
    return { status: "fail", detail: `parameterTotal is not a finite number: ${String(claimed)}` };
  }
  const file = join(cacheRoot(), slug, "model-info.json");
  if (!existsSync(file)) {
    return { status: "skipped", detail: `no ${file} (run fetch)` };
  }
  let info: { safetensors?: { total?: unknown } };
  try {
    info = readJson(file) as { safetensors?: { total?: unknown } };
  } catch (error) {
    return { status: "skipped", detail: `cannot parse ${file}: ${describe(error)}` };
  }
  const total = info?.safetensors?.total;
  if (typeof total !== "number") {
    return { status: "skipped", detail: "the hub reports no safetensors.total for this checkpoint" };
  }
  if (total === claimed) {
    return { status: "pass", detail: `parameterTotal ${String(claimed)} equals safetensors.total` };
  }
  const difference = `parameterTotal ${String(claimed)} differs from safetensors.total ${String(total)} by ${String(claimed - total)}`;
  if (typeof parameterNote === "string" && parameterNote.trim() !== "") {
    return { status: "pass", detail: `${difference}; parameterNote: ${parameterNote.trim()}` };
  }
  return { status: "fail", detail: `${difference} and facts.ts exports no parameterNote` };
}
