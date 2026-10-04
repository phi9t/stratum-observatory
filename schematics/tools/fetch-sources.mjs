#!/usr/bin/env node
// usage: node tools/fetch-sources.mjs <slug>
//
// Pins every revision in models/<slug>/sources.json to a commit, downloads the
// pinned files into .sources-cache/<slug>/<id>/<path>, saves the Hugging Face
// model-info response, and writes models/<slug>/symbols.json.
//
// env: HF_ENDPOINT (e.g. https://hf-mirror.com), HF_TOKEN, GITHUB_TOKEN
import { execFile } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import { promisify } from "node:util";

import { indexSymbols, isCodePath, sortedIndex } from "../kit/lib/symbols.ts";
import { CACHE_ROOT, HEX40, MODELS_ROOT, fail, parseArgs, requireSlug } from "./lib/common.mjs";

const run = promisify(execFile);
const HF = (process.env.HF_ENDPOINT || "https://huggingface.co").replace(/\/+$/, "");
const TIMEOUT_MS = 120_000;
const ROLES = ["config", "reference", "serving", "training", "paper", "extra"];

const { positional } = parseArgs(process.argv.slice(2));
const slug = requireSlug(positional[0], "npm run fetch -- <slug>");
const sourcesFile = join(MODELS_ROOT, slug, "sources.json");
if (!existsSync(sourcesFile)) {
  fail(`${sourcesFile} does not exist (npm run new -- ${slug} <hf-repo>)`);
}

function headers(kind) {
  const result = { "user-agent": "stratum-model-schematics" };
  if (kind === "hf" && process.env.HF_TOKEN) {
    result.authorization = `Bearer ${process.env.HF_TOKEN}`;
  }
  if (kind === "github" && process.env.GITHUB_TOKEN) {
    result.authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  }
  return result;
}

async function request(url, kind) {
  let response;
  try {
    response = await fetch(url, {
      headers: headers(kind),
      redirect: "follow",
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (error) {
    const cause = error?.cause?.code ?? error?.cause?.message ?? error?.message ?? String(error);
    throw new Error(`GET ${url}: ${cause}`);
  }
  return response;
}

function httpError(url, response, hint = "") {
  const reasons = {
    401: "unauthorized (no such repo, or gated/private: set HF_TOKEN / GITHUB_TOKEN)",
    403: "forbidden (gated, private or rate-limited; set HF_TOKEN / GITHUB_TOKEN)",
    404: "not found (check repo, revision and path)",
    422: "no such revision",
    429: "rate-limited",
  };
  return new Error(
    `GET ${url}: HTTP ${response.status} ${reasons[response.status] ?? response.statusText}${hint}`,
  );
}

function encodePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

async function hfModelInfo(repo, revision) {
  const url = `${HF}/api/models/${repo}/revision/${encodeURIComponent(revision)}`;
  const response = await request(url, "hf");
  if (!response.ok) {
    throw httpError(url, response);
  }
  const info = await response.json();
  if (!HEX40.test(String(info.sha ?? ""))) {
    throw new Error(`GET ${url}: response has no 40-hex "sha"`);
  }
  return info;
}

async function githubCommit(repo, revision) {
  const url = `https://api.github.com/repos/${repo}/commits/${encodeURIComponent(revision)}`;
  const response = await request(url, "github").catch((error) => ({ ok: false, status: 0, error }));
  if (response.ok) {
    const sha = (await response.json()).sha;
    if (HEX40.test(String(sha ?? ""))) {
      return sha;
    }
    throw new Error(`GET ${url}: response has no 40-hex "sha"`);
  }
  if (response.status === 404 || response.status === 422) {
    throw httpError(url, response);
  }
  // Rate-limited or unreachable API: ask the git remote instead.
  const remote = `https://github.com/${repo}`;
  try {
    const { stdout } = await run("git", ["ls-remote", remote, revision, `${revision}^{}`], {
      timeout: TIMEOUT_MS,
    });
    const rows = stdout
      .split("\n")
      .filter(Boolean)
      .map((row) => row.split("\t"));
    const peeled = rows.find(([, ref]) => ref.endsWith("^{}"));
    const sha = (peeled ?? rows[0])?.[0];
    if (HEX40.test(String(sha ?? ""))) {
      return sha;
    }
    throw new Error(`no ref named "${revision}"`);
  } catch (error) {
    const first = response.status ? `HTTP ${response.status}` : String(response.error?.message);
    throw new Error(
      `cannot resolve ${repo}@${revision}: GitHub API ${first}; git ls-remote ${remote}: ${error.message.trim()}`,
    );
  }
}

async function download(url, kind, target) {
  const response = await request(url, kind);
  if (!response.ok) {
    throw httpError(url, response);
  }
  const body = Buffer.from(await response.arrayBuffer());
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, body);
  return body.length;
}

function validate(sources) {
  const problems = [];
  if (!sources || typeof sources !== "object") {
    return ["sources.json is not an object"];
  }
  if (!sources.checkpoint?.repo) {
    problems.push("checkpoint.repo is missing");
  }
  if (!Array.isArray(sources.sources)) {
    return [...problems, '"sources" is not an array'];
  }
  const ids = new Set();
  sources.sources.forEach((source, index) => {
    const label = `sources[${index}]${source?.id ? ` ("${source.id}")` : ""}`;
    for (const key of ["id", "kind", "repo", "path", "role"]) {
      if (typeof source?.[key] !== "string" || source[key] === "") {
        problems.push(`${label}: "${key}" is missing`);
      }
    }
    if (source?.kind && source.kind !== "hf" && source.kind !== "github") {
      problems.push(`${label}: kind "${source.kind}" is not "hf" or "github"`);
    }
    if (source?.role && !ROLES.includes(source.role)) {
      problems.push(`${label}: role "${source.role}" is not one of ${ROLES.join(", ")}`);
    }
    if (source?.id && (ids.has(source.id) || !/^[A-Za-z0-9][\w.-]*$/.test(source.id))) {
      problems.push(`${label}: id must be unique and a plain name`);
    }
    ids.add(source?.id);
    const cacheDir = join(CACHE_ROOT, slug, String(source?.id));
    if (source?.path && !resolve(cacheDir, source.path).startsWith(cacheDir + sep)) {
      problems.push(`${label}: path escapes the cache directory`);
    }
  });
  return problems;
}

let sources;
try {
  sources = JSON.parse(readFileSync(sourcesFile, "utf8"));
} catch (error) {
  fail(`${sourcesFile}: ${error.message}`);
}
const problems = validate(sources);
if (problems.length > 0) {
  fail(`${sourcesFile}:\n  ${problems.join("\n  ")}`);
}

const failures = [];
const save = () => writeFileSync(sourcesFile, `${JSON.stringify(sources, null, 2)}\n`);
const cacheDir = join(CACHE_ROOT, slug);
mkdirSync(cacheDir, { recursive: true });

// Checkpoint: pin, and keep the model-info response for the parameters gate.
try {
  const revision = sources.checkpoint.revision || "main";
  const infoFile = join(cacheDir, "model-info.json");
  let cached;
  if (HEX40.test(revision) && existsSync(infoFile)) {
    try {
      cached = JSON.parse(readFileSync(infoFile, "utf8"));
    } catch {
      cached = undefined;
    }
  }
  if (cached?.sha === revision) {
    console.log(`checkpoint  ${sources.checkpoint.repo}@${revision.slice(0, 12)}  model-info cached`);
  } else {
    const info = await hfModelInfo(sources.checkpoint.repo, revision);
    sources.checkpoint.revision = info.sha;
    save();
    writeFileSync(infoFile, `${JSON.stringify(info, null, 2)}\n`);
    const total = info.safetensors?.total;
    console.log(
      `checkpoint  ${sources.checkpoint.repo}@${info.sha.slice(0, 12)}  safetensors.total=${total ?? "not reported"}`,
    );
  }
} catch (error) {
  failures.push(`checkpoint: ${error.message}`);
}

const resolved = new Map();
async function pin(source) {
  if (HEX40.test(String(source.revision ?? ""))) {
    return source.revision;
  }
  const revision = source.revision || "main";
  const key = `${source.kind}:${source.repo}@${revision}`;
  if (!resolved.has(key)) {
    resolved.set(
      key,
      source.kind === "hf"
        ? (await hfModelInfo(source.repo, revision)).sha
        : await githubCommit(source.repo, revision),
    );
  }
  return resolved.get(key);
}

for (const source of sources.sources) {
  try {
    const sha = await pin(source);
    if (source.revision !== sha) {
      source.revision = sha;
      save();
    }
    const target = join(cacheDir, source.id, source.path);
    const sidecar = `${target}.sha`;
    const label = `${source.id.padEnd(11)} ${source.repo}@${sha.slice(0, 12)}  ${source.path}`;
    if (existsSync(target) && existsSync(sidecar) && readFileSync(sidecar, "utf8").trim() === sha) {
      console.log(`${label}  cached`);
      continue;
    }
    const url =
      source.kind === "hf"
        ? `${HF}/${source.repo}/resolve/${sha}/${encodePath(source.path)}`
        : `https://raw.githubusercontent.com/${source.repo}/${sha}/${encodePath(source.path)}`;
    const bytes = await download(url, source.kind, target);
    writeFileSync(sidecar, `${sha}\n`);
    console.log(`${label}  ${bytes} bytes`);
  } catch (error) {
    failures.push(`source "${source.id}": ${error.message}`);
  }
}

// symbols.json: definition extents of every cached code source.
const symbolsFile = join(MODELS_ROOT, slug, "symbols.json");
const index = {};
for (const source of sources.sources) {
  const target = join(cacheDir, source.id, source.path);
  if (isCodePath(source.path) && existsSync(target)) {
    index[source.id] = indexSymbols(readFileSync(target, "utf8"));
  }
}
writeFileSync(symbolsFile, `${JSON.stringify(sortedIndex(index), null, 2)}\n`);
console.log(
  `symbols     ${Object.entries(index)
    .map(([id, symbols]) => `${id}:${Object.keys(symbols).length}`)
    .join(" ") || "no code sources"}  -> ${symbolsFile}`,
);

if (failures.length > 0) {
  console.error(`\nfetch failed for ${slug}:`);
  for (const failure of failures) {
    console.error(`  ${failure}`);
  }
  process.exit(1);
}
console.log(`fetch ok: ${sources.sources.length} sources pinned for ${slug}`);
