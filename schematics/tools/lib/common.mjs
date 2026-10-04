// Paths and argument handling shared by the tools.
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
export const MODELS_ROOT = process.env.SCHEMATICS_MODELS
  ? resolve(process.env.SCHEMATICS_MODELS)
  : join(PACKAGE_ROOT, "models");
export const CACHE_ROOT = process.env.SCHEMATICS_CACHE
  ? resolve(process.env.SCHEMATICS_CACHE)
  : join(PACKAGE_ROOT, ".sources-cache");
export const DIST_ROOT = join(PACKAGE_ROOT, "dist");
export const RUNS_ROOT = join(PACKAGE_ROOT, "runs");

export const SLUG_PATTERN = /^[a-z0-9][a-z0-9._-]*$/;
export const HEX40 = /^[0-9a-f]{40}$/;

export function fail(message) {
  console.error(`error: ${message}`);
  process.exit(1);
}

/** Splits argv into positionals, `--flag` booleans and `--key value` options. */
export function parseArgs(argv, { flags = [], options = [] } = {}) {
  const positional = [];
  const values = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) {
      positional.push(arg);
      continue;
    }
    const [name, inline] = arg.slice(2).split(/=(.*)/s, 2);
    if (flags.includes(name)) {
      values[name] = true;
    } else if (options.includes(name)) {
      const value = inline ?? argv[(index += 1)];
      if (value === undefined) {
        fail(`--${name} needs a value`);
      }
      values[name] = value;
    } else {
      fail(`unknown option --${name}`);
    }
  }
  return { positional, values };
}

export function requireSlug(slug, usage) {
  if (!slug) {
    fail(`usage: ${usage}`);
  }
  if (!SLUG_PATTERN.test(slug)) {
    fail(`slug "${slug}" must match ${String(SLUG_PATTERN)}`);
  }
  return slug;
}

/** Vite root of a page: models/<slug>, or the kit gallery for `gallery`. */
export function pageRoot(slug) {
  return slug === "gallery" ? join(PACKAGE_ROOT, "kit", "gallery") : join(MODELS_ROOT, slug);
}
