import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { cfg, code, derive } from "./facts";
import { indexSymbols, isCodePath, sortedIndex } from "./symbols";
import {
  cachePath,
  loadSources,
  topLevelClasses,
  verifyCoverage,
  verifyFacts,
  verifyParameters,
  verifyPins,
} from "./verify";

const SHA_A = "a".repeat(40);
const SHA_B = "b".repeat(40);
const SLUG = "toy";

const REFERENCE = `import torch


class ToyConfig:
    pass


class ToyRouter(nn.Module):
    def __init__(self, config):
        self.top_k = config.top_k

    def forward(self, hidden):
        scores = self.proj(hidden).sigmoid()
        return scores


class ToyAttention(nn.Module):
    class Inner:
        pass

    def forward(self, hidden):
        return torch.softmax(hidden, dim=-1)


def helper(x):
    return x.tanh()
`;

const SERVING = "class ToyServing:\n    pass\n";

let root: string;
let saved: { cache?: string; models?: string };

function write(file: string, content: string): void {
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, content);
}

function writeSources(overrides: Record<string, unknown>[] = []): void {
  const sources = [
    { id: "config", kind: "hf", repo: "org/Toy", revision: SHA_A, path: "config.json", role: "config" },
    {
      id: "reference",
      kind: "github",
      repo: "org/code",
      revision: SHA_B,
      path: "src/modeling_toy.py",
      role: "reference",
    },
    { id: "serving", kind: "github", repo: "org/serve", revision: SHA_B, path: "toy.py", role: "serving" },
    ...overrides,
  ];
  write(
    join(root, "models", SLUG, "sources.json"),
    JSON.stringify({ checkpoint: { repo: "org/Toy", revision: SHA_A }, sources }, null, 2),
  );
}

function writeSymbols(index: unknown): void {
  write(join(root, "models", SLUG, "symbols.json"), JSON.stringify(index, null, 2));
}

function writeCoverage(coverage: unknown): void {
  write(join(root, "models", SLUG, "coverage.json"), JSON.stringify(coverage, null, 2));
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "schematics-verify-"));
  saved = { cache: process.env.SCHEMATICS_CACHE, models: process.env.SCHEMATICS_MODELS };
  process.env.SCHEMATICS_CACHE = join(root, "cache");
  process.env.SCHEMATICS_MODELS = join(root, "models");
  writeSources();
  write(
    join(root, "cache", SLUG, "config", "config.json"),
    JSON.stringify({ hidden_size: 64, layers: [1, 2], text_config: { heads: 8 }, tie: false }),
  );
  write(join(root, "cache", SLUG, "reference", "src", "modeling_toy.py"), REFERENCE);
  write(join(root, "cache", SLUG, "serving", "toy.py"), SERVING);
  writeSymbols({ reference: indexSymbols(REFERENCE), serving: indexSymbols(SERVING) });
});

afterEach(() => {
  for (const [key, value] of [
    ["SCHEMATICS_CACHE", saved.cache],
    ["SCHEMATICS_MODELS", saved.models],
  ] as const) {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  rmSync(root, { recursive: true, force: true });
});

describe("loadSources / cachePath", () => {
  it("resolves a source id to its cached file", () => {
    expect(loadSources(SLUG).sources).toHaveLength(3);
    expect(cachePath(SLUG, "reference")).toBe(join(root, "cache", SLUG, "reference", "src", "modeling_toy.py"));
    expect(() => cachePath(SLUG, "nope")).toThrow(/no source with id "nope"/);
  });
});

describe("verifyPins", () => {
  it("passes when every revision is pinned and cached", () => {
    expect(verifyPins(SLUG)).toEqual([]);
  });

  it("flags a branch-name revision, a missing file and wrong role counts", () => {
    write(
      join(root, "models", SLUG, "sources.json"),
      JSON.stringify({
        checkpoint: { repo: "org/Toy", revision: "main" },
        sources: [
          { id: "config", kind: "hf", repo: "org/Toy", revision: "main", path: "config.json", role: "config" },
          { id: "ref2", kind: "github", repo: "o/c", revision: SHA_A, path: "gone.py", role: "reference" },
          { id: "ref3", kind: "github", repo: "o/c", revision: SHA_A, path: "gone.py", role: "reference" },
        ],
      }),
    );
    const failures = verifyPins(SLUG).join("\n");
    expect(failures).toMatch(/checkpoint: revision "main"/);
    expect(failures).toMatch(/source "config": revision "main"/);
    expect(failures).toMatch(/source "ref2": cached file missing/);
    expect(failures).toMatch(/exactly one "reference" source, found 2/);
    expect(failures).toMatch(/at least one "serving"/);
  });

  it("requires symbols.json to index every code source and to be current", () => {
    writeSymbols({ reference: indexSymbols(REFERENCE) });
    expect(verifyPins(SLUG)).toEqual(['symbols.json: no entry for source "serving" (run fetch)']);
    writeSymbols({ reference: { ToyConfig: [1, 2] }, serving: indexSymbols(SERVING) });
    expect(verifyPins(SLUG)).toEqual(['symbols.json: entry for source "reference" is stale (run fetch)']);
    rmSync(join(root, "models", SLUG, "symbols.json"));
    expect(verifyPins(SLUG).join("\n")).toMatch(/^symbols\.json: .*run fetch/);
  });

  it("reports an unreadable sources.json instead of throwing", () => {
    expect(verifyPins("absent")).toHaveLength(1);
  });
});

describe("verifyFacts", () => {
  it("accepts matching cfg, code and derived facts", () => {
    const hidden = cfg("hidden_size", 64);
    const heads = cfg("text_config.heads", 8);
    const facts = {
      hidden,
      heads,
      layers: cfg("layers", [1, 2]),
      tie: cfg("tie", false),
      scoring: code("sigmoid", { source: "reference", symbol: "ToyRouter", contains: "sigmoid()" }),
      method: code("softmax", { source: "reference", symbol: "ToyAttention.forward", contains: "softmax" }),
      fn: code("tanh", { source: "reference", symbol: "helper", contains: ".tanh()" }),
      present: code(true, { source: "serving", symbol: "ToyServing" }),
      headDim: derive(() => hidden.value / heads.value, [hidden, heads]),
    };
    expect(verifyFacts(SLUG, facts)).toEqual([]);
  });

  it("rejects wrong values, missing paths and unresolved locators", () => {
    const failures = verifyFacts(SLUG, {
      hidden: cfg("hidden_size", 65),
      asString: cfg("hidden_size", "64"),
      missing: cfg("text_config.nope", 1),
      noSymbol: code(1, { source: "reference", symbol: "Absent" }),
      // sigmoid() is in ToyRouter, not in ToyAttention: the body stops at the next class.
      wrongBody: code(1, { source: "reference", symbol: "ToyAttention", contains: "sigmoid()" }),
      // softmax is in ToyAttention.forward, not in ToyRouter.forward.
      wrongMethod: code(1, { source: "reference", symbol: "ToyRouter.forward", contains: "softmax" }),
      noSource: code(1, { source: "paper", symbol: "X" }),
      empty: { kind: "derived", value: 3, deps: [] },
    }).join("\n");
    expect(failures).toMatch(/hidden: cfg\("hidden_size"\) is 65 but config\.json holds 64/);
    expect(failures).toMatch(/asString: /);
    expect(failures).toMatch(/missing: config\.json has no "text_config\.nope"/);
    expect(failures).toMatch(/noSymbol: "Absent" is not defined/);
    expect(failures).toMatch(/wrongBody: "sigmoid\(\)" does not appear inside "ToyAttention"/);
    expect(failures).toMatch(/wrongMethod: /);
    expect(failures).toMatch(/noSource: sources\.json has no source "paper"/);
    expect(failures).toMatch(/empty: derived fact names no deps/);
  });

  it("asks for a qualified name when a bare method name is ambiguous", () => {
    const failures = verifyFacts(SLUG, {
      ambiguous: code(1, { source: "reference", symbol: "forward" }),
    });
    expect(failures).toHaveLength(1);
    expect(failures[0]).toMatch(/defined more than once.*ToyRouter\.forward, ToyAttention\.forward/);
  });
});

describe("indexSymbols", () => {
  it("gives 1-based inclusive extents, qualified keys and unique bare names", () => {
    const index = indexSymbols(REFERENCE);
    expect(index.ToyConfig).toEqual([4, 5]);
    expect(index.ToyRouter).toEqual([8, 14]);
    expect(index["ToyRouter.__init__"]).toEqual([9, 10]);
    expect(index["ToyRouter.forward"]).toEqual([12, 14]);
    expect(index["ToyAttention.Inner"]).toEqual([18, 19]);
    expect(index.Inner).toEqual([18, 19]);
    expect(index.__init__).toEqual([9, 10]);
    expect(index.helper).toEqual([25, 26]);
    expect(index.forward).toBeUndefined();
  });

  it("leaves the next definition's decorators out of an extent", () => {
    const text = [
      "class A:",
      "    def f(self):",
      "        return 1",
      "    @property",
      "    def g(self):",
      "        return 2",
      "",
      "@decorate(",
      "    option=True,",
      ")",
      "class B:",
      "    pass",
    ].join("\n");
    const index = indexSymbols(text);
    expect(index["A.f"]).toEqual([2, 3]);
    expect(index["A.g"]).toEqual([5, 6]);
    expect(index.A).toEqual([1, 6]);
    expect(index.B).toEqual([11, 12]);
  });

  it("indexes JS-style definitions and skips non-code paths", () => {
    const script = "export async function run(a) {\n  return a + 1;\n}\n\nexport class Box {\n  size = 3;\n}\n";
    expect(indexSymbols(script)).toEqual({ run: [1, 3], Box: [5, 7] });
    expect(isCodePath("src/modeling_toy.py")).toBe(true);
    expect(["config.json", "paper.PDF", "README.md"].map(isCodePath)).toEqual([false, false, false]);
    expect(Object.keys(sortedIndex({ b: { z: [1, 1], a: [2, 2] }, a: {} }))).toEqual(["a", "b"]);
    expect(Object.keys(sortedIndex({ b: { z: [1, 1], a: [2, 2] } }).b)).toEqual(["a", "z"]);
  });
});

describe("verifyCoverage", () => {
  const complete = {
    source: "reference",
    classes: {
      ToyConfig: { excluded: "configuration container" },
      ToyRouter: { section: "router" },
      ToyAttention: { section: "attention" },
    },
    serving: { source: "serving", notes: "none found" },
  };

  it("lists only column-0 classes", () => {
    expect(topLevelClasses(REFERENCE)).toEqual(["ToyConfig", "ToyRouter", "ToyAttention"]);
  });

  it("passes when every class is mapped or excluded", () => {
    writeCoverage(complete);
    expect(verifyCoverage(SLUG, ["router", "attention"])).toEqual([]);
  });

  it("flags unmapped classes, missing sections, empty reasons, stale entries and empty notes", () => {
    writeCoverage({
      classes: {
        ToyConfig: { excluded: " " },
        ToyRouter: { section: "routing" },
        Ghost: { section: "router" },
      },
      serving: { notes: "" },
    });
    const failures = verifyCoverage(SLUG, ["router"]).join("\n");
    expect(failures).toMatch(/class ToyConfig: "excluded" needs a non-empty reason/);
    expect(failures).toMatch(/class ToyRouter: section "routing" is not in Page\.tsx sectionIds/);
    expect(failures).toMatch(/class ToyAttention: not in coverage\.json/);
    expect(failures).toMatch(/class Ghost: listed in coverage\.json but not defined/);
    expect(failures).toMatch(/serving\.notes/);
  });

  it('lets coverage.json "source" override the reference role', () => {
    writeCoverage({
      source: "serving",
      classes: { ToyServing: { section: "serving" } },
      serving: { notes: "this is the serving file" },
    });
    expect(verifyCoverage(SLUG, ["serving"])).toEqual([]);
  });
});

describe("verifyParameters", () => {
  const info = (body: unknown): void => write(join(root, "cache", SLUG, "model-info.json"), JSON.stringify(body));

  it("is skipped without model-info or without a total", () => {
    expect(verifyParameters(SLUG, 10).status).toBe("skipped");
    info({ id: "org/Toy" });
    expect(verifyParameters(SLUG, 10).status).toBe("skipped");
  });

  it("passes on equality, fails on a difference, and passes a noted difference", () => {
    info({ safetensors: { total: 1000 } });
    expect(verifyParameters(SLUG, 1000).status).toBe("pass");
    const base = cfg("hidden_size", 64);
    expect(
      verifyParameters(
        SLUG,
        derive(() => 1000, [base]),
      ).status,
    ).toBe("pass");
    const failed = verifyParameters(SLUG, 990);
    expect(failed.status).toBe("fail");
    expect(failed.detail).toMatch(/990 differs from safetensors\.total 1000/);
    expect(verifyParameters(SLUG, 990, "  ").status).toBe("fail");
    expect(verifyParameters(SLUG, 990, "tied embeddings counted once").status).toBe("pass");
  });
});
