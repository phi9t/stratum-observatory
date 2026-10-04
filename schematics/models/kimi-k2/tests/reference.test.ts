import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { Ctx, maxAbsDiff, tensor } from "../../../tnsr/tensor";
import fixture from "../reference/fixture.json";
import { forward, type KimiConfig, routeScores, yarnFrequency } from "../reference/model";
import { real, parameterTotal, latentWidth, expandedCache, scale } from "../facts";

describe("Kimi K2 clean reference", () => {
  it("matches pinned upstream logits, latent, attention and each residual stream", () => {
    const ctx = new Ctx("value", fixture.weights);
    const out = forward(ctx, fixture.config as KimiConfig, fixture.input_ids);
    expect(maxAbsDiff(out, tensor(fixture.logits.shape, fixture.logits.data))).toBeLessThan(1e-4);
    for (const [name, value] of Object.entries(fixture.trace)) {
      expect(maxAbsDiff(ctx.trace.get(name)!, tensor(value.shape, value.data)), name).toBeLessThan(1e-4);
    }
    expect([...ctx.params.keys()].sort()).toEqual(Object.keys(fixture.weights).sort());
    const ids = ctx.trace.get("h2.mlp.indices")!.data!;
    const weights = ctx.trace.get("h2.mlp.weights")!.data!;
    const upstream = fixture.router2;
    for (let row = 0; row < ids.length / fixture.config.topK; row++) {
      const start = row * fixture.config.topK;
      const actual = Array.from(ids.slice(start, start + fixture.config.topK)).sort();
      const expected = upstream.indices.data.slice(start, start + fixture.config.topK).sort();
      expect(actual).toEqual(expected);
      for (let slot = 0; slot < fixture.config.topK; slot++) {
        const target = upstream.indices.data.slice(start, start + fixture.config.topK).indexOf(ids[start + slot]);
        expect(Math.abs(weights[start + slot] - upstream.weights.data[start + target])).toBeLessThan(1e-5);
      }
    }
  });
  it("binds the fixture to current pinned source bytes", () => {
    const source = readFileSync(".sources-cache/kimi-k2/reference/modeling_deepseek.py");
    expect(createHash("sha256").update(source).digest("hex")).toBe(fixture.source_sha256);
  });
  it("reproduces the real parameter ledger without allocating model weights", () => {
    const ctx = new Ctx("shape");
    expect(forward(ctx, real, [[0, 1, 2]]).shape).toEqual([1, 3, real.vocab]);
    expect(ctx.parameterCount()).toBe(parameterTotal.value);
    expect(ctx.shapeOf("h0.attn.latent")).toEqual([1, 3, real.kvRank]);
    expect(ctx.shapeOf("h0.attn.q")).toEqual([1, real.heads, 3, real.nope + real.rope]);
    expect(ctx.shapeOf("h60.mlp.indices")).toEqual([3, real.topK]);
    expect(latentWidth.value).toBe(real.kvRank + real.rope);
    expect(expandedCache.value).toBe(real.heads * (real.nope + real.rope + real.v));
  });
  it("correction bias selects experts but never weights their outputs", () => {
    const a = routeScores([0, 1, -1], [10, 0, 0], 2, scale.value);
    expect(a.chosen).toEqual([0, 1]);
    expect(a.weights[0] / a.weights[1]).toBeCloseTo(a.scores[0] / a.scores[1]);
    expect(a.weights.reduce((x, y) => x + y, 0)).toBeCloseTo(scale.value);
    expect(() => routeScores([0, 1], [0], 2, 1)).toThrow();
    expect(() => routeScores([0, 1], [0, 0], 3, 1)).toThrow();
  });
  it("YaRN preserves high frequencies and stretches low frequencies", () => {
    expect(yarnFrequency(0, real)).toBe(1);
    expect(yarnFrequency(real.rope / 2 - 1, real)).toBeCloseTo(
      real.theta ** ((-2 * (real.rope / 2 - 1)) / real.rope) / real.factor,
    );
  });
});
