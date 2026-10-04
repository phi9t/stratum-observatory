import { describe, it, expect } from "vitest";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import * as ref from "../../../tnsr/tensor";
import * as f from "../facts";
import { forward, REAL, TOY, fromConfig, MoEGate } from "../reference/model";
import fixture from "../reference/fixture.json";
const asTensor = (t: { shape: number[]; data: number[] }) => ref.tensor(t.shape, t.data);
describe("clean pinned DeepSeek-V3 reference", () => {
  it("reproduces upstream logits, latents, expanded KV, attention, and grouped routes", () => {
    const ctx = new ref.Ctx("value", fixture.weights);
    const logits = forward(ctx, fromConfig(fixture.config), fixture.input_ids);
    expect(ref.maxAbsDiff(logits, asTensor(fixture.logits))).toBeLessThan(1e-4);
    for (const [id, t] of Object.entries(fixture.trace)) {
      expect(ref.maxAbsDiff(ctx.trace.get(id)!, asTensor(t)), id).toBeLessThan(1e-4);
    }
    expect([...ctx.params.keys()].sort()).toEqual(Object.keys(fixture.weights).sort());
    const source = readFileSync(".sources-cache/deepseek-v3/reference/modeling_deepseek.py");
    expect(createHash("sha256").update(source).digest("hex")).toBe(fixture.source_sha256);
  });
  it("counts the full main forward at real widths and preserves diagram shapes", () => {
    const ctx = new ref.Ctx("shape");
    forward(ctx, REAL, [[1, 2]]);
    expect(ctx.parameterCount()).toBe(f.parameterTotal.value);
    expect(ctx.shapeOf("h0.kn")).toEqual([1, 2, f.kvRank.value]);
    expect(ctx.shapeOf("h0.kb")).toEqual([1, 2, f.kvWidth.value]);
    expect(ctx.shapeOf("h0.joink")).toEqual([1, f.heads.value, 2, f.qDim.value]);
    expect(ctx.shapeOf("h0.kvsplit")).toEqual([1, f.heads.value, 2, f.valueDim.value]);
    expect(ctx.shapeOf("logits")).toEqual([1, 2, f.vocab.value]);
  });
  it("masks excluded groups even when all corrected scores are negative", () => {
    const weights = {
      "gate.weight": { shape: [TOY.experts, TOY.hidden], data: Array(TOY.experts * TOY.hidden).fill(0) },
      "gate.e_score_correction_bias": { shape: [TOY.experts], data: [-4, -4, -3, -3, -2, -2, -1, -1] },
    };
    const ctx = new ref.Ctx("value", weights);
    const routes = MoEGate(ctx, "gate", "test", ref.tensor([1, 1, TOY.hidden], Array(TOY.hidden).fill(1)), TOY);
    expect(routes[0].chosenGroups).toEqual([2, 3]);
    expect(routes[0].chosen).toEqual([6, 7]);
    expect(routes[0].weights).toEqual([TOY.routeScale / 2, TOY.routeScale / 2]);
    expect(routes[0].weights.reduce((a, b) => a + b, 0)).toBeCloseTo(TOY.routeScale);
  });
});
