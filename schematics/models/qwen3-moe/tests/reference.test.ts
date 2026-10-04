import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import sources from "../sources.json";
import { describe, expect, it } from "vitest";
import { Ctx, maxAbsDiff, tensor, type Weights } from "../../../tnsr/tensor";
import * as f from "../facts";
import { forward, type QwenConfig } from "../reference/model";
import fixture from "../reference/fixture.json";
const real: QwenConfig = Object.fromEntries(
  [
    "vocab",
    "hidden",
    "layers",
    "heads",
    "kvHeads",
    "headDim",
    "experts",
    "topK",
    "expertWidth",
    "normEps",
    "theta",
  ].map((k) => [k, f.facts[k].value]),
) as QwenConfig;
describe("clean Qwen3 MoE reference", () => {
  it("binds numerical fixture to the pinned implementation bytes", () => {
    for (const [id, binding] of Object.entries(fixture.bindings)) {
      const source = sources.sources.find((s) => s.id === id)!;
      const bytes = readFileSync(`.sources-cache/qwen3-moe/${id}/${source.path}`);
      expect(binding.revision).toBe(source.revision);
      expect(binding.sha256).toBe(createHash("sha256").update(bytes).digest("hex"));
    }
  });
  it("changing a future token cannot alter earlier logits or routes", () => {
    const a = new Ctx("value", fixture.weights as Weights),
      b = new Ctx("value", fixture.weights as Weights);
    const ids = fixture.input_ids.map((row) => [...row]);
    const seq = ids[0].length;
    ids[0][seq - 1] = (ids[0][seq - 1] + 1) % f.TOY.vocab;
    const la = forward(a, f.TOY, fixture.input_ids).data!,
      lb = forward(b, f.TOY, ids).data!;
    expect(Array.from(la.slice(0, (seq - 1) * f.TOY.vocab))).toEqual(Array.from(lb.slice(0, (seq - 1) * f.TOY.vocab)));
    for (let i = 0; i < f.TOY.layers; i++) {
      const name = `h${i}.moe.selected`;
      expect(Array.from(a.trace.get(name)!.data!.slice(0, (seq - 1) * f.TOY.topK))).toEqual(
        Array.from(b.trace.get(name)!.data!.slice(0, (seq - 1) * f.TOY.topK)),
      );
    }
    expect(Array.from(la)).not.toEqual(Array.from(lb));
  });
  it("matches upstream logits, attention and token routes; consumes every parameter", () => {
    const ctx = new Ctx("value", fixture.weights as Weights);
    const result = forward(ctx, f.TOY, fixture.input_ids);
    expect(maxAbsDiff(result, tensor(fixture.logits.shape, fixture.logits.data))).toBeLessThan(1e-4);
    fixture.attention.forEach((a, i) =>
      expect(maxAbsDiff(ctx.trace.get(`h${i}.attn.weights`)!, tensor(a.shape, a.data))).toBeLessThan(1e-5),
    );
    fixture.routers.forEach((r, i) => {
      expect(maxAbsDiff(ctx.trace.get(`h${i}.moe.logits`)!, tensor(r.logits.shape, r.logits.data))).toBeLessThan(1e-5);
      expect(maxAbsDiff(ctx.trace.get(`h${i}.moe.selected`)!, tensor(r.selected.shape, r.selected.data))).toBe(0);
      expect(maxAbsDiff(ctx.trace.get(`h${i}.moe.weights`)!, tensor(r.weights.shape, r.weights.data))).toBeLessThan(
        1e-5,
      );
    });
    expect([...ctx.params.keys()].sort()).toEqual(Object.keys(fixture.weights).sort());
  });
  it("counts real checkpoint parameters and asymmetric query / KV shapes", () => {
    const ctx = new Ctx("shape");
    forward(ctx, real, [[0, 1, 2]]);
    expect(ctx.parameterCount()).toBe(f.parameterTotal.value);
    expect(ctx.shapeOf("h0.attn.q")).toEqual([1, f.heads.value, 3, f.headDim.value]);
    expect(ctx.shapeOf("h0.attn.k")).toEqual([1, f.kvHeads.value, 3, f.headDim.value]);
    expect(ctx.shapeOf("h93.moe.logits")).toEqual([3, f.experts.value]);
    expect(ctx.shapeOf("h0.moe.e0.gather")).toEqual([3, f.hidden.value]);
    expect(ctx.shapeOf("logits")).toEqual([1, 3, f.vocab.value]);
  });
  it("never routes future attention and normalized choices sum to one", () => {
    const ctx = new Ctx("value", null, 3, 4);
    forward(ctx, f.TOY, [[1, 2, 3]]);
    const a = ctx.trace.get("h0.attn.weights")!.data!;
    for (let h = 0; h < f.TOY.heads; h++)
      for (let q = 0; q < 3; q++) for (let k = q + 1; k < 3; k++) expect(a[h * 9 + q * 3 + k]).toBe(0);
    const w = ctx.trace.get("h0.moe.weights")!.data!;
    for (let t = 0; t < 3; t++) expect(w[t * f.TOY.topK] + w[t * f.TOY.topK + 1]).toBeCloseTo(1, 6);
  });
});
