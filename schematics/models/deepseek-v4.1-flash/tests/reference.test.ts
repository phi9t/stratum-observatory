import { describe, expect, it } from "vitest";
import { Ctx } from "../../../tnsr/tensor";
import { forward, REAL, TOY, TOY_IDS, toyHashes } from "../reference/model";
import { parameterTotal } from "../facts";
describe("clean text backbone", () => {
  it("counts full logical parameters and tracks real shapes without allocating weights", () => {
    const ctx = new Ctx("shape");
    const y = forward(ctx, REAL, [[0, 1, 2, 3]]);
    expect(ctx.parameterCount()).toBe(parameterTotal.value);
    expect(y.shape).toEqual([1, 4, REAL.vocab]);
    expect(ctx.shapeOf("layers.2.attn.q")).toEqual([1, 4, REAL.h, REAL.hd]);
    expect(ctx.shapeOf("layers.2.attn.compressor.latent")).toEqual([1, 2, REAL.hd]);
    expect(ctx.shapeOf("layers.20.attn.compressor.latent")).toEqual([1, 4, REAL.hd]);
    expect(ctx.params.has("layers.3.attn.compressor.wkv.weight")).toBe(false);
    expect(ctx.params.has("layers.24.attn.indexer.wk.weight")).toBe(false);
  });
  it("runs every layer ownership kind with finite logits and a routed stream", () => {
    const ctx = new Ctx("value", null, 3, 2);
    const y = forward(ctx, TOY, TOY_IDS, toyHashes(TOY, TOY_IDS));
    expect(Array.from(y.data!).every(Number.isFinite)).toBe(true);
    expect(ctx.shapeOf("layers.14.engram.output")).toEqual([1, TOY_IDS[0].length, TOY.hc, TOY.d]);
    const weights = ctx.trace.get("layers.2.ffn.gate.weights")!.data!;
    expect(weights[0] + weights[1]).toBeCloseTo(TOY.scale);
    const p = ctx.trace.get("layers.2.attn.probabilities")!;
    const columns = p.shape.at(-1)!;
    for (let t = 0; t < TOY_IDS[0].length; t++) {
      const row = Array.from(p.data!.slice(t * TOY.h * columns, (t * TOY.h + 1) * columns));
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
      for (let k = 0; k < TOY_IDS[0].length; k++) if (k > t || k < t - TOY.window + 1) expect(row[k]).toBe(0);
      // Completed-group causality, independently of fine top-k selection.
      for (let k = Math.floor((t + 1) / TOY.ratios[2]); k < columns - TOY_IDS[0].length - 1; k++)
        expect(row[TOY_IDS[0].length + k]).toBe(0);
      expect(row.at(-1)).toBeGreaterThan(0);
    }
  });
  it("refuses to fabricate tokenizer hashes", () => {
    expect(() => forward(new Ctx("value"), TOY, TOY_IDS)).toThrow("explicit Engram hash IDs");
  });
  it.skip("upstream numerical parity: publisher requires TileLang/CUDA; local PyTorch is CPU-only", () => {});
});
