import { describe, expect, it } from "vitest";

import { Ctx } from "../../../tnsr/tensor";
import * as f from "../facts";
import { forwardShape } from "../reference/model";

describe("GLM-5.3-Flash shape reference", () => {
  it("enumerates the stored checkpoint tensor inventory", () => {
    const ctx = new Ctx("shape");
    forwardShape(ctx, undefined, f.demo.batch, f.demo.tokens);
    expect(ctx.parameterCount()).toBe(f.parameterTotal.value);
    expect(ctx.shapeOf("token_embeddings")).toEqual([f.demo.batch, f.demo.tokens, f.hidden.value]);
    expect(ctx.shapeOf("mhc_streams")).toEqual([f.demo.batch, f.demo.tokens, f.hcMult.value, f.hidden.value]);
    expect(ctx.shapeOf("logits")).toEqual([f.demo.batch, f.demo.tokens, f.vocab.value]);
  });

  it("marks sparse layer score and top-k shapes", () => {
    const ctx = new Ctx("shape");
    forwardShape(ctx, undefined, f.demo.batch, f.demo.tokens);
    for (const layer of f.sparseLayers.value) {
      expect(ctx.shapeOf(`layer.${layer}.index_scores`)).toEqual([
        f.demo.batch,
        f.demo.tokens,
        f.indexHeads.value,
        Math.ceil(f.demo.tokens / f.indexKpool.value),
      ]);
      expect(ctx.shapeOf(`layer.${layer}.topk`)).toEqual([f.demo.batch, f.demo.tokens, f.indexOutputWidth.value]);
    }
  });
});
