import { describe, expect, it } from "vitest";

import { Ctx, maxAbsDiff, tensor, type Weights } from "../../../kit/ref/tensor";
import * as f from "../facts";
import fixture from "../reference/fixture.json";
import { forward, type Gpt2Config } from "../reference/model";

const real: Gpt2Config = {
  vocab: f.vocab.value,
  positions: f.positions.value,
  hidden: f.hidden.value,
  layers: f.layers.value,
  heads: f.heads.value,
  mlpRatio: f.mlpRatio.value,
  normEps: f.normEps.value,
};

describe("clean GPT-2 reference", () => {
  it("reproduces the upstream implementation's logits and attention at toy width", () => {
    const toy: Gpt2Config = {
      vocab: fixture.config.vocab_size,
      positions: fixture.config.n_positions,
      hidden: fixture.config.n_embd,
      layers: fixture.config.n_layer,
      heads: fixture.config.n_head,
      mlpRatio: f.mlpRatio.value,
      normEps: f.normEps.value,
    };
    const ctx = new Ctx("value", fixture.weights as Weights);
    const logits = forward(ctx, toy, fixture.input_ids);
    expect(maxAbsDiff(logits, tensor(fixture.logits.shape, fixture.logits.data))).toBeLessThan(1e-4);
    const attention = ctx.trace.get("h0.attn.weights");
    expect(attention).toBeDefined();
    expect(maxAbsDiff(attention!, tensor(fixture.attention_layer0.shape, fixture.attention_layer0.data))).toBeLessThan(
      1e-5,
    );
    // Every exported tensor was consumed: the reference has no parameter the clean code lacks.
    expect([...ctx.params.keys()].sort()).toEqual(Object.keys(fixture.weights).sort());
  });

  it("gives the page's parameter total and shapes when run in shape mode at real width", () => {
    const ctx = new Ctx("shape");
    const logits = forward(ctx, real, [[0, 1, 2, 3]]);
    expect(ctx.parameterCount()).toBe(f.parameterTotal.value);
    expect(logits.shape).toEqual([1, 4, f.vocab.value]);
    expect(ctx.shapeOf("h0.attn.qkv")).toEqual([1, 4, f.qkvWidth.value]);
    expect(ctx.shapeOf("h0.attn.q")).toEqual([1, f.heads.value, 4, f.headDim.value]);
    expect(ctx.shapeOf("h0.attn.scores")).toEqual([1, f.heads.value, 4, 4]);
    expect(ctx.shapeOf("h11.mlp.fc")).toEqual([1, 4, f.mlpWidth.value]);
  });
});
