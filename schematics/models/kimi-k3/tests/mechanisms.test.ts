import { expect, it } from "vitest";
import { tensor, Ctx } from "../../../kit/ref/tensor";
import { route, attnRes, kdaRecurrence, gatedNorm } from "../reference/ops";
import { forward, realConfig } from "../reference/model";
import * as f from "../facts";
it("bias selects but never becomes a mixture weight", () => {
  const r = route(tensor([1, 3], [0, 1, -2]), tensor([3], [10, 0, 0]), 2);
  expect(Array.from(r.ids.data!)).toEqual([0, 1]);
  const s0 = 0.5,
    s1 = 1 / (1 + Math.exp(-1));
  expect(r.weights.data![0]).toBeCloseTo(s0 / (s0 + s1), 6);
});
it("depth softmax combines raw candidates independently for each token", () => {
  const x = tensor([1, 1, 2], [3, 4]);
  const bank = [tensor([1, 1, 2], [1, 2])];
  const y = attnRes(x, bank, tensor([2], [1, 1]), tensor([1, 2], [0, 0]), 1e-5);
  expect(Array.from(y.output.data!)).toEqual([2, 3]);
  expect(Array.from(y.weights.data!)).toEqual([0.5, 0.5]);
});
it("delta update reads its newly written state and cannot read future tokens", () => {
  const q = tensor([1, 2, 1, 1], [1, 1]);
  const k = tensor([1, 2, 1, 1], [1, 1]);
  const v = tensor([1, 2, 1, 1], [2, 9]);
  const g = tensor([1, 2, 1, 1], [0, 0]);
  const b = tensor([1, 2, 1], [1, 1]);
  const y = kdaRecurrence(q, k, v, g, b);
  expect(Array.from(y.output.data!)).toEqual([2, 9]);
});
it("full-size shape execution counts every constructed parameter", () => {
  const ctx = new Ctx("shape");
  const logits = forward(ctx, realConfig, [[0, 1]]);
  expect(ctx.parameterCount()).toBe(f.parameterTotal.value);
  expect(logits.shape).toEqual([1, 2, f.vocab.value]);
  expect(ctx.shapeOf("h0.kda.q")).toEqual([1, 2, f.heads.value, f.dim.value]);
  expect(ctx.shapeOf("h3.mla.q")).toEqual([1, f.heads.value, 2, f.qDim.value]);
});
it("layer identity comes from explicit lists, including final adjacent full layers", () => {
  expect(f.kindOf(0)).toBe("kda-dense");
  expect(f.kindOf(91)).toBe("mla-moe");
  expect(f.kindOf(92)).toBe("mla-moe");
  expect(f.fullCount.value + f.kdaCount.value).toBe(f.layers.value);
});

it("negative KDA output-gate logits attenuate without flipping the recurrence sign", () => {
  const actual = gatedNorm(tensor([1, 1, 1, 2], [3, 4]), tensor([1, 1, 1, 2], [-2, 2]), tensor([2], [1, 1]), 1e-5);
  const rms = Math.sqrt((9 + 16) / 2 + 1e-5);
  expect(actual.data![0]).toBeCloseTo(3 / rms / (1 + Math.exp(2)), 6);
  expect(actual.data![1]).toBeCloseTo(4 / rms / (1 + Math.exp(-2)), 6);
  expect(actual.data![0]).toBeGreaterThan(0);
});

it("logical text, vision and documented serialized rate padding reconcile Hub count", () => {
  expect(f.parameterTotal.value + f.visionParams.value + f.serializationPadding.value).toBe(f.hubTotal);
});
