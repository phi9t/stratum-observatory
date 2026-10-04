import { expect, it } from "vitest";
import * as ref from "../../../tnsr/tensor";
import { forward, REAL, TOY } from "../reference/model";
import { parameterTotal, indexTypes, indexFreq, indexOffset, layers } from "../facts";
it("counts real decoder weights and traces real widths without allocation", () => {
  const ctx = new ref.Ctx("shape");
  forward(ctx, REAL, [[0, 1]]);
  expect(ctx.parameterCount()).toBe(parameterTotal.value);
  expect(ctx.shapeOf("logits")).toEqual([1, 2, REAL.vocab]);
  expect(ctx.shapeOf("l0.qnorm")).toEqual([1, 2, REAL.qRank]);
});
it("serving schedule equals explicit indexer list for every checkpoint layer", () => {
  for (let i = 0; i < layers.value; i++)
    expect(Math.max(i - indexOffset.value + 1, 0) % indexFreq.value === 0 ? "full" : "shared").toBe(
      indexTypes.value[i],
    );
});
it("rejects a shared layer without a preceding index selection", () => {
  expect(() =>
    forward(new ref.Ctx("shape"), { ...TOY, indexTypes: ["shared", ...TOY.indexTypes.slice(1)] }, [[0]]),
  ).toThrow(/selection/);
});

it("toy run stays finite and every attention row sums to unity", () => {
  const ctx = new ref.Ctx("value", null, 19);
  const y = forward(ctx, TOY, [[3, 7, 2, 11, 5, 9]]);
  expect([...y.data!].every(Number.isFinite)).toBe(true);
  for (let i = 0; i < TOY.mlpTypes.length; i++) {
    const a = ctx.trace.get(`l${i}.attention`)!;
    for (let row = 0; row < a.data!.length / 6; row++)
      expect([...a.data!.slice(row * 6, (row + 1) * 6)].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
  }
});
import fixture from "../reference/fixture.json";
it("matches pinned upstream logits and intermediates with perturbed weights", () => {
  const ctx = new ref.Ctx("value", fixture.weights);
  const biases = Object.fromEntries(Object.entries(fixture.buffers).map(([k, v]) => [k, v.data]));
  const y = forward(ctx, TOY, fixture.input_ids, biases);
  expect(ref.maxAbsDiff(y, ref.tensor(fixture.logits.shape, fixture.logits.data))).toBeLessThan(1e-4);
  for (const [id, t] of Object.entries(fixture.trace)) {
    if (id.endsWith(".indices") || id.endsWith(".route_weights")) continue; // top-k ordering is backend dependent; compared as sets separately.
    expect(ref.maxAbsDiff(ctx.trace.get(id)!, ref.tensor(t.shape, t.data)), id).toBeLessThan(1e-4);
  }
  expect([...ctx.params.keys()].sort()).toEqual(Object.keys(fixture.weights).sort());
  for (let i = 0; i < TOY.mlpTypes.length; i++) {
    const ours = ctx.trace.get(`l${i}.indices`)!.data!,
      theirs = fixture.trace[`l${i}.indices` as keyof typeof fixture.trace].data;
    for (let q = 0; q < fixture.input_ids[0].length; q++) {
      const valid = (a: readonly number[]) => a.filter((k) => k <= q).sort((a, b) => a - b);
      expect(valid([...ours.slice(q * TOY.indexTopK, (q + 1) * TOY.indexTopK)])).toEqual(
        valid(theirs.slice(q * TOY.indexTopK, (q + 1) * TOY.indexTopK)),
      );
    }
    if (TOY.mlpTypes[i] === "sparse") {
      const ours = ctx.trace.get(`l${i}.route_weights`)!.data!;
      const theirs = fixture.trace[`l${i}.route_weights` as keyof typeof fixture.trace].data;
      for (let row = 0; row < fixture.input_ids[0].length; row++) {
        const a = [...ours.slice(row * TOY.topK, (row + 1) * TOY.topK)].sort((a, b) => a - b),
          b = theirs.slice(row * TOY.topK, (row + 1) * TOY.topK).sort((a, b) => a - b);
        a.forEach((v, k) => expect(v).toBeCloseTo(b[k], 5));
      }
    }
  }
});
