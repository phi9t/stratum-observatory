import { expect, it } from "vitest";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { Ctx, tensor, maxAbsDiff, type Weights } from "../../../kit/ref/tensor";
import { KimiSparseMoeBlock, KimiMLAAttention } from "../reference/model";
import { attnRes, kdaRecurrence, decayGate } from "../reference/ops";
import * as f from "../facts";
import fixture from "../reference/component-fixture.json";
const unpack = (x: { shape: number[]; data: number[] }) => tensor(x.shape, x.data);
it("component fixtures bind to current pinned upstream definitions", () => {
  const root = new URL("../../../.sources-cache/kimi-k3/", import.meta.url);
  for (const [path, sha] of [
    ["reference/modeling_kimi_linear.py", fixture.source_sha256],
    ["kda-math/fla/ops/kda/naive.py", fixture.kda_source_sha256],
    ["kda-gate/fla/ops/kda/gate.py", fixture.gate_source_sha256],
  ]) {
    expect(
      createHash("sha256")
        .update(readFileSync(new URL(path, root)))
        .digest("hex"),
    ).toBe(sha);
  }
});
it("reproduces exact publisher latent MoE component with every parameter consumed", () => {
  const ctx = new Ctx("value", fixture.moe.weights as Weights);
  const actual = KimiSparseMoeBlock(ctx, f.TOY, unpack(fixture.x), "moe");
  expect(maxAbsDiff(actual, unpack(fixture.moe.output))).toBeLessThan(1e-4);
  expect([...ctx.params.keys()].sort()).toEqual(Object.keys(fixture.moe.weights).sort());
  const actualIds = ctx.trace.get("moe.gate.ids")!.data!;
  for (let token = 0; token < fixture.x.shape[1]; token++)
    expect(Array.from(actualIds.slice(token * f.TOY.topK, (token + 1) * f.TOY.topK)).sort()).toEqual(
      fixture.moe.ids.data.slice(token * f.TOY.topK, (token + 1) * f.TOY.topK).sort(),
    );
});
it("reproduces publisher NoPE gated MLA component with every parameter consumed", () => {
  const ctx = new Ctx("value", fixture.mla.weights as Weights);
  const actual = KimiMLAAttention(ctx, f.TOY, unpack(fixture.x), "mla");
  expect(maxAbsDiff(actual, unpack(fixture.mla.output))).toBeLessThan(1e-4);
  expect([...ctx.params.keys()].sort()).toEqual(Object.keys(fixture.mla.weights).sort());
});
it("reproduces publisher raw-value depth mixture", () => {
  const bank = Array.from({ length: 2 }, (_, n) =>
    tensor(
      [1, 3, 12],
      Array.from(
        { length: 3 * 12 },
        (_, i) => fixture.depth.bank.data[Math.floor(i / 12) * 2 * 12 + n * 12 + (i % 12)],
      ),
    ),
  );
  const actual = attnRes(unpack(fixture.x), bank, unpack(fixture.depth.gain), unpack(fixture.depth.proj), f.eps.value);
  expect(maxAbsDiff(actual.output, tensor([1, 3, 12], fixture.depth.output.data))).toBeLessThan(1e-4);
});
it("reproduces pinned FLA naive lowerbound gate and recurrent output/state", () => {
  const x = fixture.kda;
  const gate = decayGate(unpack(x.g), unpack(x.a), unpack(x.dt), f.lowerBound.value);
  expect(maxAbsDiff(gate, unpack(x.gate))).toBeLessThan(1e-5);
  const y = kdaRecurrence(unpack(x.q), unpack(x.k), unpack(x.v), gate, unpack(x.beta));
  expect(maxAbsDiff(y.output, unpack(x.output))).toBeLessThan(1e-4);
  expect(maxAbsDiff(y.state, unpack(x.state))).toBeLessThan(1e-4);
});
