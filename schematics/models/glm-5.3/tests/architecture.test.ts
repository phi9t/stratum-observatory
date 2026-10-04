import { expect, it } from "vitest";
import { kvGraph, indexQGraph, expertGraph, moeGraph, attendGraph, indexScoreGraph } from "../architecture";
import { GlmMoeDsaTopkRouter, TOY } from "../reference/model";
import { Ctx, tensor } from "../../../kit/ref/tensor";
it("preserves independent projection and shared-expert dependencies", () => {
  expect(kvGraph().edges).toContainEqual({ from: "kvsplit", to: "krope" });
  expect(kvGraph().edges).not.toContainEqual({ from: "kvnorm", to: "krope" });
  expect(indexQGraph().edges).not.toContainEqual({ from: "indexqsplit", to: "wk" });
  expect(expertGraph(0).edges).not.toContainEqual({ from: "silu", to: "up_proj" });
  expect(moeGraph().edges).toContainEqual({ from: "input", to: "sharedffn" });
  expect(moeGraph().edges).not.toContainEqual({ from: "mixture", to: "sharedffn" });
});
it("uses correction for expert choice while weighting uncorrected sigmoid scores", () => {
  const weight = Array(TOY.experts * TOY.hidden).fill(0);
  [0, 1, 2, 3].forEach((v, i) => (weight[i * TOY.hidden] = v));
  const ctx = new Ctx("value", { "gate.weight": { shape: [TOY.experts, TOY.hidden], data: weight } });
  const x = tensor([1, 1, TOY.hidden], [1, ...Array(TOY.hidden - 1).fill(0)]);
  const { indices, weights } = GlmMoeDsaTopkRouter(ctx, TOY, x, "gate", "test", [10, 10, 0, 0]);
  expect([...indices.data!].sort()).toEqual([0, 1]);
  const raw = [0.5, 1 / (1 + Math.exp(-1))],
    total = raw[0] + raw[1];
  [...indices.data!].forEach((expert, k) => expect(weights.data![k]).toBeCloseTo((raw[expert] / total) * TOY.scale, 6));
});

it("shows packed routed-expert parameters rather than standalone MLP weights", () => {
  const sparse = expertGraph(3).ops;
  expect(sparse[0].detail?.refs).toContainEqual({ source: "reference", symbol: "GlmMoeDsaExperts.forward" });
  expect(sparse[0].detail?.params?.[0]?.[0]).toBe("experts.gate_up_proj");
  expect(sparse.find((op) => op.id === "down_proj")?.detail?.params?.[0]?.[0]).toBe("experts.down_proj");
});

it("uses retained-key width for sequence-score gauges", () => {
  for (const g of [attendGraph(), indexScoreGraph()]) {
    for (const op of g.ops.filter((op) =>
      ["scores", "mask", "softmax", "indexdot", "relu", "sumheads", "indexcausal"].includes(op.id),
    )) {
      expect(op.width).toBe(2048);
    }
  }
});
