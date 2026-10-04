import { describe, expect, it } from "vitest";
import { route, candidateMask, sinkhorn, rotaryFrequency } from "../reference/ops";
describe("V4.1 rules and forbidden substitutions", () => {
  it("keeps short YaRN wavelengths and stretches long ones", () => {
    expect(rotaryFrequency(0, 64, 160000, 65536, 16, 32, 1)).toBe(1);
    expect(rotaryFrequency(31, 64, 160000, 65536, 16, 32, 1)).toBeCloseTo(160000 ** (-62 / 64) / 16, 12);
    expect(rotaryFrequency(31, 64, 10000, 0, 16, 32, 1)).toBeCloseTo(10000 ** (-62 / 64), 12);
  });
  it("uses correction bias for selection, never for weights", () => {
    const a = route([0, 1, 2], [10, 0, 0], 2, 1.5);
    expect(a.chosen).toEqual([0, 2]);
    const raw = [0, 2].map((x) => Math.sqrt(Math.log1p(Math.exp(x))));
    expect(a.weights[0]).toBeCloseTo((raw[0] / (raw[0] + raw[1])) * 1.5);
  });
  it("pins newest candidate block and excludes unreachable slots", () => {
    expect(candidateMask([10, 9, 1, 0, 100, 100], 4, 1, 2)).toEqual([false, false, true, true, false, false]);
  });
  it("normalizes rows and columns instead of replacing the matrix with identity", () => {
    const m = sinkhorn([1, 2, 3, 4], 2, 20, 1e-6);
    expect(m[1]).toBeGreaterThan(0);
    expect(m[0] + m[1]).toBeCloseTo(1, 4);
    expect(m[0] + m[2]).toBeCloseTo(1, 4);
  });
});
