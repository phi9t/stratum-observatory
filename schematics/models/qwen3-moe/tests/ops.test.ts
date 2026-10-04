import { describe, expect, it } from "vitest";
import { tensor } from "../../../tnsr/tensor";
import { repeatKV, rotary } from "../reference/ops";
describe("head-axis contracts", () => {
  it("rejects fractional KV repetition rather than inventing a head axis", () => {
    expect(() => repeatKV(tensor([1, 2, 3, 4], null), 1.5)).toThrow(/integer/);
  });
  it("rotates split halves, not adjacent channel pairs", () => {
    const result = rotary(tensor([1, 1, 2, 4], [1, 2, 3, 4, 1, 2, 3, 4]), 1000000);
    expect(result.data![4]).toBeCloseTo(Math.cos(1) - 3 * Math.sin(1), 6);
    expect(result.data![6]).toBeCloseTo(3 * Math.cos(1) + Math.sin(1), 6);
  });
});
