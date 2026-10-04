import { it, expect } from "vitest";
import { tensor } from "../../../tnsr/tensor";
import { repeatHeads, rotateYarn } from "../reference/ops";
it("broadcasts a shared key across heads without mixing batches", () => {
  expect(Array.from(repeatHeads(tensor([2, 1, 1, 2], [1, 2, 3, 4]), 2).data!)).toEqual([1, 2, 1, 2, 3, 4, 3, 4]);
  expect(repeatHeads(tensor([2, 1, 3, 4], null), 2)).toEqual({ shape: [2, 2, 3, 4], data: null });
  expect(() => repeatHeads(tensor([1, 2, 1, 1], [1, 2]), 2)).toThrow();
});
it("permutes adjacent rotary pairs before applying half-layout rotations", () => {
  const out = rotateYarn(tensor([1, 1, 2, 4], [1, 2, 3, 4, 1, 2, 3, 4]), [Math.PI / 2, 0], 1);
  expect(Array.from(out.data!).slice(0, 4)).toEqual([1, 3, 2, 4]);
  expect(Array.from(out.data!).slice(4)).toEqual([-2, 3, 1, 4]);
  expect(rotateYarn(tensor([1, 2, 3, 4], null), [1, 2], 1).data).toBeNull();
  expect(() => rotateYarn(tensor([1, 1, 1, 4], null), [1], 1)).toThrow();
});
