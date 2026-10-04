// Every op's derivative in tensor.ts, checked against central finite differences.
//
// For an op f and a fixed random weighting c, the scalar L(x) = Σ c · f(x) has the
// gradient the tape reports, and also (L(x + h) − L(x − h)) / 2h in each coordinate.
// The two must agree for every element of every input.

import { describe, expect, it } from "vitest";

import { rng } from "../lib/rng";
import * as ref from "./tensor";
import { type Tensor } from "./tensor";

const STEP = 1e-2;
/** Allowed error: 2% of the gradient's size, plus a floor for Float32 rounding in L. */
const tolerance = (size: number): number => 2e-2 * size + 2e-3;

const count = (shape: readonly number[]): number => shape.reduce((product, dim) => product * dim, 1);

/** A tensor of seeded uniform values in [low, high). */
function random(shape: readonly number[], seed: number, low = -1, high = 1): Tensor {
  const uniform = rng(seed);
  return ref.tensor(
    shape,
    Array.from({ length: count(shape) }, () => low + (high - low) * uniform()),
  );
}

/** Checks ∂L/∂input for every input of `f`, where L = Σ c · f(inputs). */
function check(inputs: readonly Tensor[], f: (...inputs: Tensor[]) => Tensor): void {
  const weighting = random(f(...inputs).shape, 99, 0.5, 1.5);
  /** L, summed in double precision from the op's Float32 output. */
  const scalar = (at: readonly Tensor[]): number => {
    const out = f(...at).data!;
    return out.reduce((total, value, index) => total + value * weighting.data![index], 0);
  };

  const tape = new ref.Tape();
  const watched = inputs.map((input) => tape.watch(input));
  tape.backward(ref.sum(ref.mul(f(...watched), weighting)));

  inputs.forEach((input, which) => {
    const analytic = tape.grad(watched[which]).data!;
    for (let index = 0; index < input.data!.length; index += 1) {
      const nudged = (delta: number): Tensor[] => {
        const data = Float32Array.from(input.data!);
        data[index] += delta;
        return inputs.map((other, k) => (k === which ? { shape: input.shape, data } : other));
      };
      const [up, down] = [nudged(STEP), nudged(-STEP)];
      const moved = up[which].data![index] - down[which].data![index]; // what Float32 really moved by
      const numeric = (scalar(up) - scalar(down)) / moved;
      expect(
        Math.abs(analytic[index] - numeric),
        `input ${String(which)}, element ${String(index)}: tape ${String(analytic[index])}, numeric ${String(numeric)}`,
      ).toBeLessThan(tolerance(Math.abs(numeric)));
    }
  });
}

describe("vector-Jacobian products agree with finite differences", () => {
  it("add, sub, mul: same shape", () => {
    const [a, b] = [random([2, 3, 4], 1), random([2, 3, 4], 2)];
    check([a, b], ref.add);
    check([a, b], ref.sub);
    check([a, b], ref.mul);
  });

  it("add, sub, mul: a bias-like operand repeated over leading axes", () => {
    const [a, b] = [random([2, 3, 4], 1), random([4], 2)];
    check([a, b], ref.add);
    check([a, b], ref.sub);
    check([a, b], ref.mul);
  });

  it("add, mul: an operand with a size-1 axis", () => {
    const [a, b] = [random([2, 3, 4], 1), random([3, 1], 2)];
    check([a, b], ref.add);
    check([a, b], ref.mul);
  });

  it("scale, sigmoid, silu, softplus, geluTanh", () => {
    const x = random([3, 5], 3, -3, 3);
    check([x], (t) => ref.scale(t, -1.7));
    check([x], ref.sigmoid);
    check([x], ref.silu);
    check([x], ref.softplus);
    check([x], ref.geluTanh);
  });

  it("sqrt", () => {
    check([random([3, 5], 4, 0.5, 3)], ref.sqrt);
  });

  it("sum, mean", () => {
    const x = random([2, 3, 4], 5);
    check([x], ref.sum);
    check([x], ref.mean);
  });

  it("affine: torch.nn.Linear layout [out, in], with and without bias", () => {
    const [x, w, b] = [random([2, 3, 4], 6), random([5, 4], 7), random([5], 8)];
    check([x, w, b], (xs, ws, bs) => ref.affine(xs, ws, bs, "out_in"));
    check([x, w], (xs, ws) => ref.affine(xs, ws, null, "out_in"));
  });

  it("affine: Conv1D layout [in, out]", () => {
    const [x, w, b] = [random([2, 3, 4], 6), random([4, 5], 7), random([5], 8)];
    check([x, w, b], (xs, ws, bs) => ref.affine(xs, ws, bs, "in_out"));
  });

  it("matmul, batched", () => {
    check([random([2, 3, 4], 9), random([2, 4, 5], 10)], ref.matmul);
  });

  it("reshape, permute", () => {
    const x = random([2, 3, 4], 11);
    check([x], (t) => ref.reshape(t, [4, 6]));
    check([x], (t) => ref.permute(t, [2, 0, 1]));
    check([x], (t) => ref.permute(ref.reshape(t, [2, 3, 2, 2]), [0, 2, 1, 3]));
  });

  it("splitLast, concatLast", () => {
    const x = random([2, 3, 6], 12);
    check([x], (t) => {
      const [left, right] = ref.splitLast(t, [3, 3]);
      return ref.mul(left, right);
    });
    check([x], (t) => ref.splitLast(t, [1, 3, 2])[1]);
    check([random([2, 3], 13), random([2, 1], 14), random([2, 4], 15)], (a, b, c) => ref.concatLast([c, a, b]));
  });

  it("layerNorm: input, gain and bias", () => {
    const [x, gain, bias] = [random([2, 3, 6], 16), random([6], 17, 0.5, 1.5), random([6], 18)];
    check([x, gain, bias], (xs, gs, bs) => ref.layerNorm(xs, gs, bs, 1e-5));
  });

  it("rmsNorm: input and gain", () => {
    const [x, gain] = [random([2, 3, 6], 19), random([6], 20, 0.5, 1.5)];
    check([x, gain], (xs, gs) => ref.rmsNorm(xs, gs, 1e-6));
  });

  it("softmax", () => {
    check([random([2, 3, 5], 21, -2, 2)], ref.softmax);
  });

  it("softmax over causally masked scores, square and with cached keys", () => {
    check([random([2, 2, 4, 4], 22, -2, 2)], (scores) => ref.softmax(ref.causalMask(scores)));
    check([random([2, 3, 5], 23, -2, 2)], (scores) => ref.softmax(ref.causalMask(scores)));
  });

  it("causalMask: masked scores get exactly zero gradient, visible ones pass through", () => {
    // The sum of the masked scores is −infinity, but its derivative is still 1 for every
    // element of the mask's output, which is what the mask's own derivative has to split.
    const tape = new ref.Tape();
    const scores = tape.watch(random([2, 3, 4], 24));
    tape.backward(ref.sum(ref.causalMask(scores)));
    const visible = [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]; // 3 queries × 4 keys, one cached key
    expect([...tape.grad(scores).data!]).toEqual([...visible, ...visible]);
  });

  it("embedding: rows used twice receive both gradients", () => {
    const ids = [
      [0, 2, 2],
      [4, 0, 1],
    ];
    const table = random([5, 3], 25);
    check([table], (t) => ref.embedding(t, ids));
    const tape = new ref.Tape();
    const watched = tape.watch(table);
    tape.backward(ref.sum(ref.embedding(watched, ids)));
    // Row 3 is never looked up; rows 0 and 2 are looked up twice.
    expect([...tape.grad(watched).data!]).toEqual([2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1]);
  });

  it("topk values, and topk indices used to gather from another tensor", () => {
    // Entries far enough apart that a nudge of STEP cannot change which are largest.
    const scores = ref.tensor([2, 5], [0.9, -0.3, 0.2, 1.6, -1.1, -0.8, 0.5, 1.3, -0.1, 0.1]);
    const other = random([2, 5], 26);
    check([scores], (s) => ref.topk(s, 2).values);
    check([scores, other], (s, o) => {
      const { values, indices } = ref.topk(s, 3);
      return ref.mul(ref.softmax(values), ref.gatherLast(o, indices));
    });
    expect(ref.topk(new ref.Tape().watch(scores), 2).indices.tape).toBeUndefined();
  });

  it("crossEntropy", () => {
    const targets = [
      [1, 4, 0],
      [2, 2, 3],
    ];
    check([random([2, 3, 5], 27, -2, 2)], (logits) => ref.crossEntropy(logits, targets));
  });

  it("a composition: one attention-shaped block down to a loss", () => {
    const [x, w, gain, bias] = [random([1, 4, 6], 28), random([6, 6], 29), random([6], 30, 0.5, 1.5), random([6], 31)];
    check([x, w, gain, bias], (xs, ws, gs, bs) => {
      const h = ref.layerNorm(xs, gs, bs, 1e-5);
      const scores = ref.scale(ref.matmul(h, ref.permute(ref.affine(h, ws, null, "in_out"), [0, 2, 1])), 0.5);
      const mixed = ref.matmul(ref.softmax(ref.causalMask(scores)), h);
      return ref.crossEntropy(ref.add(xs, ref.geluTanh(mixed)), [[0, 5, 2, 2]]);
    });
  });
});

describe("crossEntropy", () => {
  it("is the mean of −log softmax at the targets", () => {
    const logits = ref.tensor([1, 2, 3], [1, 2, 3, 0, 0, 0]);
    const expected = (-Math.log(Math.exp(2) / (Math.exp(1) + Math.exp(2) + Math.exp(3))) + Math.log(3)) / 2;
    expect(ref.crossEntropy(logits, [[1, 0]]).data![0]).toBeCloseTo(expected, 6);
  });

  it("stays finite for logits whose exponentials overflow", () => {
    const loss = ref.crossEntropy(ref.tensor([1, 1, 3], [1000, 0, -1000]), [[1]]);
    expect(loss.data![0]).toBeCloseTo(1000, 3);
  });

  it("rejects targets of the wrong shape or outside the vocabulary", () => {
    const logits = random([1, 2, 3], 1);
    expect(() => ref.crossEntropy(logits, [[0]])).toThrow(/crossEntropy/);
    expect(() => ref.crossEntropy(logits, [[0, 3]])).toThrow(/crossEntropy/);
  });
});

describe("the tape", () => {
  it("adds up the gradients of a tensor used more than once", () => {
    const tape = new ref.Tape();
    const x = tape.watch(ref.tensor([3], [1, -2, 3]));
    tape.backward(ref.sum(ref.mul(x, x)));
    expect([...tape.grad(x).data!]).toEqual([2, -4, 6]);
  });

  it("gives parameter gradients by name, one tensor per name", () => {
    const tape = new ref.Tape();
    const ctx = new ref.Ctx("value", { w: { shape: [2], data: [3, 4] } }, 0, 1, tape);
    expect(ctx.param("w", [2])).toBe(ctx.param("w", [2]));
    tape.backward(ref.sum(ref.mul(ctx.param("w", [2]), ctx.param("w", [2]))));
    expect([...ctx.grads().get("w")!.data!]).toEqual([6, 8]);
  });

  it("gives a zero gradient to a tensor the loss does not depend on", () => {
    const tape = new ref.Tape();
    const [x, unused] = [tape.watch(random([2], 1)), tape.watch(random([2], 2))];
    ref.scale(unused, 2);
    tape.backward(ref.sum(x));
    expect([...tape.grad(unused).data!]).toEqual([0, 0]);
  });

  it("only differentiates a scalar", () => {
    const tape = new ref.Tape();
    expect(() => tape.backward(tape.watch(random([2], 1)))).toThrow(/scalar/);
  });

  it("records nothing unless a tensor was put on it", () => {
    const out = ref.softmax(ref.add(random([2, 3], 1), random([3], 2)));
    expect(out.tape).toBeUndefined();
    expect(() => new ref.Ctx("value").grads()).toThrow(/without a tape/);
  });

  it("leaves shape mode alone: no data, no tape, shapes only", () => {
    const ctx = new ref.Ctx("shape", null, 0, 1, new ref.Tape());
    const logits = ref.affine(
      ref.embedding(ctx.param("table", [7, 4]), [[1, 2, 3]]),
      ctx.param("head", [7, 4]),
      null,
      "out_in",
    );
    expect(logits).toEqual({ shape: [1, 3, 7], data: null });
    expect(ref.crossEntropy(logits, [[0, 1, 2]])).toEqual({ shape: [], data: null });
    expect(ref.sum(logits)).toEqual({ shape: [], data: null });
    expect(ref.mean(logits)).toEqual({ shape: [], data: null });
    expect(ref.sub(logits, logits)).toEqual({ shape: [1, 3, 7], data: null });
  });
});
