// The tape: reverse-mode automatic differentiation for the ops in tensor.ts.
//
// Recording is opt-in. A tensor made by `tape.watch(...)` (or a parameter of a Ctx that
// was given a tape) carries a reference to the tape; every op whose inputs include such
// a tensor appends one step to it and passes the reference on to its output. Model code
// never mentions the tape, and without one nothing is recorded.

import type { Tensor } from "./tensor";

/**
 * The vector-Jacobian product of one op. `g` is ∂loss/∂output; the function ADDS each
 * input's share ∂loss/∂input into `into[k]` (same order as the op's inputs). `y` is the
 * op's output values, for derivatives that are simplest in terms of the output.
 */
export type Back = (g: Float32Array, into: Float32Array[], y: Float32Array) => void;

type Step = Readonly<{ out: Tensor; inputs: readonly Tensor[]; back: Back }>;

export class Tape {
  private readonly steps: Step[] = [];
  private readonly grads = new Map<Tensor, Float32Array>();

  /** A leaf to differentiate with respect to: the same values, attached to this tape. */
  watch(t: Tensor): Tensor {
    return { shape: [...t.shape], data: t.data, tape: this };
  }

  /** Called by the ops: remembers how `out` was computed from `inputs`. */
  record(out: Tensor, inputs: readonly Tensor[], back: Back): void {
    this.steps.push({ out, inputs, back });
  }

  /**
   * Fills in ∂loss/∂t for every tensor t the loss was computed from. The steps were
   * recorded in the order they ran, so walking them backwards visits every tensor after
   * all of its uses: by then its gradient is complete and can be handed to its inputs.
   */
  backward(loss: Tensor): this {
    if (loss.data === null || loss.data.length !== 1) {
      throw new Error(`ref: backward needs a scalar value, got shape [${loss.shape.join(", ")}]`);
    }
    this.grads.clear();
    this.buffer(loss)[0] = 1;
    for (let at = this.steps.length - 1; at >= 0; at -= 1) {
      const { out, inputs, back } = this.steps[at];
      const g = this.grads.get(out);
      if (g === undefined) continue; // this output never reached the loss
      back(
        g,
        inputs.map((input) => this.buffer(input)),
        out.data as Float32Array,
      );
    }
    return this;
  }

  /** ∂loss/∂t after `backward`, by tensor identity. Zero if the loss does not depend on t. */
  grad(t: Tensor): Tensor {
    const length = t.shape.reduce((product, dim) => product * dim, 1);
    return { shape: [...t.shape], data: this.grads.get(t) ?? new Float32Array(length) };
  }

  /** The gradient accumulator of a tensor, created at zero on first use. */
  private buffer(t: Tensor): Float32Array {
    let found = this.grads.get(t);
    if (found === undefined) {
      found = new Float32Array((t.data as Float32Array).length);
      this.grads.set(t, found);
    }
    return found;
  }
}
