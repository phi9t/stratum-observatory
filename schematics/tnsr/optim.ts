// Optimizers and one training step for models written on tensor.ts.
//
// Ops never change a tensor in place. The optimizers are the one exception: they
// overwrite the values of a `Params` object, which is then handed to a fresh Ctx for
// the next forward pass.

import { crossEntropy, Ctx, Tape, type Tensor, type Weights } from "./tensor";

/** Trainable weights by parameter name. Assignable to `Weights`, so `new Ctx("value", params)` loads it. */
export type Params = Record<string, { shape: readonly number[]; data: Float32Array }>;

/** A mutable Float32 copy of loaded weights (for example a parity fixture's). */
export function trainable(weights: Weights): Params {
  return Object.fromEntries(
    Object.entries(weights).map(([name, { shape, data }]) => [
      name,
      { shape: [...shape], data: Float32Array.from(data) },
    ]),
  );
}

export interface Optimizer {
  /** Updates `params` in place from the gradients of one loss. */
  step(params: Params, grads: ReadonlyMap<string, Tensor>): void;
}

function gradientOf(grads: ReadonlyMap<string, Tensor>, name: string, length: number): Float32Array {
  const g = grads.get(name)?.data;
  if (g === undefined || g === null || g.length !== length) {
    throw new Error(`ref: no gradient of the right size for parameter ${name}`);
  }
  return g;
}

/** Plain gradient descent: p ← p − lr · g. */
export function sgd(params: Params, grads: ReadonlyMap<string, Tensor>, { lr }: { lr: number }): void {
  for (const [name, { data }] of Object.entries(params)) {
    const g = gradientOf(grads, name, data.length);
    for (let i = 0; i < data.length; i += 1) data[i] -= lr * g[i];
  }
}

export type AdamWOptions = Readonly<{
  lr: number;
  betas?: readonly [number, number];
  eps?: number;
  weightDecay?: number;
}>;

/**
 * AdamW with the semantics (and defaults) of torch.optim.AdamW. Per element, at step t:
 *
 *   p ← p · (1 − lr · weightDecay)           decay is applied to the weight directly,
 *                                            not mixed into the gradient ("decoupled")
 *   m ← β₁ m + (1 − β₁) g                    running mean of the gradient
 *   v ← β₂ v + (1 − β₂) g²                   running mean of its square
 *   m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)    undo the bias of starting both at zero
 *   p ← p − lr · m̂ / (√v̂ + eps)             eps is added outside the square root
 */
export class AdamW implements Optimizer {
  private readonly lr: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly eps: number;
  private readonly weightDecay: number;
  private readonly moments = new Map<string, { m: Float32Array; v: Float32Array }>();
  private steps = 0;

  constructor({ lr, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.01 }: AdamWOptions) {
    this.lr = lr;
    [this.beta1, this.beta2] = betas;
    this.eps = eps;
    this.weightDecay = weightDecay;
  }

  step(params: Params, grads: ReadonlyMap<string, Tensor>): void {
    this.steps += 1;
    const correction1 = 1 - this.beta1 ** this.steps;
    const correction2 = 1 - this.beta2 ** this.steps;
    for (const [name, { data }] of Object.entries(params)) {
      const g = gradientOf(grads, name, data.length);
      let state = this.moments.get(name);
      if (state === undefined) {
        state = { m: new Float32Array(data.length), v: new Float32Array(data.length) };
        this.moments.set(name, state);
      }
      const { m, v } = state;
      for (let i = 0; i < data.length; i += 1) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] * g[i];
        const mHat = m[i] / correction1;
        const vHat = v[i] / correction2;
        data[i] = data[i] * (1 - this.lr * this.weightDecay) - (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}

export type Batch = Readonly<{ inputs: readonly (readonly number[])[]; targets: readonly (readonly number[])[] }>;

/**
 * One training step of an autoregressive decoder: forward pass, mean next-token
 * cross-entropy, backward pass, optimizer update. `weights` is updated in place.
 * Returns the loss before the update and the L2 norm of every parameter's gradient.
 */
export function trainStep({
  forward,
  weights,
  batch,
  optimizer,
}: {
  /** The model: token ids [B, S] to logits [B, S, vocab], reading parameters from ctx. */
  forward: (ctx: Ctx, inputs: readonly (readonly number[])[]) => Tensor;
  weights: Params;
  batch: Batch;
  optimizer: Optimizer;
}): { loss: number; gradNorms: Record<string, number> } {
  const tape = new Tape();
  const ctx = new Ctx("value", weights, 0, 1, tape);
  const loss = crossEntropy(forward(ctx, batch.inputs), batch.targets);
  tape.backward(loss);
  const grads = ctx.grads();
  optimizer.step(weights, grads);

  const gradNorms: Record<string, number> = {};
  for (const [name, g] of grads) {
    let squares = 0;
    for (const value of g.data as Float32Array) squares += value * value;
    gradNorms[name] = Math.sqrt(squares);
  }
  return { loss: (loss.data as Float32Array)[0], gradNorms };
}
