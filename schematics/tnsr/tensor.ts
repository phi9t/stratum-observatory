// A minimal tensor library for clean reference implementations.
//
// The same model code runs in two modes:
//   shape mode  data is null; every op only computes its output shape. Run it at the
//               checkpoint's real widths to get the true shape of every operation and
//               an exact parameter count, with no memory cost.
//   value mode  data is a Float32Array. Run it at toy widths, with weights loaded from
//               a parity fixture or drawn from a seed, to compute real numbers in the
//               browser or in a test.
//
// Value mode can also be differentiated. Every op below states its derivative next to
// its forward computation, as a function that takes g = ∂loss/∂output and adds each
// input's share into that input's gradient. A Tape (tape.ts) records the ops as they
// run and replays those functions backwards; without a tape nothing is recorded.
//
// Nothing here is fast. It is meant to be read, and to be small enough to trust.

import { rng } from "./rng";
import type { Back, Tape } from "./tape";

export { Tape } from "./tape";

/** `tape` is set on tensors whose computation is being recorded for differentiation. */
export type Tensor = Readonly<{ shape: readonly number[]; data: Float32Array | null; tape?: Tape }>;

const size = (shape: readonly number[]): number => shape.reduce((product, dim) => product * dim, 1);

function fail(message: string): never {
  throw new Error(`ref: ${message}`);
}

export function tensor(shape: readonly number[], data: ArrayLike<number> | null): Tensor {
  if (data !== null && data.length !== size(shape)) {
    fail(`data of length ${String(data.length)} does not fill shape [${shape.join(", ")}]`);
  }
  return { shape: [...shape], data: data === null ? null : Float32Array.from(data) };
}

/**
 * Builds the output of an op. `compute` fills in the values and `back` is the op's
 * derivative (see Back in tape.ts). In shape mode neither is called.
 */
function make(
  shape: readonly number[],
  inputs: readonly Tensor[],
  compute: (out: Float32Array) => void,
  back: Back,
): Tensor {
  if (inputs.some((input) => input.data === null)) {
    return { shape: [...shape], data: null };
  }
  const out = new Float32Array(size(shape));
  compute(out);
  return track({ shape: [...shape], data: out }, inputs, back);
}

/** If any input is on a tape, records the op there and puts the output on it too. */
function track(out: Tensor, inputs: readonly Tensor[], back: Back): Tensor {
  const tape = inputs.find((input) => input.tape !== undefined)?.tape;
  if (tape === undefined) {
    return out;
  }
  const tracked = { ...out, tape };
  tape.record(tracked, inputs, back);
  return tracked;
}

const last = (t: Tensor): number => t.shape[t.shape.length - 1];
const d = (t: Tensor): Float32Array => t.data as Float32Array;

// ---- context: parameters and trace -------------------------------------------

export type Weights = Readonly<Record<string, Readonly<{ shape: readonly number[]; data: ArrayLike<number> }>>>;

export class Ctx {
  /** Every parameter the model asked for, by name. Summing their sizes is the parameter count. */
  readonly params = new Map<string, readonly number[]>();
  /** Tensors the model marked, by op id: the shapes (and values) a page shows. */
  readonly trace = new Map<string, Tensor>();
  /** One tensor per parameter name, so every use of a parameter adds to the same gradient. */
  private readonly tensors = new Map<string, Tensor>();
  private readonly random: () => number;

  constructor(
    readonly mode: "shape" | "value",
    private readonly weights: Weights | null = null,
    seed = 0,
    /** Spread of seeded weights. Raise it when a toy run looks uniform (attention near 1/n everywhere). */
    private readonly gain = 1,
    /** Give a tape to differentiate: parameters are then watched by it (value mode only). */
    private readonly tape: Tape | null = null,
  ) {
    this.random = rng(seed);
  }

  /**
   * A parameter tensor. In value mode it comes from the loaded weights when present
   * (and must match the shape), otherwise from the seed, scaled like a small init.
   */
  param(name: string, shape: readonly number[]): Tensor {
    const seen = this.params.get(name);
    if (seen !== undefined && seen.join() !== shape.join()) {
      fail(`parameter ${name} requested with two shapes`);
    }
    this.params.set(name, [...shape]);
    let found = this.tensors.get(name);
    if (found === undefined) {
      found = this.create(name, shape);
      this.tensors.set(name, found);
    }
    return found;
  }

  private create(name: string, shape: readonly number[]): Tensor {
    if (this.mode === "shape") {
      return { shape: [...shape], data: null };
    }
    const watched = (t: Tensor): Tensor => (this.tape === null ? t : this.tape.watch(t));
    const loaded = this.weights?.[name];
    if (loaded !== undefined) {
      if (loaded.shape.join() !== shape.join()) {
        fail(`weights hold ${name} as [${loaded.shape.join(", ")}], model wants [${shape.join(", ")}]`);
      }
      return watched(tensor(shape, loaded.data));
    }
    if (this.weights !== null) {
      fail(`weights have no tensor named ${name}`);
    }
    const scale = this.gain / Math.sqrt(shape[shape.length - 1]);
    return watched(
      tensor(
        shape,
        Array.from({ length: size(shape) }, () => (this.random() * 2 - 1) * scale),
      ),
    );
  }

  /** After `tape.backward(loss)`: ∂loss/∂parameter for every parameter, by name. */
  grads(): Map<string, Tensor> {
    const tape = this.tape ?? fail("grads: this context was made without a tape");
    return new Map([...this.tensors].map(([name, parameter]) => [name, tape.grad(parameter)]));
  }

  /** Records a tensor under an op id and returns it unchanged. */
  mark(id: string, t: Tensor): Tensor {
    this.trace.set(id, t);
    return t;
  }

  parameterCount(): number {
    let total = 0;
    for (const shape of this.params.values()) {
      total += size(shape);
    }
    return total;
  }

  shapeOf(id: string): readonly number[] {
    const found = this.trace.get(id);
    if (found === undefined) {
      fail(`no op marked "${id}"`);
    }
    return found.shape;
  }
}

/** An input tensor: real values in value mode, shape only in shape mode. */
export function input(ctx: Ctx, shape: readonly number[], values: () => ArrayLike<number>): Tensor {
  return ctx.mode === "shape" ? { shape: [...shape], data: null } : tensor(shape, values());
}

// ---- elementwise ---------------------------------------------------------------

/** y = f(x) elementwise. `df(x, y)` is dy/dx, so ∂L/∂x = g · df. */
function map(x: Tensor, f: (value: number) => number, df: (value: number, y: number) => number): Tensor {
  return make(
    x.shape,
    [x],
    (out) => {
      const a = d(x);
      for (let i = 0; i < out.length; i += 1) out[i] = f(a[i]);
    },
    (g, [gx], y) => {
      const a = d(x);
      for (let i = 0; i < g.length; i += 1) gx[i] += g[i] * df(a[i], y[i]);
    },
  );
}

/**
 * out = f(a, b) elementwise. `b` may have fewer leading axes or axes of size 1 (it is
 * repeated), e.g. a bias of shape [d]. `dfa` and `dfb` are ∂f/∂a and ∂f/∂b. One element
 * of a repeated `b` feeds many outputs, so its gradient is the sum over all of them.
 */
function zip(
  a: Tensor,
  b: Tensor,
  name: string,
  f: (x: number, y: number) => number,
  dfa: (x: number, y: number) => number,
  dfb: (x: number, y: number) => number,
): Tensor {
  const tail = a.shape.slice(a.shape.length - b.shape.length);
  const fits = tail.every((dim, index) => dim === b.shape[index] || b.shape[index] === 1);
  if (!fits) {
    fail(`${name}: [${b.shape.join(", ")}] does not broadcast onto [${a.shape.join(", ")}]`);
  }
  const partner = broadcastIndex(tail, b.shape);
  return make(
    a.shape,
    [a, b],
    (out) => {
      const x = d(a);
      const y = d(b);
      for (let i = 0; i < out.length; i += 1) out[i] = f(x[i], y[partner(i)]);
    },
    (g, [ga, gb]) => {
      const x = d(a);
      const y = d(b);
      for (let i = 0; i < g.length; i += 1) {
        const j = partner(i);
        ga[i] += g[i] * dfa(x[i], y[j]);
        gb[j] += g[i] * dfb(x[i], y[j]);
      }
    },
  );
}

/** For flat index i into a tensor ending in `tail`, the flat index of its partner in b. */
function broadcastIndex(tail: readonly number[], bShape: readonly number[]): (i: number) => number {
  const block = size(tail);
  if (bShape.every((dim, index) => dim === tail[index])) {
    return (i) => i % block;
  }
  // Size-1 axes inside b: index the slow way.
  const strides = bShape.map((_dim, axis) => size(bShape.slice(axis + 1)));
  const aStrides = tail.map((_dim, axis) => size(tail.slice(axis + 1)));
  return (i) => {
    let rest = i % block;
    let j = 0;
    for (let axis = 0; axis < tail.length; axis += 1) {
      const at = Math.floor(rest / aStrides[axis]);
      rest %= aStrides[axis];
      j += (bShape[axis] === 1 ? 0 : at) * strides[axis];
    }
    return j;
  };
}

const one = (): number => 1;
const GELU_RATE = Math.sqrt(2 / Math.PI);
const logistic = (value: number): number => 1 / (1 + Math.exp(-value));

export const add = (a: Tensor, b: Tensor): Tensor => zip(a, b, "add", (x, y) => x + y, one, one);
export const sub = (a: Tensor, b: Tensor): Tensor =>
  zip(
    a,
    b,
    "sub",
    (x, y) => x - y,
    one,
    () => -1,
  );
export const mul = (a: Tensor, b: Tensor): Tensor =>
  zip(
    a,
    b,
    "mul",
    (x, y) => x * y,
    (_x, y) => y,
    (x) => x,
  );
export const scale = (x: Tensor, factor: number): Tensor =>
  map(
    x,
    (value) => value * factor,
    () => factor,
  );
/** σ(x) = 1 / (1 + e^−x); σ′ = σ (1 − σ). */
export const sigmoid = (x: Tensor): Tensor => map(x, logistic, (_value, y) => y * (1 - y));
/** x σ(x); derivative σ + x σ (1 − σ). */
export const silu = (x: Tensor): Tensor =>
  map(
    x,
    (value) => value * logistic(value),
    (value) => logistic(value) * (1 + value * (1 - logistic(value))),
  );
/** log(1 + e^x); derivative σ(x). */
export const softplus = (x: Tensor): Tensor => map(x, (value) => Math.log1p(Math.exp(value)), logistic);
/** √x; derivative 1 / (2 √x). */
export const sqrt = (x: Tensor): Tensor => map(x, Math.sqrt, (_value, y) => 0.5 / y);

/**
 * GELU in the tanh form used by GPT-2 ("gelu_new"): y = ½ x (1 + tanh u) with
 * u = √(2/π) (x + 0.044715 x³). Derivative: ½ (1 + tanh u) + ½ x (1 − tanh² u) · du/dx.
 */
export const geluTanh = (x: Tensor): Tensor =>
  map(
    x,
    (v) => 0.5 * v * (1 + Math.tanh(GELU_RATE * (v + 0.044715 * v ** 3))),
    (v) => {
      const t = Math.tanh(GELU_RATE * (v + 0.044715 * v ** 3));
      return 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * GELU_RATE * (1 + 3 * 0.044715 * v * v);
    },
  );

// ---- reductions and losses -----------------------------------------------------

/** Sum of all elements, as a scalar (shape []). Every element's derivative is 1. */
export function sum(x: Tensor): Tensor {
  return make(
    [],
    [x],
    (out) => {
      let total = 0;
      for (const value of d(x)) total += value;
      out[0] = total;
    },
    (g, [gx]) => {
      for (let i = 0; i < gx.length; i += 1) gx[i] += g[0];
    },
  );
}

/** Mean of all elements, as a scalar (shape []). Every element's derivative is 1/n. */
export function mean(x: Tensor): Tensor {
  const n = size(x.shape);
  return make(
    [],
    [x],
    (out) => {
      let total = 0;
      for (const value of d(x)) total += value;
      out[0] = total / n;
    },
    (g, [gx]) => {
      for (let i = 0; i < gx.length; i += 1) gx[i] += g[0] / n;
    },
  );
}

/**
 * Mean next-token loss: logits [B, S, V] and one target id per position → a scalar,
 * the average over all B·S positions of −log softmax(logits)[target].
 *
 * Computed as logsumexp(z) − z[target], with the row's maximum taken out of the
 * exponentials so they cannot overflow. Its derivative is the famous one:
 * ∂loss/∂z = (softmax(z) − onehot(target)) / (B·S).
 */
export function crossEntropy(logits: Tensor, targets: readonly (readonly number[])[]): Tensor {
  const [batch, seq, vocab] = logits.shape;
  const wanted = targets.flat();
  if (logits.shape.length !== 3 || targets.length !== batch || wanted.length !== batch * seq) {
    fail(`crossEntropy: logits [${logits.shape.join(", ")}] need targets [${String(batch)}, ${String(seq)}]`);
  }
  if (wanted.some((id) => !Number.isInteger(id) || id < 0 || id >= vocab)) {
    fail(`crossEntropy: a target is outside 0 … ${String(vocab - 1)}`);
  }
  const rows = batch * seq;
  const logSumExp = (z: Float32Array, r: number): number => {
    let max = -Infinity;
    for (let i = 0; i < vocab; i += 1) max = Math.max(max, z[r * vocab + i]);
    let total = 0;
    for (let i = 0; i < vocab; i += 1) total += Math.exp(z[r * vocab + i] - max);
    return max + Math.log(total);
  };
  return make(
    [],
    [logits],
    (out) => {
      const z = d(logits);
      let total = 0;
      for (let r = 0; r < rows; r += 1) total += logSumExp(z, r) - z[r * vocab + wanted[r]];
      out[0] = total / rows;
    },
    (g, [gz]) => {
      const z = d(logits);
      for (let r = 0; r < rows; r += 1) {
        const norm = logSumExp(z, r);
        for (let i = 0; i < vocab; i += 1) {
          const probability = Math.exp(z[r * vocab + i] - norm);
          gz[r * vocab + i] += (g[0] * (probability - (i === wanted[r] ? 1 : 0))) / rows;
        }
      }
    },
  );
}

// ---- linear algebra ------------------------------------------------------------

/**
 * y = x @ W (+ b) over the last axis. `layout` says how W is stored:
 * "out_in" is torch.nn.Linear ([out, in]); "in_out" is GPT-2's Conv1D ([in, out]).
 *
 * With y[o] = Σ_i x[i] W[i→o] + b[o] per row:
 * ∂L/∂x[i] = Σ_o g[o] W[i→o], ∂L/∂W[i→o] = Σ_rows x[i] g[o], ∂L/∂b[o] = Σ_rows g[o].
 */
export function affine(x: Tensor, w: Tensor, b: Tensor | null, layout: "out_in" | "in_out"): Tensor {
  const [inW, outW] = layout === "out_in" ? [w.shape[1], w.shape[0]] : [w.shape[0], w.shape[1]];
  if (last(x) !== inW) {
    fail(`affine: input width ${String(last(x))} does not match weight [${w.shape.join(", ")}] (${layout})`);
  }
  /** Where the weight from input i to output o sits in W's flat data. */
  const at = layout === "out_in" ? (i: number, o: number) => o * inW + i : (i: number, o: number) => i * outW + o;
  const shape = [...x.shape.slice(0, -1), outW];
  const rows = size(shape) / outW;
  return make(
    shape,
    b === null ? [x, w] : [x, w, b],
    (out) => {
      const a = d(x);
      const m = d(w);
      for (let r = 0; r < rows; r += 1) {
        for (let o = 0; o < outW; o += 1) {
          let total = b === null ? 0 : d(b)[o];
          for (let i = 0; i < inW; i += 1) total += a[r * inW + i] * m[at(i, o)];
          out[r * outW + o] = total;
        }
      }
    },
    (g, [gx, gw, gb]) => {
      const a = d(x);
      const m = d(w);
      for (let r = 0; r < rows; r += 1) {
        for (let o = 0; o < outW; o += 1) {
          const go = g[r * outW + o];
          if (b !== null) gb[o] += go;
          for (let i = 0; i < inW; i += 1) {
            gx[r * inW + i] += go * m[at(i, o)];
            gw[at(i, o)] += a[r * inW + i] * go;
          }
        }
      }
    },
  );
}

/**
 * Batched matrix product over the last two axes: C = A @ B, [..., n, m] @ [..., m, p].
 * ∂L/∂A = G @ Bᵀ and ∂L/∂B = Aᵀ @ G: each product A[i,k]·B[k,j] sends G[i,j] back to
 * both of its factors, weighted by the other one.
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const [n, m] = a.shape.slice(-2);
  const [m2, p] = b.shape.slice(-2);
  if (m !== m2 || a.shape.slice(0, -2).join() !== b.shape.slice(0, -2).join()) {
    fail(`matmul: [${a.shape.join(", ")}] @ [${b.shape.join(", ")}]`);
  }
  const batches = size(a.shape.slice(0, -2));
  return make(
    [...a.shape.slice(0, -2), n, p],
    [a, b],
    (out) => {
      const x = d(a);
      const y = d(b);
      for (let t = 0; t < batches; t += 1) {
        for (let i = 0; i < n; i += 1) {
          for (let j = 0; j < p; j += 1) {
            let total = 0;
            for (let k = 0; k < m; k += 1) total += x[t * n * m + i * m + k] * y[t * m * p + k * p + j];
            out[t * n * p + i * p + j] = total;
          }
        }
      }
    },
    (g, [ga, gb]) => {
      const x = d(a);
      const y = d(b);
      for (let t = 0; t < batches; t += 1) {
        for (let i = 0; i < n; i += 1) {
          for (let j = 0; j < p; j += 1) {
            const gij = g[t * n * p + i * p + j];
            for (let k = 0; k < m; k += 1) {
              ga[t * n * m + i * m + k] += gij * y[t * m * p + k * p + j];
              gb[t * m * p + k * p + j] += x[t * n * m + i * m + k] * gij;
            }
          }
        }
      }
    },
  );
}

// ---- shape changes -------------------------------------------------------------

/** Same values under a new shape. The gradient passes straight through, element for element. */
export function reshape(x: Tensor, shape: readonly number[]): Tensor {
  if (size(shape) !== size(x.shape)) {
    fail(`reshape: [${x.shape.join(", ")}] to [${shape.join(", ")}]`);
  }
  return track({ shape: [...shape], data: x.data }, [x], (g, [gx]) => {
    for (let i = 0; i < g.length; i += 1) gx[i] += g[i];
  });
}

/**
 * Reorders axes: permute(x, [0, 2, 1, 3]) swaps axes 1 and 2. Each output element is a
 * copy of one input element, so its gradient goes back to exactly that element.
 */
export function permute(x: Tensor, order: readonly number[]): Tensor {
  const shape = order.map((axis) => x.shape[axis]);
  const inStrides = x.shape.map((_dim, axis) => size(x.shape.slice(axis + 1)));
  const outStrides = shape.map((_dim, axis) => size(shape.slice(axis + 1)));
  /** The flat input index that output element i is copied from. */
  const source = (i: number): number => {
    let rest = i;
    let j = 0;
    for (let axis = 0; axis < shape.length; axis += 1) {
      const at = Math.floor(rest / outStrides[axis]);
      rest %= outStrides[axis];
      j += at * inStrides[order[axis]];
    }
    return j;
  };
  return make(
    shape,
    [x],
    (out) => {
      const a = d(x);
      for (let i = 0; i < out.length; i += 1) out[i] = a[source(i)];
    },
    (g, [gx]) => {
      for (let i = 0; i < g.length; i += 1) gx[source(i)] += g[i];
    },
  );
}

/** Cuts the last axis into pieces of the given widths. Each piece's gradient returns to its own columns. */
export function splitLast(x: Tensor, widths: readonly number[]): Tensor[] {
  const width = last(x);
  if (widths.reduce((total, w) => total + w, 0) !== width) {
    fail(`splitLast: widths ${widths.join(" + ")} do not sum to ${String(width)}`);
  }
  const rows = size(x.shape) / width;
  let offset = 0;
  return widths.map((w) => {
    const start = offset;
    offset += w;
    return make(
      [...x.shape.slice(0, -1), w],
      [x],
      (out) => {
        const a = d(x);
        for (let r = 0; r < rows; r += 1) {
          for (let i = 0; i < w; i += 1) out[r * w + i] = a[r * width + start + i];
        }
      },
      (g, [gx]) => {
        for (let r = 0; r < rows; r += 1) {
          for (let i = 0; i < w; i += 1) gx[r * width + start + i] += g[r * w + i];
        }
      },
    );
  });
}

/** Joins tensors along the last axis. Each part receives the gradient of the columns it filled. */
export function concatLast(parts: readonly Tensor[]): Tensor {
  const width = parts.reduce((total, part) => total + last(part), 0);
  const lead = parts[0].shape.slice(0, -1);
  const rows = size(lead);
  return make(
    [...lead, width],
    parts,
    (out) => {
      let offset = 0;
      for (const part of parts) {
        const w = last(part);
        const a = d(part);
        for (let r = 0; r < rows; r += 1) {
          for (let i = 0; i < w; i += 1) out[r * width + offset + i] = a[r * w + i];
        }
        offset += w;
      }
    },
    (g, into) => {
      let offset = 0;
      parts.forEach((part, index) => {
        const w = last(part);
        for (let r = 0; r < rows; r += 1) {
          for (let i = 0; i < w; i += 1) into[index][r * w + i] += g[r * width + offset + i];
        }
        offset += w;
      });
    },
  );
}

// ---- normalisation, attention pieces ------------------------------------------

/**
 * LayerNorm over the last axis: y = x̂ · gain + bias with x̂ = (x − mean) / σ and
 * σ = sqrt(var + eps).
 *
 * ∂L/∂bias = g and ∂L/∂gain = g · x̂, summed over rows. For x, write ĝ = g · gain (the
 * gradient arriving at x̂). Moving one x also moves the row's mean and σ, which removes
 * from ĝ its component along the constant direction and along x̂:
 * ∂L/∂x = (ĝ − mean(ĝ) − x̂ · mean(ĝ · x̂)) / σ.
 */
export function layerNorm(x: Tensor, gain: Tensor, bias: Tensor, eps: number): Tensor {
  const w = last(x);
  const rows = size(x.shape) / w;
  /** Mean and 1/σ of row r. */
  const moments = (a: Float32Array, r: number): [number, number] => {
    let mean = 0;
    for (let i = 0; i < w; i += 1) mean += a[r * w + i];
    mean /= w;
    let variance = 0;
    for (let i = 0; i < w; i += 1) variance += (a[r * w + i] - mean) ** 2;
    return [mean, 1 / Math.sqrt(variance / w + eps)];
  };
  return make(
    x.shape,
    [x, gain, bias],
    (out) => {
      const a = d(x);
      for (let r = 0; r < rows; r += 1) {
        const [mean, inv] = moments(a, r);
        for (let i = 0; i < w; i += 1) out[r * w + i] = (a[r * w + i] - mean) * inv * d(gain)[i] + d(bias)[i];
      }
    },
    (g, [gx, gGain, gBias]) => {
      const a = d(x);
      const gains = d(gain);
      for (let r = 0; r < rows; r += 1) {
        const [mean, inv] = moments(a, r);
        let meanG = 0; // mean(ĝ)
        let meanGX = 0; // mean(ĝ · x̂)
        for (let i = 0; i < w; i += 1) {
          const normed = (a[r * w + i] - mean) * inv;
          gBias[i] += g[r * w + i];
          gGain[i] += g[r * w + i] * normed;
          meanG += (g[r * w + i] * gains[i]) / w;
          meanGX += (g[r * w + i] * gains[i] * normed) / w;
        }
        for (let i = 0; i < w; i += 1) {
          const normed = (a[r * w + i] - mean) * inv;
          gx[r * w + i] += (g[r * w + i] * gains[i] - meanG - normed * meanGX) * inv;
        }
      }
    },
  );
}

/**
 * RMSNorm over the last axis: y = x / rms · gain with rms = sqrt(mean(x²) + eps).
 *
 * ∂L/∂gain = g · x / rms, summed over rows. For x, with ĝ = g · gain, moving one x also
 * moves the row's rms: ∂L/∂x = ĝ / rms − x · mean(ĝ · x) / rms³.
 */
export function rmsNorm(x: Tensor, gain: Tensor, eps: number): Tensor {
  const w = last(x);
  const rows = size(x.shape) / w;
  /** 1/rms of row r. */
  const inverseRms = (a: Float32Array, r: number): number => {
    let square = 0;
    for (let i = 0; i < w; i += 1) square += a[r * w + i] ** 2;
    return 1 / Math.sqrt(square / w + eps);
  };
  return make(
    x.shape,
    [x, gain],
    (out) => {
      const a = d(x);
      for (let r = 0; r < rows; r += 1) {
        const inv = inverseRms(a, r);
        for (let i = 0; i < w; i += 1) out[r * w + i] = a[r * w + i] * inv * d(gain)[i];
      }
    },
    (g, [gx, gGain]) => {
      const a = d(x);
      const gains = d(gain);
      for (let r = 0; r < rows; r += 1) {
        const inv = inverseRms(a, r);
        let meanGX = 0; // mean(ĝ · x)
        for (let i = 0; i < w; i += 1) {
          gGain[i] += g[r * w + i] * a[r * w + i] * inv;
          meanGX += (g[r * w + i] * gains[i] * a[r * w + i]) / w;
        }
        for (let i = 0; i < w; i += 1) {
          gx[r * w + i] += g[r * w + i] * gains[i] * inv - a[r * w + i] * meanGX * inv ** 3;
        }
      }
    },
  );
}

/**
 * Softmax over the last axis: y[i] = e^x[i] / Σ_j e^x[j].
 *
 * ∂y[i]/∂x[j] = y[i] (δ_ij − y[j]), so ∂L/∂x[j] = y[j] (g[j] − Σ_i g[i] y[i]). An entry
 * masked to −infinity has y = 0 and therefore gets no gradient.
 */
export function softmax(x: Tensor): Tensor {
  const w = last(x);
  const rows = size(x.shape) / w;
  return make(
    x.shape,
    [x],
    (out) => {
      const a = d(x);
      for (let r = 0; r < rows; r += 1) {
        let max = -Infinity;
        for (let i = 0; i < w; i += 1) max = Math.max(max, a[r * w + i]);
        let total = 0;
        for (let i = 0; i < w; i += 1) {
          out[r * w + i] = Math.exp(a[r * w + i] - max);
          total += out[r * w + i];
        }
        for (let i = 0; i < w; i += 1) out[r * w + i] /= total;
      }
    },
    (g, [gx], y) => {
      for (let r = 0; r < rows; r += 1) {
        let dot = 0; // Σ_i g[i] y[i]
        for (let i = 0; i < w; i += 1) dot += g[r * w + i] * y[r * w + i];
        for (let i = 0; i < w; i += 1) gx[r * w + i] += y[r * w + i] * (g[r * w + i] - dot);
      }
    },
  );
}

/**
 * Causal mask on scores [..., S, T]: query i may read keys 0 … (T − S) + i.
 * Everything later is set to −infinity. T > S when earlier keys come from a cache.
 * Visible scores pass their gradient through; masked ones were overwritten and get none.
 */
export function causalMask(scores: Tensor): Tensor {
  const [s, t] = scores.shape.slice(-2);
  /** Calls `visit` with the flat index of every masked score. */
  const eachMasked = (length: number, visit: (index: number) => void): void => {
    for (let base = 0; base < length; base += s * t) {
      for (let i = 0; i < s; i += 1) {
        for (let j = t - s + i + 1; j < t; j += 1) visit(base + i * t + j);
      }
    }
  };
  return make(
    scores.shape,
    [scores],
    (out) => {
      out.set(d(scores));
      eachMasked(out.length, (index) => {
        out[index] = -Infinity;
      });
    },
    (g, [gScores]) => {
      const visible = Float32Array.from(g);
      eachMasked(g.length, (index) => {
        visible[index] = 0;
      });
      for (let i = 0; i < g.length; i += 1) gScores[i] += visible[i];
    },
  );
}

/**
 * Rows of `table` selected by integer ids: ids [B, S] → [B, S, d]. Each output row's
 * gradient is added into the table row it was copied from (a row used twice gets both).
 */
export function embedding(table: Tensor, ids: readonly (readonly number[])[]): Tensor {
  const [rows, w] = table.shape;
  const shape = [ids.length, ids[0].length, w];
  const flat = ids.flat();
  return make(
    shape,
    [table],
    (out) => {
      flat.forEach((id, at) => {
        if (id < 0 || id >= rows) fail(`embedding: id ${String(id)} outside 0 … ${String(rows - 1)}`);
        out.set(d(table).subarray(id * w, id * w + w), at * w);
      });
    },
    (g, [gTable]) => {
      flat.forEach((id, at) => {
        for (let i = 0; i < w; i += 1) gTable[id * w + i] += g[at * w + i];
      });
    },
  );
}

/**
 * Largest k entries of the last axis, as values and indices, sorted descending.
 * `values` are copies of the chosen entries, so their gradient returns to those entries;
 * `indices` are integers and carry no gradient.
 */
export function topk(x: Tensor, k: number): { values: Tensor; indices: Tensor } {
  const w = last(x);
  const shape = [...x.shape.slice(0, -1), k];
  if (x.data === null) {
    return { values: { shape, data: null }, indices: { shape, data: null } };
  }
  const values = new Float32Array(size(shape));
  const indices = new Float32Array(size(shape));
  const a = d(x);
  const rows = a.length / w;
  for (let r = 0; r < rows; r += 1) {
    const order = Array.from({ length: w }, (_u, i) => i).sort((p, q) => a[r * w + q] - a[r * w + p] || p - q);
    for (let i = 0; i < k; i += 1) {
      indices[r * k + i] = order[i];
      values[r * k + i] = a[r * w + order[i]];
    }
  }
  const back: Back = (g, [gx]) => {
    for (let r = 0; r < rows; r += 1) {
      for (let i = 0; i < k; i += 1) gx[r * w + indices[r * k + i]] += g[r * k + i];
    }
  };
  return { values: track({ shape, data: values }, [x], back), indices: { shape, data: indices } };
}

/**
 * Picks entries of the last axis of x at the given indices (same leading shape).
 * Each picked entry's gradient returns to where it was picked from.
 */
export function gatherLast(x: Tensor, indices: Tensor): Tensor {
  const w = last(x);
  const k = last(indices);
  return make(
    indices.shape,
    [x, indices],
    (out) => {
      for (let r = 0; r < out.length / k; r += 1) {
        for (let i = 0; i < k; i += 1) out[r * k + i] = d(x)[r * w + d(indices)[r * k + i]];
      }
    },
    (g, [gx]) => {
      for (let r = 0; r < g.length / k; r += 1) {
        for (let i = 0; i < k; i += 1) gx[r * w + d(indices)[r * k + i]] += g[r * k + i];
      }
    },
  );
}

/** Largest absolute difference between two value tensors of the same shape. */
export function maxAbsDiff(a: Tensor, b: Tensor): number {
  if (a.shape.join() !== b.shape.join() || a.data === null || b.data === null) {
    fail(`maxAbsDiff: [${a.shape.join(", ")}] vs [${b.shape.join(", ")}]`);
  }
  let worst = 0;
  for (let i = 0; i < a.data.length; i += 1) worst = Math.max(worst, Math.abs(a.data[i] - b.data[i]));
  return worst;
}
