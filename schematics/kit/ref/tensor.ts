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
// Nothing here is fast. It is meant to be read, and to be small enough to trust.

import { rng } from "../lib/rng";

export type Tensor = Readonly<{ shape: readonly number[]; data: Float32Array | null }>;

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

/** Builds the output of an op: in shape mode `compute` is never called. */
function make(shape: readonly number[], inputs: readonly Tensor[], compute: (out: Float32Array) => void): Tensor {
  if (inputs.some((input) => input.data === null)) {
    return { shape: [...shape], data: null };
  }
  const out = new Float32Array(size(shape));
  compute(out);
  return { shape: [...shape], data: out };
}

const last = (t: Tensor): number => t.shape[t.shape.length - 1];
const d = (t: Tensor): Float32Array => t.data as Float32Array;

// ---- context: parameters and trace -------------------------------------------

export type Weights = Readonly<Record<string, Readonly<{ shape: readonly number[]; data: readonly number[] }>>>;

export class Ctx {
  /** Every parameter the model asked for, by name. Summing their sizes is the parameter count. */
  readonly params = new Map<string, readonly number[]>();
  /** Tensors the model marked, by op id: the shapes (and values) a page shows. */
  readonly trace = new Map<string, Tensor>();
  private readonly random: () => number;

  constructor(
    readonly mode: "shape" | "value",
    private readonly weights: Weights | null = null,
    seed = 0,
    /** Spread of seeded weights. Raise it when a toy run looks uniform (attention near 1/n everywhere). */
    private readonly gain = 1,
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
    if (this.mode === "shape") {
      return { shape: [...shape], data: null };
    }
    const loaded = this.weights?.[name];
    if (loaded !== undefined) {
      if (loaded.shape.join() !== shape.join()) {
        fail(`weights hold ${name} as [${loaded.shape.join(", ")}], model wants [${shape.join(", ")}]`);
      }
      return tensor(shape, loaded.data);
    }
    if (this.weights !== null) {
      fail(`weights have no tensor named ${name}`);
    }
    const scale = this.gain / Math.sqrt(shape[shape.length - 1]);
    return tensor(
      shape,
      Array.from({ length: size(shape) }, () => (this.random() * 2 - 1) * scale),
    );
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

function map(x: Tensor, f: (value: number) => number): Tensor {
  return make(x.shape, [x], (out) => {
    const a = d(x);
    for (let i = 0; i < out.length; i += 1) out[i] = f(a[i]);
  });
}

/** a + b. `b` may have fewer leading axes (it is repeated), e.g. a bias of shape [d]. */
function zip(a: Tensor, b: Tensor, f: (x: number, y: number) => number, name: string): Tensor {
  const tail = a.shape.slice(a.shape.length - b.shape.length);
  const fits = tail.every((dim, index) => dim === b.shape[index] || b.shape[index] === 1);
  if (!fits) {
    fail(`${name}: [${b.shape.join(", ")}] does not broadcast onto [${a.shape.join(", ")}]`);
  }
  return make(a.shape, [a, b], (out) => {
    const x = d(a);
    const y = d(b);
    if (b.shape.every((dim, index) => dim === tail[index])) {
      for (let i = 0; i < out.length; i += 1) out[i] = f(x[i], y[i % y.length]);
      return;
    }
    // Size-1 axes inside b: index the slow way.
    const strides = b.shape.map((_dim, axis) => size(b.shape.slice(axis + 1)));
    const aStrides = tail.map((_dim, axis) => size(tail.slice(axis + 1)));
    const block = size(tail);
    for (let i = 0; i < out.length; i += 1) {
      let rest = i % block;
      let j = 0;
      for (let axis = 0; axis < tail.length; axis += 1) {
        const at = Math.floor(rest / aStrides[axis]);
        rest %= aStrides[axis];
        j += (b.shape[axis] === 1 ? 0 : at) * strides[axis];
      }
      out[i] = f(x[i], y[j]);
    }
  });
}

export const add = (a: Tensor, b: Tensor): Tensor => zip(a, b, (x, y) => x + y, "add");
export const mul = (a: Tensor, b: Tensor): Tensor => zip(a, b, (x, y) => x * y, "mul");
export const scale = (x: Tensor, factor: number): Tensor => map(x, (value) => value * factor);
export const sigmoid = (x: Tensor): Tensor => map(x, (value) => 1 / (1 + Math.exp(-value)));
export const silu = (x: Tensor): Tensor => map(x, (value) => value / (1 + Math.exp(-value)));
export const softplus = (x: Tensor): Tensor => map(x, (value) => Math.log1p(Math.exp(value)));
export const sqrt = (x: Tensor): Tensor => map(x, Math.sqrt);
/** GELU in the tanh form used by GPT-2 ("gelu_new"). */
export const geluTanh = (x: Tensor): Tensor =>
  map(x, (v) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3))));

// ---- linear algebra ------------------------------------------------------------

/**
 * y = x @ W (+ b) over the last axis. `layout` says how W is stored:
 * "out_in" is torch.nn.Linear ([out, in]); "in_out" is GPT-2's Conv1D ([in, out]).
 */
export function affine(x: Tensor, w: Tensor, b: Tensor | null, layout: "out_in" | "in_out"): Tensor {
  const [inW, outW] = layout === "out_in" ? [w.shape[1], w.shape[0]] : [w.shape[0], w.shape[1]];
  if (last(x) !== inW) {
    fail(`affine: input width ${String(last(x))} does not match weight [${w.shape.join(", ")}] (${layout})`);
  }
  const shape = [...x.shape.slice(0, -1), outW];
  return make(shape, b === null ? [x, w] : [x, w, b], (out) => {
    const a = d(x);
    const m = d(w);
    const rows = a.length / inW;
    for (let r = 0; r < rows; r += 1) {
      for (let o = 0; o < outW; o += 1) {
        let sum = b === null ? 0 : d(b)[o];
        for (let i = 0; i < inW; i += 1) {
          sum += a[r * inW + i] * (layout === "out_in" ? m[o * inW + i] : m[i * outW + o]);
        }
        out[r * outW + o] = sum;
      }
    }
  });
}

/** Batched matrix product over the last two axes: [..., n, m] @ [..., m, p]. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const [n, m] = a.shape.slice(-2);
  const [m2, p] = b.shape.slice(-2);
  if (m !== m2 || a.shape.slice(0, -2).join() !== b.shape.slice(0, -2).join()) {
    fail(`matmul: [${a.shape.join(", ")}] @ [${b.shape.join(", ")}]`);
  }
  return make([...a.shape.slice(0, -2), n, p], [a, b], (out) => {
    const x = d(a);
    const y = d(b);
    const batches = x.length / (n * m);
    for (let t = 0; t < batches; t += 1) {
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < p; j += 1) {
          let sum = 0;
          for (let k = 0; k < m; k += 1) sum += x[t * n * m + i * m + k] * y[t * m * p + k * p + j];
          out[t * n * p + i * p + j] = sum;
        }
      }
    }
  });
}

// ---- shape changes -------------------------------------------------------------

export function reshape(x: Tensor, shape: readonly number[]): Tensor {
  if (size(shape) !== size(x.shape)) {
    fail(`reshape: [${x.shape.join(", ")}] to [${shape.join(", ")}]`);
  }
  return { shape: [...shape], data: x.data };
}

/** Reorders axes: permute(x, [0, 2, 1, 3]) swaps axes 1 and 2. */
export function permute(x: Tensor, order: readonly number[]): Tensor {
  const shape = order.map((axis) => x.shape[axis]);
  return make(shape, [x], (out) => {
    const a = d(x);
    const inStrides = x.shape.map((_dim, axis) => size(x.shape.slice(axis + 1)));
    const outStrides = shape.map((_dim, axis) => size(shape.slice(axis + 1)));
    for (let i = 0; i < out.length; i += 1) {
      let rest = i;
      let j = 0;
      for (let axis = 0; axis < shape.length; axis += 1) {
        const at = Math.floor(rest / outStrides[axis]);
        rest %= outStrides[axis];
        j += at * inStrides[order[axis]];
      }
      out[i] = a[j];
    }
  });
}

/** Cuts the last axis into pieces of the given widths. */
export function splitLast(x: Tensor, widths: readonly number[]): Tensor[] {
  const width = last(x);
  if (widths.reduce((sum, w) => sum + w, 0) !== width) {
    fail(`splitLast: widths ${widths.join(" + ")} do not sum to ${String(width)}`);
  }
  let offset = 0;
  return widths.map((w) => {
    const start = offset;
    offset += w;
    return make([...x.shape.slice(0, -1), w], [x], (out) => {
      const a = d(x);
      const rows = a.length / width;
      for (let r = 0; r < rows; r += 1) {
        for (let i = 0; i < w; i += 1) out[r * w + i] = a[r * width + start + i];
      }
    });
  });
}

/** Joins tensors along the last axis. */
export function concatLast(parts: readonly Tensor[]): Tensor {
  const width = parts.reduce((sum, part) => sum + last(part), 0);
  const lead = parts[0].shape.slice(0, -1);
  return make([...lead, width], parts, (out) => {
    const rows = size(lead);
    let offset = 0;
    for (const part of parts) {
      const w = last(part);
      const a = d(part);
      for (let r = 0; r < rows; r += 1) {
        for (let i = 0; i < w; i += 1) out[r * width + offset + i] = a[r * w + i];
      }
      offset += w;
    }
  });
}

// ---- normalisation, attention pieces ------------------------------------------

/** LayerNorm over the last axis: (x − mean) / sqrt(var + eps) · gain + bias. */
export function layerNorm(x: Tensor, gain: Tensor, bias: Tensor, eps: number): Tensor {
  const w = last(x);
  return make(x.shape, [x, gain, bias], (out) => {
    const a = d(x);
    for (let r = 0; r < a.length / w; r += 1) {
      let mean = 0;
      for (let i = 0; i < w; i += 1) mean += a[r * w + i];
      mean /= w;
      let variance = 0;
      for (let i = 0; i < w; i += 1) variance += (a[r * w + i] - mean) ** 2;
      const inv = 1 / Math.sqrt(variance / w + eps);
      for (let i = 0; i < w; i += 1) out[r * w + i] = (a[r * w + i] - mean) * inv * d(gain)[i] + d(bias)[i];
    }
  });
}

/** RMSNorm over the last axis: x / sqrt(mean(x²) + eps) · gain. */
export function rmsNorm(x: Tensor, gain: Tensor, eps: number): Tensor {
  const w = last(x);
  return make(x.shape, [x, gain], (out) => {
    const a = d(x);
    for (let r = 0; r < a.length / w; r += 1) {
      let square = 0;
      for (let i = 0; i < w; i += 1) square += a[r * w + i] ** 2;
      const inv = 1 / Math.sqrt(square / w + eps);
      for (let i = 0; i < w; i += 1) out[r * w + i] = a[r * w + i] * inv * d(gain)[i];
    }
  });
}

/** Softmax over the last axis. */
export function softmax(x: Tensor): Tensor {
  const w = last(x);
  return make(x.shape, [x], (out) => {
    const a = d(x);
    for (let r = 0; r < a.length / w; r += 1) {
      let max = -Infinity;
      for (let i = 0; i < w; i += 1) max = Math.max(max, a[r * w + i]);
      let sum = 0;
      for (let i = 0; i < w; i += 1) {
        out[r * w + i] = Math.exp(a[r * w + i] - max);
        sum += out[r * w + i];
      }
      for (let i = 0; i < w; i += 1) out[r * w + i] /= sum;
    }
  });
}

/**
 * Causal mask on scores [..., S, T]: query i may read keys 0 … (T − S) + i.
 * Everything later is set to −infinity. T > S when earlier keys come from a cache.
 */
export function causalMask(scores: Tensor): Tensor {
  const [s, t] = scores.shape.slice(-2);
  return make(scores.shape, [scores], (out) => {
    out.set(d(scores));
    for (let base = 0; base < out.length; base += s * t) {
      for (let i = 0; i < s; i += 1) {
        for (let j = t - s + i + 1; j < t; j += 1) out[base + i * t + j] = -Infinity;
      }
    }
  });
}

/** Rows of `table` selected by integer ids: ids [B, S] → [B, S, d]. */
export function embedding(table: Tensor, ids: readonly (readonly number[])[]): Tensor {
  const [rows, w] = table.shape;
  const shape = [ids.length, ids[0].length, w];
  return make(shape, [table], (out) => {
    ids.flat().forEach((id, at) => {
      if (id < 0 || id >= rows) fail(`embedding: id ${String(id)} outside 0 … ${String(rows - 1)}`);
      out.set(d(table).subarray(id * w, id * w + w), at * w);
    });
  });
}

/** Largest k entries of the last axis, as values and indices, sorted descending. */
export function topk(x: Tensor, k: number): { values: Tensor; indices: Tensor } {
  const w = last(x);
  const shape = [...x.shape.slice(0, -1), k];
  if (x.data === null) {
    return { values: { shape, data: null }, indices: { shape, data: null } };
  }
  const values = new Float32Array(size(shape));
  const indices = new Float32Array(size(shape));
  const a = d(x);
  for (let r = 0; r < a.length / w; r += 1) {
    const order = Array.from({ length: w }, (_u, i) => i).sort((p, q) => a[r * w + q] - a[r * w + p] || p - q);
    for (let i = 0; i < k; i += 1) {
      indices[r * k + i] = order[i];
      values[r * k + i] = a[r * w + order[i]];
    }
  }
  return { values: { shape, data: values }, indices: { shape, data: indices } };
}

/** Picks entries of the last axis of x at the given indices (same leading shape). */
export function gatherLast(x: Tensor, indices: Tensor): Tensor {
  const w = last(x);
  const k = last(indices);
  return make(indices.shape, [x, indices], (out) => {
    for (let r = 0; r < out.length / k; r += 1) {
      for (let i = 0; i < k; i += 1) out[r * k + i] = d(x)[r * w + d(indices)[r * k + i]];
    }
  });
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
