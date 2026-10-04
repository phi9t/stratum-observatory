import { tensor, type Tensor } from "../../../tnsr/tensor";

/** Split-half rotation, not adjacent pairs; q/k are [B,H,S,D]. */
export function rotary(x: Tensor, theta: number): Tensor {
  const [b, h, s, d] = x.shape;
  if (x.data === null) return { shape: x.shape, data: null };
  const out = new Float32Array(x.data.length);
  for (let batch = 0; batch < b; batch++)
    for (let head = 0; head < h; head++)
      for (let pos = 0; pos < s; pos++)
        for (let pair = 0; pair < d / 2; pair++) {
          const i = ((batch * h + head) * s + pos) * d + pair;
          const angle = pos * theta ** ((-2 * pair) / d),
            c = Math.cos(angle),
            sn = Math.sin(angle);
          out[i] = x.data[i] * c - x.data[i + d / 2] * sn;
          out[i + d / 2] = x.data[i + d / 2] * c + x.data[i] * sn;
        }
  return tensor(x.shape, out);
}
/** Repeat KV heads contiguously, matching repeat_kv, only after cache capture. */
export function repeatKV(x: Tensor, n: number): Tensor {
  if (!Number.isInteger(n) || n < 1) throw new RangeError("repeatKV needs a positive integer repeat count");
  const [b, h, s, d] = x.shape;
  const shape = [b, h * n, s, d];
  if (x.data === null) return { shape, data: null };
  const out = new Float32Array(b * h * n * s * d);
  for (let bi = 0; bi < b; bi++)
    for (let hi = 0; hi < h; hi++)
      for (let r = 0; r < n; r++) {
        const offset = (bi * h + hi) * s * d;
        out.set(x.data.subarray(offset, offset + s * d), ((bi * h + hi) * n + r) * s * d);
      }
  return tensor(shape, out);
}
/** Renormalize selected scores, not all experts. */
export function normalizeSelected(x: Tensor): Tensor {
  if (x.data === null) return { shape: x.shape, data: null };
  const k = x.shape[x.shape.length - 1],
    out = new Float32Array(x.data.length);
  for (let r = 0; r < out.length / k; r++) {
    let sum = 0;
    for (let j = 0; j < k; j++) sum += x.data[r * k + j];
    for (let j = 0; j < k; j++) out[r * k + j] = x.data[r * k + j] / sum;
  }
  return tensor(x.shape, out);
}
/** Token dispatch indices for one expert, used by gather / weighted scatter. */
export function expertAssignments(indices: Tensor, expert: number): { tokens: number[]; slots: number[] } {
  const tokens: number[] = [],
    slots: number[] = [];
  if (indices.data !== null)
    for (let i = 0; i < indices.data.length; i++)
      if (indices.data[i] === expert) {
        tokens.push(Math.floor(i / indices.shape[1]));
        slots.push(i % indices.shape[1]);
      }
  return { tokens, slots };
}
export function gatherTokens(x: Tensor, tokens: readonly number[]): Tensor {
  const d = x.shape[1],
    shape = [tokens.length, d];
  if (x.data === null) return { shape, data: null };
  return tensor(
    shape,
    tokens.flatMap((t) => Array.from(x.data!.subarray(t * d, (t + 1) * d))),
  );
}
export function scatterWeighted(
  out: Float32Array,
  updates: Tensor,
  tokens: readonly number[],
  slots: readonly number[],
  weights: Tensor,
): void {
  const d = updates.shape[1],
    k = weights.shape[1];
  tokens.forEach((t, i) => {
    for (let j = 0; j < d; j++) out[t * d + j] += updates.data![i * d + j] * weights.data![t * k + slots[i]];
  });
}
