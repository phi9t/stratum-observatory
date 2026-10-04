// Model-local shape/value operations absent from tnsr/tensor. No autodiff is
// claimed: these are inference operations and intentionally do not attach a tape.
import { tensor, type Tensor } from "../../../tnsr/tensor";

/** Broadcast a shared positional key's head axis, retaining batch/sequence. */
export function repeatHeads(x: Tensor, heads: number): Tensor {
  const [b, one, s, d] = x.shape;
  if (one !== 1 || !Number.isInteger(heads) || heads < 1) throw new RangeError("shared head expected");
  const shape = [b, heads, s, d];
  if (x.data === null) return tensor(shape, null);
  const values = new Float32Array(b * heads * s * d);
  for (let i = 0; i < b; i++)
    for (let h = 0; h < heads; h++) values.set(x.data.subarray(i * s * d, (i + 1) * s * d), (i * heads + h) * s * d);
  return tensor(shape, values);
}

/** Pinned apply_rotary_pos_emb first turns adjacent pairs into half layout. */
export function rotateYarn(x: Tensor, frequencies: readonly number[], amplitude: number): Tensor {
  const [b, h, s, d] = x.shape;
  if (d !== frequencies.length * 2) throw new RangeError("rotary pair count mismatch");
  if (x.data === null) return tensor(x.shape, null);
  const out = new Float32Array(x.data.length);
  for (let batch = 0; batch < b; batch++)
    for (let head = 0; head < h; head++)
      for (let pos = 0; pos < s; pos++) {
        const offset = ((batch * h + head) * s + pos) * d;
        for (let j = 0; j < d / 2; j++) {
          const angle = pos * frequencies[j],
            c = Math.cos(angle) * amplitude,
            sn = Math.sin(angle) * amplitude;
          const a = x.data[offset + 2 * j],
            v = x.data[offset + 2 * j + 1];
          out[offset + j] = a * c - v * sn;
          out[offset + d / 2 + j] = v * c + a * sn;
        }
      }
  return tensor(x.shape, out);
}
