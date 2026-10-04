// Scalar operations missing from kit/ref. Shape mode never allocates value arrays.
import { tensor, sigmoid, add, topk, gatherLast, softmax, rmsNorm, type Tensor } from "../../../kit/ref/tensor";
export function mapped(x: Tensor, fn: (x: number, i: number) => number): Tensor {
  return x.data === null ? { shape: x.shape, data: null } : tensor(x.shape, Array.from(x.data, fn));
}
export function route(logits: Tensor, bias: Tensor, k: number) {
  const scores = sigmoid(logits);
  const choice = add(scores, bias);
  const ids = topk(choice, k).indices;
  const selected = gatherLast(scores, ids);
  const weights = mapped(selected, (x, i) => {
    const start = Math.floor(i / k) * k;
    let sum = 1e-20;
    for (let j = 0; j < k; j++) sum += selected.data![start + j];
    return x / sum;
  });
  return { scores, choice, ids, weights };
}
export function attnRes(prefix: Tensor, bank: readonly Tensor[], gain: Tensor, proj: Tensor, eps: number) {
  const [B, S, D] = prefix.shape,
    N = bank.length + 1;
  const candidates = [...bank, prefix];
  if (prefix.data === null)
    return {
      output: { shape: prefix.shape, data: null } as Tensor,
      weights: { shape: [B, S, N], data: null } as Tensor,
    };
  const scores = new Float32Array(B * S * N);
  for (let r = 0; r < B * S; r++)
    for (let n = 0; n < N; n++) {
      let sq = 0;
      for (let d = 0; d < D; d++) sq += candidates[n].data![r * D + d] ** 2;
      let score = 0;
      for (let d = 0; d < D; d++)
        score += (candidates[n].data![r * D + d] / Math.sqrt(sq / D + eps)) * gain.data![d] * proj.data![d];
      scores[r * N + n] = score;
    }
  const weights = softmax(tensor([B, S, N], scores)),
    out = new Float32Array(B * S * D);
  for (let r = 0; r < B * S; r++)
    for (let d = 0; d < D; d++)
      for (let n = 0; n < N; n++) out[r * D + d] += weights.data![r * N + n] * candidates[n].data![r * D + d];
  return { output: tensor(prefix.shape, out), weights };
}
export function convolution(x: Tensor, w: Tensor): Tensor {
  const [B, S, D] = x.shape,
    K = w.shape[2];
  if (!x.data) return { shape: x.shape, data: null };
  const out = new Float32Array(B * S * D);
  for (let b = 0; b < B; b++)
    for (let s = 0; s < S; s++)
      for (let d = 0; d < D; d++) {
        let z = 0;
        for (let k = 0; k < K; k++) {
          const at = s + k - (K - 1);
          if (at >= 0) z += x.data[(b * S + at) * D + d] * w.data![d * K + k];
        }
        out[(b * S + s) * D + d] = z / (1 + Math.exp(-z));
      }
  return tensor(x.shape, out);
}
export function l2norm(x: Tensor): Tensor {
  const D = x.shape.at(-1)!;
  return mapped(x, (v, i) => {
    let sq = 1e-6;
    const start = Math.floor(i / D) * D;
    for (let d = 0; d < D; d++) sq += x.data![start + d] ** 2;
    return v / Math.sqrt(sq);
  });
}
export function decayGate(g: Tensor, a: Tensor, bias: Tensor, lower: number): Tensor {
  const H = g.shape[2],
    D = g.shape[3];
  return mapped(
    g,
    (v, i) => lower / (1 + Math.exp(-Math.exp(a.data![Math.floor(i / D) % H]) * (v + bias.data![i % (H * D)]))),
  );
}
// Logical K×V layout; publisher asks the GPU kernel for transposed physical storage.
export function kdaRecurrence(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) {
  const [B, S, H, D] = q.shape,
    V = v.shape[3];
  if (!q.data)
    return { output: { shape: v.shape, data: null } as Tensor, state: { shape: [B, H, D, V], data: null } as Tensor };
  const state = new Float32Array(B * H * D * V),
    out = new Float32Array(B * S * H * V);
  for (let b = 0; b < B; b++)
    for (let s = 0; s < S; s++)
      for (let h = 0; h < H; h++) {
        const off = ((b * S + s) * H + h) * D,
          so = (b * H + h) * D * V,
          vo = ((b * S + s) * H + h) * V;
        for (let d = 0; d < D; d++) for (let j = 0; j < V; j++) state[so + d * V + j] *= Math.exp(g.data![off + d]);
        const delta = new Float32Array(V);
        for (let j = 0; j < V; j++) {
          let pred = 0;
          for (let d = 0; d < D; d++) pred += k.data![off + d] * state[so + d * V + j];
          delta[j] = (v.data![vo + j] - pred) * beta.data![(b * S + s) * H + h];
        }
        for (let d = 0; d < D; d++) for (let j = 0; j < V; j++) state[so + d * V + j] += k.data![off + d] * delta[j];
        for (let j = 0; j < V; j++)
          for (let d = 0; d < D; d++) out[vo + j] += (q.data[off + d] / Math.sqrt(D)) * state[so + d * V + j];
      }
  return { output: tensor(v.shape, out), state: tensor([B, H, D, V], state) };
}
export function gatedNorm(x: Tensor, g: Tensor, gain: Tensor, eps: number): Tensor {
  const norm = rmsNorm(x, gain, eps);
  return mapped(norm, (v, i) => v / (1 + Math.exp(-g.data![i])));
}
