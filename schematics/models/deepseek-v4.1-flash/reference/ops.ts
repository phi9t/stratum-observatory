import { type Tensor, tensor } from "../../../kit/ref/tensor";

export function route(logits: readonly number[], bias: readonly number[], k: number, scale: number) {
  const scores = logits.map((x) => Math.sqrt(Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)))));
  const chosen = scores
    .map((_, i) => i)
    .sort((a, b) => scores[b] + bias[b] - scores[a] - bias[a] || a - b)
    .slice(0, k);
  const sum = chosen.reduce((s, i) => s + scores[i], 0) + 1e-20;
  return { scores, chosen, weights: chosen.map((i) => (scores[i] / sum) * scale), chosenGroups: [0] };
}
export function candidateMask(scores: readonly number[], visible: number, k: number, block: number) {
  const last = Math.floor((visible - 1) / block);
  const blocks = Array.from({ length: Math.ceil(scores.length / block) }, (_, b) => ({
    b,
    score:
      b === last
        ? Infinity
        : Math.max(
            ...scores.slice(b * block, (b + 1) * block).map((x, i) => (b * block + i < visible ? x : -Infinity)),
          ),
  }));
  const chosen = blocks
    .sort((a, b) => b.score - a.score || a.b - b.b)
    .slice(0, k)
    .filter((x) => x.score > -Infinity)
    .map((x) => x.b);
  return scores.map((_, i) => i < visible && chosen.includes(Math.floor(i / block)));
}
export function sinkhorn(logits: readonly number[], h: number, iters: number, eps: number) {
  const m = [...logits];
  for (let r = 0; r < h; r++) {
    const max = Math.max(...m.slice(r * h, (r + 1) * h));
    let sum = 0;
    for (let c = 0; c < h; c++) {
      m[r * h + c] = Math.exp(m[r * h + c] - max);
      sum += m[r * h + c];
    }
    for (let c = 0; c < h; c++) m[r * h + c] = m[r * h + c] / sum + eps;
  }
  for (let t = 0; t < iters; t++) {
    if (t > 0)
      for (let r = 0; r < h; r++) {
        const sum = m.slice(r * h, (r + 1) * h).reduce((a, b) => a + b, 0) + eps;
        for (let c = 0; c < h; c++) m[r * h + c] /= sum;
      }
    for (let c = 0; c < h; c++) {
      let sum = eps;
      for (let r = 0; r < h; r++) sum += m[r * h + c];
      for (let r = 0; r < h; r++) m[r * h + c] /= sum;
    }
  }
  return m;
}
export function map(x: Tensor, f: (x: number) => number): Tensor {
  return tensor(x.shape, x.data === null ? null : Array.from(x.data, f));
}
export function make(shape: readonly number[], inputs: readonly Tensor[], f: () => ArrayLike<number>): Tensor {
  return tensor(shape, inputs.some((x) => x.data === null) ? null : f());
}
export function rope(
  x: Tensor,
  rd: number,
  base: number,
  ratio: number,
  original: number,
  factor: number,
  fast: number,
  slow: number,
  inverse = false,
): Tensor {
  const width = x.shape.at(-1)!;
  const seq = x.shape[1];
  const heads = x.shape.length === 4 ? x.shape[2] : 1;
  return make(x.shape, [x], () => {
    const a = Float32Array.from(x.data!);
    for (let i = 0; i < a.length; i += width) {
      const pos = (Math.floor(i / width / heads) % seq) * ratio;
      for (let p = 0; p < rd / 2; p++) {
        const freq = rotaryFrequency(p, rd, base, original, factor, fast, slow);
        const ang = pos * freq * (inverse ? -1 : 1),
          j = i + width - rd + 2 * p,
          u = a[j],
          v = a[j + 1];
        a[j] = u * Math.cos(ang) - v * Math.sin(ang);
        a[j + 1] = u * Math.sin(ang) + v * Math.cos(ang);
      }
    }
    return a;
  });
}

export function rotaryFrequency(
  pair: number,
  rd: number,
  base: number,
  original: number,
  factor: number,
  fast: number,
  slow: number,
) {
  const frequency = base ** ((-2 * pair) / rd);
  if (!original) return frequency;
  const corrected = (r: number) => (rd * Math.log(original / (r * 2 * Math.PI))) / (2 * Math.log(base));
  const low = Math.max(Math.floor(corrected(fast)), 0),
    high = Math.min(Math.ceil(corrected(slow)), rd - 1);
  const smooth = 1 - Math.max(0, Math.min(1, (pair - low) / Math.max(high - low, 1e-3)));
  return (frequency / factor) * (1 - smooth) + frequency * smooth;
}
