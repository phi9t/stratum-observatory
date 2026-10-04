import { expect, it } from "vitest";
import { Ctx } from "../../../kit/ref/tensor";
import { forward, realConfig } from "../reference/model";
import * as f from "../facts";
it("real shape mode reconciles the ledger and grouped head shapes", () => {
  const c = new Ctx("shape");
  const y = forward(c, realConfig, [[0, 1, 2]]);
  expect(c.parameterCount()).toBe(f.parameterTotal.value);
  expect(y.shape).toEqual([1, 3, f.vocab.value]);
  expect(c.shapeOf("h0.attn.k")).toEqual([1, f.kvHeads.value, 3, f.headDim.value]);
  expect(c.shapeOf("h0.attn.q")).toEqual([1, f.heads.value, 3, f.headDim.value]);
});

import fixture from "../reference/fixture.json";
import { maxAbsDiff, tensor, type Weights } from "../../../kit/ref/tensor";
it("matches pinned upstream logits, attention and parameter names with perturbed weights", () => {
  const cfg = {
    vocab: fixture.config.vocab_size,
    hidden: fixture.config.hidden_size,
    layers: fixture.config.num_hidden_layers,
    heads: fixture.config.num_attention_heads,
    kvHeads: fixture.config.num_key_value_heads,
    headDim: fixture.config.head_dim,
    inner: fixture.config.intermediate_size,
    normEps: fixture.config.rms_norm_eps,
    theta: fixture.config.rope_theta,
  };
  const c = new Ctx("value", fixture.weights as Weights);
  const y = forward(c, cfg, fixture.input_ids);
  expect(maxAbsDiff(y, tensor(fixture.logits.shape, fixture.logits.data))).toBeLessThan(1e-4);
  expect(
    maxAbsDiff(c.trace.get("h0.attn.weights")!, tensor(fixture.attention_layer0.shape, fixture.attention_layer0.data)),
  ).toBeLessThan(1e-5);
  expect([...c.params.keys()].sort()).toEqual(Object.keys(fixture.weights).sort());
});
import { repeatKv, Qwen3RotaryEmbedding } from "../reference/model";
it("repeats each KV head consecutively, without modifying the source", () => {
  const x = tensor([1, 2, 1, 2], [1, 2, 3, 4]);
  expect([...repeatKv(x, 2).data!]).toEqual([1, 2, 1, 2, 3, 4, 3, 4]);
  expect([...x.data!]).toEqual([1, 2, 3, 4]);
});
it("rotates half-split pairs and leaves the zero position unchanged", () => {
  const x = tensor([1, 1, 2, 4], [1, 2, 3, 4, 1, 2, 3, 4]);
  const y = Qwen3RotaryEmbedding(x, 1000000);
  expect([...y.data!.slice(0, 4)]).toEqual([1, 2, 3, 4]);
  expect(y.data![4]).toBeCloseTo(Math.cos(1) - 3 * Math.sin(1), 5);
});

it("future token changes cannot affect earlier logits", () => {
  const cfg = { ...f.TOY, normEps: f.normEps.value, theta: f.theta.value };
  const a = forward(new Ctx("value", fixture.weights as Weights), cfg, [[3, 17, 5, 29, 11, 0, 38, 8]]);
  const b = forward(new Ctx("value", fixture.weights as Weights), cfg, [[3, 17, 5, 29, 11, 0, 38, 9]]);
  expect([...a.data!.slice(0, -cfg.vocab)]).toEqual([...b.data!.slice(0, -cfg.vocab)]);
  expect([...a.data!.slice(-cfg.vocab)]).not.toEqual([...b.data!.slice(-cfg.vocab)]);
});

it("exposes computed projection, head-normalization and rotary score stages", () => {
  const cfg = { ...f.TOY, normEps: f.normEps.value, theta: f.theta.value };
  const c = new Ctx("value", null, 1, 4);
  forward(c, cfg, [[1, 8, 15, 22, 29, 36, 3, 10]]);
  for (const name of ["q_raw", "k_raw", "q_norm", "k_norm", "scores_raw", "scores_norm", "k_repeated", "v_repeated"])
    expect(c.trace.get(`h0.attn.${name}`)?.data).toBeTruthy();
  expect(maxAbsDiff(c.trace.get("h0.attn.scores_raw")!, c.trace.get("h0.attn.scores_norm")!)).toBeGreaterThan(0.1);
  expect(maxAbsDiff(c.trace.get("h0.attn.scores_norm")!, c.trace.get("h0.attn.scores")!)).toBeGreaterThan(0.1);
});

it("shared KV copies are identical while queries differ; traced scores equal scalar dot products", () => {
  const cfg = { ...f.TOY, normEps: f.normEps.value, theta: f.theta.value };
  const c = new Ctx("value", null, 1, f.DISPLAY.runGain);
  forward(c, cfg, [[1, 8, 15, 22, 29, 36, 3, 10]]);
  const vec = (name: string, head: number, pos: number) =>
    Array.from(
      c.trace
        .get(`h0.attn.${name}`)!
        .data!.slice((head * cfg.tokens + pos) * cfg.headDim, (head * cfg.tokens + pos + 1) * cfg.headDim),
    );
  for (const name of ["k_repeated", "v_repeated"])
    for (let pos = 0; pos < cfg.tokens; pos++) expect(vec(name, 0, pos)).toEqual(vec(name, 1, pos));
  expect(vec("q", 0, 7)).not.toEqual(vec("q", 1, 7));
  for (const [score, q, k] of [
    ["scores_raw", "q_raw", "k_raw"],
    ["scores_norm", "q_norm", "k_norm"],
    ["scores", "q", "k"],
  ]) {
    const a = vec(q, 1, 7),
      b = vec(k, 0, 2);
    const expected = a.reduce((sum, value, i) => sum + value * b[i], 0) / Math.sqrt(cfg.headDim);
    expect(c.trace.get(`h0.attn.${score}`)!.data![(cfg.tokens + 7) * cfg.tokens + 2]).toBeCloseTo(expected, 5);
  }
  const a = vec("q_norm", 1, 7),
    b = vec("q", 1, 7);
  expect(Math.hypot(a[0], a[cfg.headDim / 2])).toBeCloseTo(Math.hypot(b[0], b[cfg.headDim / 2]), 5);
  const probabilities = c.trace.get("h0.attn.weights")!.data!.slice(7 * cfg.tokens, 8 * cfg.tokens);
  expect(Math.max(...probabilities) - Math.min(...probabilities)).toBeGreaterThan(0.1);
});
