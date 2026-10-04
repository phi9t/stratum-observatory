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
