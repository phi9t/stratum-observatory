import { expect, it } from "vitest";
import { Ctx } from "../../../tnsr/tensor";
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
import { maxAbsDiff, tensor, type Weights } from "../../../tnsr/tensor";
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
import { repeatKv, LlamaRotaryEmbedding } from "../reference/model";
it("repeats each KV head consecutively, without modifying the source", () => {
  const x = tensor([1, 2, 1, 2], [1, 2, 3, 4]);
  expect([...repeatKv(x, 2).data!]).toEqual([1, 2, 1, 2, 3, 4, 3, 4]);
  expect([...x.data!]).toEqual([1, 2, 3, 4]);
});
it("rotates half-split pairs and leaves the zero position unchanged", () => {
  const x = tensor([1, 1, 2, 4], [1, 2, 3, 4, 1, 2, 3, 4]);
  const y = LlamaRotaryEmbedding(x, 1000000);
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

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import sources from "../sources.json";
it("binds the upstream fixture to the pinned reference bytes", () => {
  const source = sources.sources.find((s) => s.id === "reference")!;
  const bytes = readFileSync(`.sources-cache/smollm2-135m/reference/${source.path}`);
  expect(fixture.source_sha256).toBe(createHash("sha256").update(bytes).digest("hex"));
  expect(fixture.source_revision).toBe(source.revision);
  expect(fixture.transformers_version).toBe("4.51.3");
});
it("never allocates a separate output head or head normalization gains", () => {
  const c = new Ctx("shape");
  forward(c, realConfig, [[0]]);
  expect(
    [...c.params.keys()].filter(
      (name) => name.startsWith("lm_head") || name.includes("q_norm") || name.includes("k_norm"),
    ),
  ).toEqual([]);
});
it("shared KV heads remain identical while their query probabilities can differ", () => {
  const cfg = { ...f.TOY, normEps: f.normEps.value, theta: f.theta.value };
  const c = new Ctx("value", null, 1, f.DISPLAY.runGain);
  forward(c, cfg, [[1, 8, 15, 22, 29, 36, 3, 10]]);
  const stride = cfg.tokens * cfg.headDim;
  const k = c.trace.get("h0.attn.k_repeated")!.data!;
  expect([...k.slice(0, stride)]).toEqual([...k.slice(stride, 2 * stride)]);
  const w = c.trace.get("h0.attn.weights")!.data!;
  const row = cfg.tokens * cfg.tokens;
  expect([...w.slice(0, row)]).not.toEqual([...w.slice(row, 2 * row)]);
});
