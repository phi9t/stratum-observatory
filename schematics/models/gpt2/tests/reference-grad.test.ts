// The clean GPT-2, differentiated and trained, against PyTorch: the loss, the gradient
// of every parameter, and three optimizer steps must reproduce what autograd and
// torch.optim computed for the upstream model (reference/make_fixture.py).

import { describe, expect, it } from "vitest";

import { AdamW, type Optimizer, sgd, trainable, trainStep } from "../../../kit/ref/optim";
import { crossEntropy, Ctx, Tape, type Weights } from "../../../kit/ref/tensor";
import * as f from "../facts";
import fixture from "../reference/fixture.json";
import { forward, type Gpt2Config } from "../reference/model";

const toy: Gpt2Config = {
  vocab: fixture.config.vocab_size,
  positions: fixture.config.n_positions,
  hidden: fixture.config.n_embd,
  layers: fixture.config.n_layer,
  heads: fixture.config.n_head,
  mlpRatio: f.mlpRatio.value,
  normEps: f.normEps.value,
};
const batch = { inputs: fixture.input_ids, targets: fixture.targets };

type Values = { shape: readonly number[]; data: ArrayLike<number> | null };

/** The parameter on which two sets of tensors differ most, and by how much. */
function worst(
  ours: ReadonlyMap<string, Values>,
  theirs: Weights,
  skip: (name: string, index: number) => boolean = () => false,
): { name: string; diff: number } {
  expect([...ours.keys()].sort()).toEqual(Object.keys(theirs).sort());
  let found = { name: "", diff: 0 };
  for (const [name, { shape, data }] of ours) {
    expect(shape).toEqual(theirs[name].shape);
    for (let index = 0; index < data!.length; index += 1) {
      const diff = Math.abs(data![index] - theirs[name].data[index]);
      if (diff >= found.diff && !skip(name, index)) found = { name, diff };
    }
  }
  return found;
}

/**
 * The key third of c_attn.bias. Adding a bias to every key shifts all of a query's
 * scores by the same amount, which softmax ignores: the true gradient is exactly zero.
 * In Float32 both PyTorch and this library compute rounding noise there instead (about
 * 1e-10). SGD multiplies that noise by lr and nothing happens, but Adam divides each
 * gradient by its own magnitude, so the noise becomes a step of up to lr in a direction
 * that no two implementations agree on. These 24 values per layer are compared apart.
 */
const isKeyBias = (name: string, index: number): boolean =>
  name.endsWith(".attn.c_attn.bias") && index >= toy.hidden && index < 2 * toy.hidden;

/** Three training steps from the fixture's weights: per-step losses and the final parameters. */
function train(optimizer: Optimizer): { losses: number[]; weights: Map<string, Values> } {
  const weights = trainable(fixture.weights);
  const model = (ctx: Ctx, ids: readonly (readonly number[])[]) => forward(ctx, toy, ids);
  const losses = [0, 1, 2].map(() => trainStep({ forward: model, weights, batch, optimizer }).loss);
  return { losses, weights: new Map(Object.entries(weights)) };
}

function expectLosses(ours: readonly number[], theirs: readonly number[]): void {
  expect(ours).toHaveLength(theirs.length);
  ours.forEach((loss, step) => {
    expect(Math.abs(loss - theirs[step]), `loss before step ${String(step + 1)}`).toBeLessThan(1e-4);
  });
}

describe("clean GPT-2 reference, differentiated", () => {
  const tape = new Tape();
  const ctx = new Ctx("value", fixture.weights, 0, 1, tape);
  const loss = crossEntropy(forward(ctx, toy, batch.inputs), batch.targets);
  tape.backward(loss);

  it("reproduces PyTorch's next-token loss", () => {
    expect(loss.shape).toEqual([]);
    expect(Math.abs(loss.data![0] - fixture.loss)).toBeLessThan(1e-5);
  });

  it("reproduces PyTorch's gradient for every parameter", () => {
    const { name, diff } = worst(ctx.grads(), fixture.grads);
    expect(diff, `worst gradient is ${name}`).toBeLessThan(1e-4);
    // The token embedding is both the input table and the output head: one tensor, one
    // gradient holding the sum of both uses. Rows of tokens absent from the input can
    // only get theirs through the head.
    const wte = ctx.grads().get("transformer.wte.weight")!;
    const unusedToken = 1;
    expect(batch.inputs.flat()).not.toContain(unusedToken);
    expect(Math.abs(wte.data![unusedToken * toy.hidden])).toBeGreaterThan(0);
  });

  it("gives gradients by tensor identity as well as by name", () => {
    const logits = ctx.trace.get("logits")!;
    expect(tape.grad(logits).shape).toEqual(logits.shape);
    // ∂loss/∂logits = (softmax − onehot) / positions: every row sums to zero.
    const row = tape.grad(logits).data!.subarray(0, toy.vocab);
    expect(Math.abs(row.reduce((total, value) => total + value, 0))).toBeLessThan(1e-6);
  });

  it("reproduces three steps of torch.optim.AdamW", () => {
    const { lr, weight_decay: weightDecay, betas, eps } = fixture.adamw.options;
    const run = train(new AdamW({ lr, weightDecay, betas: [betas[0], betas[1]], eps }));
    expectLosses(run.losses, fixture.adamw.losses);
    const { name, diff } = worst(run.weights, fixture.adamw.weights, isKeyBias);
    expect(diff, `final parameters, worst is ${name}`).toBeLessThan(1e-4);

    // The key biases (see isKeyBias): PyTorch's gradient there is noise around zero, and
    // three Adam steps on noise cannot move a value further than 3 · lr.
    const keyBiasOnly = (n: string, index: number): boolean => !isKeyBias(n, index);
    expect(worst(ctx.grads(), fixture.grads, keyBiasOnly).diff).toBeLessThan(1e-8);
    expect(worst(run.weights, fixture.adamw.weights, keyBiasOnly).diff).toBeLessThan(3 * lr);
  });

  it("reproduces three steps of torch.optim.SGD", () => {
    const { lr } = fixture.sgd.options;
    const run = train({ step: (params, grads) => sgd(params, grads, { lr }) });
    expectLosses(run.losses, fixture.sgd.losses);
    const { name, diff } = worst(run.weights, fixture.sgd.weights);
    expect(diff, `final parameters, worst is ${name}`).toBeLessThan(1e-4);
  });

  it("still runs in shape mode, with no tape and no data", () => {
    const shapes = new Ctx("shape");
    const shapeLoss = crossEntropy(forward(shapes, toy, batch.inputs), batch.targets);
    expect(shapeLoss).toEqual({ shape: [], data: null });
  });
});
