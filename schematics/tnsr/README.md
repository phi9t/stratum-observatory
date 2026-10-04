# tnsr: a tensor library meant to be read

`tensor.ts` is what the clean reference implementations (`models/<slug>/reference/`) are
written on, so that a forward pass, its gradients and a training step can be read and
trusted: every piece is checked against PyTorch. Autoregressive decoders only.

## Two modes, one model

- **Shape mode** (`new Ctx("shape")`): no data; every op only computes its output shape.
  Run at the checkpoint's real widths for exact shapes and an exact parameter count.
- **Value mode** (`new Ctx("value", weights)`): tensors hold a `Float32Array`. Run at toy
  widths, with weights from a parity fixture or drawn from a seed.

## How the tape works

Each op in `tensor.ts` is a pure function that states, next to its forward computation,
how a gradient on its output is shared out among its inputs. A `Tape` (`tape.ts`) is
opt-in: a tensor made by `tape.watch(t)`, or a parameter of a `Ctx` that was given a
tape, carries a reference to it, and any op with such an input appends one step to the
tape and passes the reference on. `tape.backward(loss)` sets the scalar loss's gradient
to 1 and replays the steps in reverse, so each tensor's gradient is complete before it
is handed to the tensors it was computed from. A tensor used twice (a tied embedding, a
residual stream) simply receives two additions into the same buffer, which is why
`ctx.param(name)` returns the same tensor every time. Without a tape nothing is
recorded, and model code is the same either way.

```ts
const tape = new Tape();
const ctx = new Ctx("value", weights, 0, 1, tape);
const loss = crossEntropy(forward(ctx, config, inputs), targets);
tape.backward(loss);
const byName = ctx.grads(); // Map<string, Tensor>; or tape.grad(anyTensor)
```

## Ops

Elementwise: `add`, `sub`, `mul` (the second operand may be repeated over leading or
size-1 axes), `scale`, `sigmoid`, `silu`, `softplus`, `sqrt`, `geluTanh`.
Linear algebra: `affine` (weight stored `[out, in]` or `[in, out]`), `matmul`.
Shape: `reshape`, `permute`, `splitLast`, `concatLast`. Model pieces: `layerNorm`,
`rmsNorm`, `softmax`, `causalMask`, `embedding`, `topk`, `gatherLast`.
Reductions and loss: `sum`, `mean`, `crossEntropy`.
`optim.ts` adds `sgd`, `AdamW` (as `torch.optim.AdamW`) and `trainStep`: forward, loss,
backward and one update of a `Params` object.

## What proves it

- `autodiff.test.ts` checks every op's derivative against central finite differences.
- `models/gpt2/reference/make_fixture.py` builds the upstream GPT-2 at toy width and
  exports weights, logits, the loss, autograd's gradient of every parameter, and three
  steps of AdamW and of SGD. `models/gpt2/tests/reference*.test.ts` reproduce all of it.

## Limits

- Toy scale, plain loops: about 4 ms per training step for a 2-layer, width-24 GPT-2.
- Float32 values and gradients, like PyTorch's default; agreement is to about 1e-6.
- No in-place ops: every op returns a new tensor. Only the optimizers overwrite values.
- First derivatives only: the backward pass is not itself recorded.
- Where the true gradient is zero but Float32 computes noise (attention's key bias),
  Adam's steps differ between implementations; the GPT-2 test compares those apart.
