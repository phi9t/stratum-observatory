// GPT-2, written once and cleanly: the forward pass of GPT2LMHeadModel with no
// framework glue, no optional branches this checkpoint does not take, and no caching.
// Parameter names are the checkpoint's own (transformer.h.0.attn.c_attn.weight, …) so
// weights exported from the reference implementation load without a mapping.
//
// Distilled from transformers modeling_gpt2.py (GPT2Model, GPT2Block, GPT2Attention,
// GPT2MLP, eager_attention_forward) at the commit pinned in ../sources.json, and
// checked against it numerically by ../tests/reference.test.ts.

import {
  add,
  affine,
  causalMask,
  type Ctx,
  embedding,
  geluTanh,
  layerNorm,
  matmul,
  permute,
  reshape,
  scale,
  softmax,
  splitLast,
  type Tensor,
} from "../../../kit/ref/tensor";

export type Gpt2Config = Readonly<{
  vocab: number;
  positions: number;
  hidden: number;
  layers: number;
  heads: number;
  mlpRatio: number;
  normEps: number;
}>;

function norm(ctx: Ctx, name: string, x: Tensor, cfg: Gpt2Config): Tensor {
  return layerNorm(x, ctx.param(`${name}.weight`, [cfg.hidden]), ctx.param(`${name}.bias`, [cfg.hidden]), cfg.normEps);
}

/** GPT-2's Conv1D: a linear layer whose weight is stored [in, out]. */
function conv1d(ctx: Ctx, name: string, x: Tensor, inWidth: number, outWidth: number): Tensor {
  return affine(x, ctx.param(`${name}.weight`, [inWidth, outWidth]), ctx.param(`${name}.bias`, [outWidth]), "in_out");
}

function attention(ctx: Ctx, name: string, id: string, x: Tensor, cfg: Gpt2Config): Tensor {
  const [batch, seq] = x.shape;
  const headDim = cfg.hidden / cfg.heads;
  const toHeads = (t: Tensor): Tensor => permute(reshape(t, [batch, seq, cfg.heads, headDim]), [0, 2, 1, 3]);

  const qkv = ctx.mark(`${id}.qkv`, conv1d(ctx, `${name}.c_attn`, x, cfg.hidden, 3 * cfg.hidden));
  const [q, k, v] = splitLast(qkv, [cfg.hidden, cfg.hidden, cfg.hidden]).map(toHeads);
  ctx.mark(`${id}.q`, q);

  const scores = ctx.mark(`${id}.scores`, scale(matmul(q, permute(k, [0, 1, 3, 2])), headDim ** -0.5));
  const weights = ctx.mark(`${id}.weights`, softmax(causalMask(scores)));
  const mixed = ctx.mark(`${id}.mixed`, matmul(weights, v));
  const merged = ctx.mark(`${id}.merged`, reshape(permute(mixed, [0, 2, 1, 3]), [batch, seq, cfg.hidden]));
  return ctx.mark(`${id}.out`, conv1d(ctx, `${name}.c_proj`, merged, cfg.hidden, cfg.hidden));
}

function mlp(ctx: Ctx, name: string, id: string, x: Tensor, cfg: Gpt2Config): Tensor {
  const inner = cfg.mlpRatio * cfg.hidden;
  const widened = ctx.mark(`${id}.fc`, conv1d(ctx, `${name}.c_fc`, x, cfg.hidden, inner));
  const activated = ctx.mark(`${id}.act`, geluTanh(widened));
  return ctx.mark(`${id}.out`, conv1d(ctx, `${name}.c_proj`, activated, inner, cfg.hidden));
}

function block(ctx: Ctx, index: number, x: Tensor, cfg: Gpt2Config): Tensor {
  const name = `transformer.h.${index}`;
  const id = `h${index}`;
  const afterAttention = add(x, attention(ctx, `${name}.attn`, `${id}.attn`, norm(ctx, `${name}.ln_1`, x, cfg), cfg));
  const afterMlp = add(
    afterAttention,
    mlp(ctx, `${name}.mlp`, `${id}.mlp`, norm(ctx, `${name}.ln_2`, afterAttention, cfg), cfg),
  );
  return ctx.mark(`${id}.out`, afterMlp);
}

/** Token ids [B, S] to next-token logits [B, S, vocab]. */
export function forward(ctx: Ctx, cfg: Gpt2Config, ids: readonly (readonly number[])[]): Tensor {
  const seq = ids[0].length;
  const wte = ctx.param("transformer.wte.weight", [cfg.vocab, cfg.hidden]);
  const wpe = ctx.param("transformer.wpe.weight", [cfg.positions, cfg.hidden]);
  const positions = ids.map(() => Array.from({ length: seq }, (_unused, position) => position));

  let hidden = ctx.mark("embed", add(embedding(wte, ids), embedding(wpe, positions)));
  for (let index = 0; index < cfg.layers; index += 1) {
    hidden = block(ctx, index, hidden, cfg);
  }
  hidden = ctx.mark("final_norm", norm(ctx, "transformer.ln_f", hidden, cfg));
  // The head has no weights of its own: it is the token embedding, used as [vocab, hidden].
  return ctx.mark("logits", affine(hidden, wte, null, "out_in"));
}
