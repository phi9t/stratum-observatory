// Distilled from Qwen3MoeModel, DecoderLayer, Attention, SparseMoeBlock, MLP,
// RMSNorm and RotaryEmbedding at the transformers commit pinned in ../sources.json.
// Full unpadded prefill to untied logits. Cache and serving payload are drawn
// separately; no sampling, losses, padding, quantization or GPU kernel execution.
// Float32 toy arithmetic; shape mode allocates no model weights.
import {
  add,
  affine,
  causalMask,
  type Ctx,
  embedding,
  matmul,
  mul,
  permute,
  reshape,
  rmsNorm,
  scale,
  silu,
  softmax,
  type Tensor,
  tensor,
  topk,
} from "../../../tnsr/tensor";
import { expertAssignments, gatherTokens, normalizeSelected, repeatKV, rotary, scatterWeighted } from "./ops";
export type QwenConfig = Readonly<{
  vocab: number;
  hidden: number;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  experts: number;
  topK: number;
  expertWidth: number;
  normEps: number;
  theta: number;
}>;
function linear(ctx: Ctx, name: string, x: Tensor, width: number): Tensor {
  return affine(x, ctx.param(`${name}.weight`, [width, x.shape[x.shape.length - 1]]), null, "out_in");
}
function Qwen3MoeRMSNorm(ctx: Ctx, name: string, x: Tensor, c: QwenConfig): Tensor {
  return rmsNorm(x, ctx.param(`${name}.weight`, [x.shape[x.shape.length - 1]]), c.normEps);
}
function Qwen3MoeAttention(ctx: Ctx, name: string, id: string, x: Tensor, c: QwenConfig): Tensor {
  const [b, s] = x.shape;
  const project = (part: string, h: number): Tensor =>
    ctx.mark(
      `${id}.${part}_heads`,
      reshape(ctx.mark(`${id}.${part}_proj`, linear(ctx, `${name}.${part}_proj`, x, h * c.headDim)), [
        b,
        s,
        h,
        c.headDim,
      ]),
    );
  const qNorm = ctx.mark(`${id}.q_norm`, Qwen3MoeRMSNorm(ctx, `${name}.q_norm`, project("q", c.heads), c));
  const kNorm = ctx.mark(`${id}.k_norm`, Qwen3MoeRMSNorm(ctx, `${name}.k_norm`, project("k", c.kvHeads), c));
  const v = ctx.mark(`${id}.v`, permute(project("v", c.kvHeads), [0, 2, 1, 3]));
  const q = ctx.mark(`${id}.q`, ctx.mark(`${id}.q_rope`, rotary(permute(qNorm, [0, 2, 1, 3]), c.theta)));
  const k = ctx.mark(`${id}.k`, ctx.mark(`${id}.k_rope`, rotary(permute(kNorm, [0, 2, 1, 3]), c.theta)));
  const kRep = ctx.mark(`${id}.repeat_k`, repeatKV(k, c.heads / c.kvHeads));
  const vRep = ctx.mark(`${id}.repeat_v`, repeatKV(v, c.heads / c.kvHeads));
  const dot = ctx.mark(`${id}.dot`, matmul(q, permute(kRep, [0, 1, 3, 2])));
  const scores = ctx.mark(`${id}.scores`, ctx.mark(`${id}.scale`, scale(dot, c.headDim ** -0.5)));
  const masked = ctx.mark(`${id}.mask`, causalMask(scores));
  const weights = ctx.mark(`${id}.weights`, softmax(masked));
  const mixed = ctx.mark(`${id}.mixed`, matmul(weights, vRep));
  const merged = ctx.mark(`${id}.merge`, reshape(permute(mixed, [0, 2, 1, 3]), [b, s, c.heads * c.headDim]));
  return ctx.mark(`${id}.out`, ctx.mark(`${id}.o_proj`, linear(ctx, `${name}.o_proj`, merged, c.hidden)));
}

function Qwen3MoeMLP(ctx: Ctx, name: string, id: string, x: Tensor, c: QwenConfig): Tensor {
  const gate = ctx.mark(`${id}.gate`, linear(ctx, `${name}.gate_proj`, x, c.expertWidth));
  const activated = ctx.mark(`${id}.silu`, silu(gate));
  const up = ctx.mark(`${id}.up`, linear(ctx, `${name}.up_proj`, x, c.expertWidth));
  const gated = ctx.mark(`${id}.mul`, mul(activated, up));
  return ctx.mark(`${id}.down`, linear(ctx, `${name}.down_proj`, gated, c.hidden));
}
/** Same router arithmetic used by the sparse block and the browser experiment. */
export function Qwen3MoeRouter(ctx: Ctx, name: string, id: string, x: Tensor, c: QwenConfig) {
  const logits = ctx.mark(`${id}.logits`, linear(ctx, name, x, c.experts));
  const scores = ctx.mark(`${id}.scores`, softmax(logits));
  const selected = topk(scores, c.topK);
  ctx.mark(`${id}.selected`, selected.indices);
  const weights = ctx.mark(`${id}.weights`, normalizeSelected(selected.values));
  return { logits, scores, selected, weights };
}
function Qwen3MoeSparseMoeBlock(ctx: Ctx, name: string, id: string, x: Tensor, c: QwenConfig): Tensor {
  const [b, s, d] = x.shape;
  const flat = ctx.mark(`${id}.flat`, reshape(x, [b * s, d]));
  const { selected, weights } = Qwen3MoeRouter(ctx, `${name}.gate`, id, flat, c);
  const out = ctx.mode === "shape" ? null : new Float32Array(b * s * d);
  for (let e = 0; e < c.experts; e++) {
    const { tokens, slots } = expertAssignments(selected.indices, e);
    // Empty experts still request their real parameters. No invented always-on path.
    // Without values, expert load is unknown. Use the all-token upper bound,
    // never interpret absent router choices as an empty expert workload.
    const current = ctx.mark(
      `${id}.e${e}.gather`,
      ctx.mode === "shape" ? { shape: [b * s, d], data: null } : gatherTokens(flat, tokens),
    );
    const update = Qwen3MoeMLP(ctx, `${name}.experts.${e}`, `${id}.e${e}`, current, c);
    if (out !== null) scatterWeighted(out, update, tokens, slots, weights);
  }
  return ctx.mark(`${id}.out`, reshape(tensor([b * s, d], out), [b, s, d]));
}
function Qwen3MoeDecoderLayer(ctx: Ctx, i: number, x: Tensor, c: QwenConfig): Tensor {
  const name = `model.layers.${i}`,
    id = `h${i}`;
  const normalized = ctx.mark(`${id}.input_norm`, Qwen3MoeRMSNorm(ctx, `${name}.input_layernorm`, x, c));
  const after = ctx.mark(
    `${id}.add_attn`,
    add(x, Qwen3MoeAttention(ctx, `${name}.self_attn`, `${id}.attn`, normalized, c)),
  );
  const post = ctx.mark(`${id}.post_norm`, Qwen3MoeRMSNorm(ctx, `${name}.post_attention_layernorm`, after, c));
  return ctx.mark(`${id}.out`, add(after, Qwen3MoeSparseMoeBlock(ctx, `${name}.mlp`, `${id}.moe`, post, c)));
}
/** All layers are MoE because mlp_only_layers=[] and decoder_sparse_step=1. */
export function forward(ctx: Ctx, c: QwenConfig, ids: readonly (readonly number[])[]): Tensor {
  let x = ctx.mark("embed", embedding(ctx.param("model.embed_tokens.weight", [c.vocab, c.hidden]), ids));
  for (let i = 0; i < c.layers; i++) x = Qwen3MoeDecoderLayer(ctx, i, x, c);
  x = ctx.mark("final_norm", Qwen3MoeRMSNorm(ctx, "model.norm", x, c));
  return ctx.mark("logits", linear(ctx, "lm_head", x, c.vocab));
}
