// Shape-only clean reference for GLM-5.3-Flash.
//
// Scope: exact stored-tensor inventory, layer schedule and major tensor shapes.
// It deliberately does not run upstream numerical parity: that would require
// loading model weights, which this schematic task forbids.
import { Ctx, type Tensor } from "../../../tnsr/tensor";
import * as f from "../facts";

export type Config = {
  hidden: number;
  vocab: number;
  layers: number;
  nextn: number;
  heads: number;
  qRank: number;
  kvRank: number;
  qkDim: number;
  nopeDim: number;
  valueDim: number;
  denseIntermediate: number;
  moeIntermediate: number;
  experts: number;
  sharedExperts: number;
  linearHeads: number;
  linearHeadDim: number;
  convKernel: number;
  indexHeads: number;
  indexHeadDim: number;
  indexTopK: number;
  indexKpool: number;
  hcMult: number;
  visionDepth: number;
  visionHidden: number;
  visionHeads: number;
  visionPatch: number;
  visionTemporalPatch: number;
  visionChannels: number;
  visionIntermediate: number;
  visionOut: number;
  visionProjectionIntermediate: number;
  visionMerge: number;
};

export const REAL: Config = {
  hidden: f.hidden.value,
  vocab: f.vocab.value,
  layers: f.layers.value,
  nextn: f.nextn.value,
  heads: f.heads.value,
  qRank: f.qRank.value,
  kvRank: f.kvRank.value,
  qkDim: f.qkDim.value,
  nopeDim: f.nopeDim.value,
  valueDim: f.valueDim.value,
  denseIntermediate: f.denseIntermediate.value,
  moeIntermediate: f.moeIntermediate.value,
  experts: f.experts.value,
  sharedExperts: f.sharedExperts.value,
  linearHeads: f.linearHeads.value,
  linearHeadDim: f.linearHeadDim.value,
  convKernel: f.convKernel.value,
  indexHeads: f.indexHeads.value,
  indexHeadDim: f.indexHeadDim.value,
  indexTopK: f.indexTopK.value,
  indexKpool: f.indexKpool.value,
  hcMult: f.hcMult.value,
  visionDepth: f.visionDepth.value,
  visionHidden: f.visionHidden.value,
  visionHeads: f.visionHeads.value,
  visionPatch: f.visionPatch.value,
  visionTemporalPatch: f.visionTemporalPatch.value,
  visionChannels: f.visionChannels.value,
  visionIntermediate: f.visionIntermediate.value,
  visionOut: f.visionOut.value,
  visionProjectionIntermediate: f.visionProjectionIntermediate.value,
  visionMerge: f.visionMerge.value,
};

export const TOY: Config = {
  ...REAL,
  hidden: 16,
  vocab: 32,
  layers: 5,
  heads: 4,
  qRank: 8,
  kvRank: 4,
  qkDim: 8,
  nopeDim: 8,
  valueDim: 8,
  denseIntermediate: 24,
  moeIntermediate: 8,
  experts: 6,
  linearHeads: 4,
  linearHeadDim: 4,
  indexHeads: 2,
  indexHeadDim: 4,
  indexTopK: 8,
  hcMult: 4,
  visionDepth: 2,
  visionHidden: 8,
  visionHeads: 2,
  visionIntermediate: 16,
  visionOut: 16,
  visionProjectionIntermediate: 24,
};

const shape = (...dims: readonly number[]): Tensor => ({ shape: [...dims], data: null });

function p(ctx: Ctx, name: string, dims: readonly number[]): void {
  ctx.param(name, dims);
}

function addDenseMlp(ctx: Ctx, c: Config, prefix: string): void {
  p(ctx, `${prefix}.gate_proj.weight`, [c.denseIntermediate, c.hidden]);
  p(ctx, `${prefix}.up_proj.weight`, [c.denseIntermediate, c.hidden]);
  p(ctx, `${prefix}.down_proj.weight`, [c.hidden, c.denseIntermediate]);
}

function addExpertMlp(ctx: Ctx, c: Config, prefix: string, count: number): void {
  p(ctx, `${prefix}.experts.gate_up_proj`, [count, 2 * c.moeIntermediate, c.hidden]);
  p(ctx, `${prefix}.experts.down_proj`, [count, c.hidden, c.moeIntermediate]);
}

function addSharedExpert(ctx: Ctx, c: Config, prefix: string): void {
  const width = c.sharedExperts * c.moeIntermediate;
  p(ctx, `${prefix}.shared_experts.gate_proj.weight`, [width, c.hidden]);
  p(ctx, `${prefix}.shared_experts.up_proj.weight`, [width, c.hidden]);
  p(ctx, `${prefix}.shared_experts.down_proj.weight`, [c.hidden, width]);
}

function addMoE(ctx: Ctx, c: Config, prefix: string): void {
  p(ctx, `${prefix}.gate.weight`, [c.experts, c.hidden]);
  p(ctx, `${prefix}.gate.e_score_correction_bias`, [c.experts]);
  addExpertMlp(ctx, c, prefix, c.experts);
  addSharedExpert(ctx, c, prefix);
}

function addKda(ctx: Ctx, c: Config, prefix: string): void {
  const wide = c.linearHeads * c.linearHeadDim;
  p(ctx, `${prefix}.q_proj.weight`, [wide, c.hidden]);
  p(ctx, `${prefix}.k_proj.weight`, [wide, c.hidden]);
  p(ctx, `${prefix}.v_proj.weight`, [wide, c.hidden]);
  p(ctx, `${prefix}.conv1d.weight`, [3 * wide, 1, c.convKernel]);
  p(ctx, `${prefix}.forget_gate.f_a_proj.weight`, [c.linearHeadDim, c.hidden]);
  p(ctx, `${prefix}.forget_gate.f_b_proj.weight`, [wide, c.linearHeadDim]);
  p(ctx, `${prefix}.forget_gate.dt_bias`, [wide]);
  p(ctx, `${prefix}.forget_gate.A_log`, [c.linearHeads]);
  p(ctx, `${prefix}.b_proj.weight`, [c.linearHeads, c.hidden]);
  p(ctx, `${prefix}.g_a_proj.weight`, [c.linearHeadDim, c.hidden]);
  p(ctx, `${prefix}.g_b_proj.weight`, [wide, c.linearHeadDim]);
  p(ctx, `${prefix}.o_norm.weight`, [c.linearHeadDim]);
  p(ctx, `${prefix}.o_proj.weight`, [c.hidden, wide]);
}

function addSparseMla(ctx: Ctx, c: Config, prefix: string): void {
  p(ctx, `${prefix}.q_a_proj.weight`, [c.qRank, c.hidden]);
  p(ctx, `${prefix}.q_a_layernorm.weight`, [c.qRank]);
  p(ctx, `${prefix}.q_b_proj.weight`, [c.heads * c.qkDim, c.qRank]);
  p(ctx, `${prefix}.kv_a_proj_with_mqa.weight`, [c.kvRank, c.hidden]);
  p(ctx, `${prefix}.kv_a_layernorm.weight`, [c.kvRank]);
  p(ctx, `${prefix}.kv_b_proj.weight`, [c.heads * (c.nopeDim + c.valueDim), c.kvRank]);
  p(ctx, `${prefix}.o_proj.weight`, [c.hidden, c.heads * c.valueDim]);
}

function addIndexer(ctx: Ctx, c: Config, prefix: string): void {
  p(ctx, `${prefix}.indexer.wq_b.weight`, [c.indexHeads * c.indexHeadDim, c.qRank]);
  p(ctx, `${prefix}.indexer.wk.weight`, [c.indexHeadDim, c.hidden]);
  p(ctx, `${prefix}.indexer.k_norm.weight`, [c.indexHeadDim]);
  p(ctx, `${prefix}.indexer.k_norm.bias`, [c.indexHeadDim]);
  p(ctx, `${prefix}.indexer.weights_proj.weight`, [c.indexHeads, c.hidden]);
  p(ctx, `${prefix}.indexer.index_kpool_compress_gate`, [c.indexHeadDim, c.hidden]);
  p(ctx, `${prefix}.indexer.index_kpool_compress_ape`, [c.indexKpool, c.indexHeadDim]);
}

function addMhc(ctx: Ctx, c: Config, prefix: string): void {
  const mix = (2 + c.hcMult) * c.hcMult;
  const stream = c.hcMult * c.hidden;
  for (const side of ["attn_hc", "ffn_hc"]) {
    p(ctx, `${prefix}.${side}.fn`, [mix, stream]);
    p(ctx, `${prefix}.${side}.base`, [mix]);
    p(ctx, `${prefix}.${side}.scale`, [3]);
  }
}

function addLayer(ctx: Ctx, c: Config, layer: number): void {
  const prefix = `model.language_model.layers.${layer}`;
  p(ctx, `${prefix}.input_layernorm.weight`, [c.hidden]);
  p(ctx, `${prefix}.post_attention_layernorm.weight`, [c.hidden]);
  addMhc(ctx, c, prefix);
  if (f.isSparseAttentionLayer(layer)) {
    addSparseMla(ctx, c, `${prefix}.self_attn`);
    addIndexer(ctx, c, `${prefix}.self_attn`);
  } else {
    addKda(ctx, c, `${prefix}.self_attn`);
  }
  if (f.isMoELayer(layer)) {
    addMoE(ctx, c, `${prefix}.mlp`);
  } else {
    addDenseMlp(ctx, c, `${prefix}.mlp`);
  }
}

function addMtp(ctx: Ctx, c: Config): void {
  for (let i = 0; i < c.nextn; i += 1) {
    const prefix = `model.language_model.mtp.layers.${i}`;
    p(ctx, `${prefix}.enorm.weight`, [c.hidden]);
    p(ctx, `${prefix}.hnorm.weight`, [c.hidden]);
    p(ctx, `${prefix}.eh_proj.weight`, [c.hidden, 2 * c.hidden]);
    p(ctx, `${prefix}.input_layernorm.weight`, [c.hidden]);
    p(ctx, `${prefix}.post_attention_layernorm.weight`, [c.hidden]);
    addSparseMla(ctx, c, `${prefix}.self_attn`);
    addIndexer(ctx, c, `${prefix}.self_attn`);
    addMoE(ctx, c, `${prefix}.mlp`);
    p(ctx, `${prefix}.shared_head.norm.weight`, [c.hidden]);
  }
}

function addVision(ctx: Ctx, c: Config): void {
  p(ctx, "model.visual.patch_embed.proj.weight", [
    c.visionHidden,
    c.visionChannels,
    c.visionTemporalPatch,
    c.visionPatch,
    c.visionPatch,
  ]);
  p(ctx, "model.visual.patch_embed.proj.bias", [c.visionHidden]);
  const headDim = c.visionHidden / c.visionHeads;
  for (let i = 0; i < c.visionDepth; i += 1) {
    const prefix = `model.visual.blocks.${i}`;
    p(ctx, `${prefix}.norm1.weight`, [c.visionHidden]);
    p(ctx, `${prefix}.norm2.weight`, [c.visionHidden]);
    p(ctx, `${prefix}.attn.qkv.weight`, [3 * c.visionHidden, c.visionHidden]);
    p(ctx, `${prefix}.attn.qkv.bias`, [3 * c.visionHidden]);
    p(ctx, `${prefix}.attn.proj.weight`, [c.visionHidden, c.visionHidden]);
    p(ctx, `${prefix}.attn.proj.bias`, [c.visionHidden]);
    p(ctx, `${prefix}.attn.q_norm.weight`, [headDim]);
    p(ctx, `${prefix}.attn.k_norm.weight`, [headDim]);
    p(ctx, `${prefix}.mlp.gate_proj.weight`, [c.visionIntermediate, c.visionHidden]);
    p(ctx, `${prefix}.mlp.gate_proj.bias`, [c.visionIntermediate]);
    p(ctx, `${prefix}.mlp.up_proj.weight`, [c.visionIntermediate, c.visionHidden]);
    p(ctx, `${prefix}.mlp.up_proj.bias`, [c.visionIntermediate]);
    p(ctx, `${prefix}.mlp.down_proj.weight`, [c.visionHidden, c.visionIntermediate]);
    p(ctx, `${prefix}.mlp.down_proj.bias`, [c.visionHidden]);
  }
  p(ctx, "model.visual.post_layernorm.weight", [c.visionHidden]);
  p(ctx, "model.visual.downsample.weight", [c.visionOut, c.visionHidden, c.visionMerge, c.visionMerge]);
  p(ctx, "model.visual.downsample.bias", [c.visionOut]);
  p(ctx, "model.visual.merger.proj.weight", [c.visionOut, c.visionOut]);
  p(ctx, "model.visual.merger.post_projection_norm.weight", [c.visionOut]);
  p(ctx, "model.visual.merger.post_projection_norm.bias", [c.visionOut]);
  p(ctx, "model.visual.merger.gate_proj.weight", [c.visionProjectionIntermediate, c.visionOut]);
  p(ctx, "model.visual.merger.up_proj.weight", [c.visionProjectionIntermediate, c.visionOut]);
  p(ctx, "model.visual.merger.down_proj.weight", [c.visionOut, c.visionProjectionIntermediate]);
}

export function enumerateStoredTensors(ctx: Ctx, c: Config = REAL): void {
  p(ctx, "model.language_model.embed_tokens.weight", [c.vocab, c.hidden]);
  p(ctx, "lm_head.weight", [c.vocab, c.hidden]);
  for (let layer = 0; layer < c.layers; layer += 1) {
    addLayer(ctx, c, layer);
  }
  p(ctx, "model.language_model.norm.weight", [c.hidden]);
  addMtp(ctx, c);
  addVision(ctx, c);
}

export function forwardShape(ctx: Ctx, c: Config = REAL, batch = 1, sequence = 12): Tensor {
  enumerateStoredTensors(ctx, c);
  ctx.mark("input_ids", shape(batch, sequence));
  ctx.mark("token_embeddings", shape(batch, sequence, c.hidden));
  ctx.mark("mhc_streams", shape(batch, sequence, c.hcMult, c.hidden));
  for (const layer of f.sparseLayers.value) {
    ctx.mark(`layer.${layer}.index_scores`, shape(batch, sequence, c.indexHeads, Math.ceil(sequence / c.indexKpool)));
    ctx.mark(`layer.${layer}.topk`, shape(batch, sequence, c.indexTopK + c.indexKpool - 1));
  }
  ctx.mark("mtp.hidden", shape(batch, sequence, c.hidden));
  const logits = shape(batch, sequence, c.vocab);
  return ctx.mark("logits", logits);
}
