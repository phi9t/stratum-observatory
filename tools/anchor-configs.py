"""Propose json_pointer anchors for MODELS.* citations of publisher configuration.

Writes a candidate bundle only. Every anchor is resolved against the cached bytes before it
is written; an anchor that does not resolve is reported as residue and never emitted.

POINTERS is authored by MEANING, not by value equality. A MODELS field whose value is
inherited from the `common` defaults block in src/models.mjs asserts the ABSENCE of a
mechanism in that checkpoint; such a field never gets an entry here, no matter what a
publisher config happens to contain at that key. See
.superpowers/sdd/2026-09-09-evidence-anchors-and-dependency-provenance/task-2-authored-pointers.md
for the per-entry justification and the value-equality-coincidence hazard it documents.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.evidence import canonical, loads, safe_path, select_anchor

# (repo, MODELS field) -> JSON pointer in that publisher's config.json.
# Keyed by repo because the same fact has different upstream names per family, and because
# each repo in this manifest names exactly one checkpoint (so this is also, in effect, keyed
# per model). Every entry below is either a field with no "common" default at all (an always
# model-specific fact: d, layers, heads, vocab, ff, total, configMax, scoring, ...) or a field
# whose value in that specific model OVERRIDES the common default (verified against
# src/models.mjs before being added) -- never a field left at the inherited default.
POINTERS = {}

# --- DeepSeek-V2 / DeepSeek-V3: flat config, pointers are top-level. ---
for _repo in ('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V3'):
    POINTERS.update({
        (_repo, 'd'): '/hidden_size',
        (_repo, 'layers'): '/num_hidden_layers',
        (_repo, 'heads'): '/num_attention_heads',
        (_repo, 'vocab'): '/vocab_size',
        (_repo, 'configMax'): '/max_position_embeddings',
        (_repo, 'kvRank'): '/kv_lora_rank',
        (_repo, 'qRank'): '/q_lora_rank',
        (_repo, 'nopeDim'): '/qk_nope_head_dim',
        (_repo, 'ropeDim'): '/qk_rope_head_dim',
        (_repo, 'denseLayers'): '/first_k_dense_replace',
        (_repo, 'shared'): '/n_shared_experts',
        (_repo, 'experts'): '/n_routed_experts',
        (_repo, 'topK'): '/num_experts_per_tok',
        (_repo, 'ff'): '/intermediate_size',
        (_repo, 'expertFF'): '/moe_intermediate_size',
        (_repo, 'scoring'): '/scoring_func',
    })
# DeepSeek-V3 alone overrides mtp (=1); DeepSeek-V2 leaves it at the common default (0), so
# DeepSeek-V2 gets no mtp entry -- an unmapped field, not a zero.
POINTERS[('deepseek-ai/DeepSeek-V3', 'mtp')] = '/num_nextn_predict_layers'

# --- Kimi K3: everything real lives under /text_config/. ---
_k3 = 'moonshotai/Kimi-K3'
POINTERS.update({
    (_k3, 'd'): '/text_config/hidden_size',
    (_k3, 'layers'): '/text_config/num_hidden_layers',
    (_k3, 'heads'): '/text_config/num_attention_heads',
    (_k3, 'kvHeads'): '/text_config/num_key_value_heads',
    (_k3, 'vocab'): '/text_config/vocab_size',
    (_k3, 'kvRank'): '/text_config/kv_lora_rank',
    (_k3, 'qRank'): '/text_config/q_lora_rank',
    (_k3, 'nopeDim'): '/text_config/qk_nope_head_dim',
    (_k3, 'ropeDim'): '/text_config/qk_rope_head_dim',
    (_k3, 'denseLayers'): '/text_config/first_k_dense_replace',
    (_k3, 'expertFF'): '/text_config/moe_intermediate_size',
    (_k3, 'ff'): '/text_config/intermediate_size',
    (_k3, 'vDim'): '/text_config/v_head_dim',
    (_k3, 'attnResBlock'): '/text_config/attn_res_block_size',
    (_k3, 'shared'): '/text_config/num_shared_experts',
    (_k3, 'experts'): '/text_config/num_experts',
    (_k3, 'topK'): '/text_config/num_experts_per_token',
    (_k3, 'gate'): '/text_config/mla_use_output_gate',
    (_k3, 'scoring'): '/text_config/moe_router_activation_func',
    (_k3, 'latent'): '/text_config/routed_expert_hidden_size',
    (_k3, 'maxContext'): '/text_config/max_position_embeddings',
    # Kimi K3 declares one head count and one head width for the whole KDA path, inside
    # linear_attn_config; the short-conv width sits in the same block.
    (_k3, 'linearHeadDim'): '/text_config/linear_attn_config/head_dim',
    (_k3, 'convKernel'): '/text_config/linear_attn_config/short_conv_kernel_size',
})

# --- Qwen3 dense/MoE family: flat config. ---
for _repo in ('Qwen/Qwen3-8B', 'Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-235B-A22B'):
    POINTERS.update({
        (_repo, 'd'): '/hidden_size',
        (_repo, 'layers'): '/num_hidden_layers',
        (_repo, 'heads'): '/num_attention_heads',
        (_repo, 'kvHeads'): '/num_key_value_heads',
        (_repo, 'headDim'): '/head_dim',
        (_repo, 'vocab'): '/vocab_size',
        (_repo, 'ff'): '/intermediate_size',
    })
for _repo in ('Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-235B-A22B'):
    POINTERS.update({
        (_repo, 'experts'): '/num_experts',
        (_repo, 'topK'): '/num_experts_per_tok',
        (_repo, 'expertFF'): '/moe_intermediate_size',
    })

# RULING (L/N) refinement, applied in Task 4. A `common` default that is a GENUINE VALUE may be
# anchored wherever an upstream key names the same quantity; the test is semantic, never whether
# the value arrived via `common`. transformers' Qwen3Attention sizes the q, k AND v projections
# from the single `head_dim` key, so that key names the V head dimension as much as the Q one.
for _repo in ('Qwen/Qwen3-8B', 'Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-235B-A22B'):
    POINTERS[(_repo, 'vDim')] = '/head_dim'

# --- Qwen3.8-27B and Qwen3.8-Flash-Next: everything real lives under /text_config/. ---
_q3827 = 'Qwen/Qwen3.8-27B'
POINTERS.update({
    (_q3827, 'd'): '/text_config/hidden_size',
    (_q3827, 'layers'): '/text_config/num_hidden_layers',
    (_q3827, 'heads'): '/text_config/num_attention_heads',
    (_q3827, 'kvHeads'): '/text_config/num_key_value_heads',
    (_q3827, 'headDim'): '/text_config/head_dim',
    (_q3827, 'vocab'): '/text_config/vocab_size',
    (_q3827, 'ff'): '/text_config/intermediate_size',
    (_q3827, 'globalEvery'): '/text_config/full_attention_interval',
    (_q3827, 'linearHeads'): '/text_config/linear_num_value_heads',
    (_q3827, 'linearKeyHeads'): '/text_config/linear_num_key_heads',
    (_q3827, 'gate'): '/text_config/attn_output_gate',
    (_q3827, 'linearGate'): '/text_config/output_gate_type',
    (_q3827, 'mtp'): '/text_config/mtp_num_hidden_layers',
    (_q3827, 'maxContext'): '/text_config/max_position_embeddings',
    # RULING (L/N) refinement: genuine values that happen to equal the `common` default, each
    # named outright by Qwen3.8's own configuration.
    (_q3827, 'vDim'): '/text_config/head_dim',
    (_q3827, 'linearHeadDim'): '/text_config/linear_value_head_dim',
    (_q3827, 'convKernel'): '/text_config/linear_conv_kernel_dim',
})

_q38flash = 'Qwen/Qwen3.8-Flash-Next'
POINTERS.update({
    (_q38flash, 'd'): '/text_config/hidden_size',
    (_q38flash, 'layers'): '/text_config/num_hidden_layers',
    (_q38flash, 'heads'): '/text_config/num_attention_heads',
    (_q38flash, 'kvHeads'): '/text_config/num_key_value_heads',
    (_q38flash, 'headDim'): '/text_config/head_dim',
    (_q38flash, 'vocab'): '/text_config/vocab_size',
    (_q38flash, 'experts'): '/text_config/num_experts',
    (_q38flash, 'topK'): '/text_config/num_experts_per_tok',
    (_q38flash, 'expertFF'): '/text_config/moe_intermediate_size',
    (_q38flash, 'globalEvery'): '/text_config/full_attention_interval',
    (_q38flash, 'linearHeads'): '/text_config/linear_num_value_heads',
    (_q38flash, 'linearKeyHeads'): '/text_config/linear_num_key_heads',
    (_q38flash, 'linearGate'): '/text_config/output_gate_type',
    (_q38flash, 'indexer'): '/text_config/indexer_head_dim',
    (_q38flash, 'indexerBudget'): '/text_config/indexer_budget',
    (_q38flash, 'indexerCompression'): '/text_config/indexer_compress_ratio',
    (_q38flash, 'indexerHeads'): '/text_config/indexer_n_heads',
    (_q38flash, 'streams'): '/text_config/hc_count',
    (_q38flash, 'mixRank'): '/text_config/hc_lowrank',
    (_q38flash, 'pleLayers'): '/text_config/ple_layer_ids',
    (_q38flash, 'maxContext'): '/text_config/max_position_embeddings',
    (_q38flash, 'vDim'): '/text_config/head_dim',
    (_q38flash, 'linearHeadDim'): '/text_config/linear_value_head_dim',
    (_q38flash, 'convKernel'): '/text_config/linear_conv_kernel_dim',
})


def propose(target: dict, sources: dict, root: Path) -> tuple[dict, list[dict]]:
    """Return (new_target, residue). Never writes an anchor that does not resolve."""
    out = copy.deepcopy(target)
    residue: list[dict] = []
    for claim in out.get('claims', []):
        cid = claim.get('id', '')
        if not cid.startswith('MODELS.'):
            continue
        field = cid.split('.')[-1]
        for entry in claim.get('evidence', []):
            if entry.get('anchor'):
                continue
            source = sources.get(entry.get('source'))
            if not source or source.get('role') != 'publisher_config':
                continue
            row = {'claim': cid, 'source': source['id'], 'expected': claim.get('expected'),
                   'pointer': None, 'upstream': None, 'reason': ''}
            pointer = POINTERS.get((source.get('repo'), field))
            if not pointer:
                row['reason'] = 'no mapped pointer'
                residue.append(row)
                continue
            row['pointer'] = pointer
            anchor = {'kind': 'json_pointer', 'pointer': pointer}
            try:
                raw = safe_path(root, source['cache']).read_bytes()
                selected, _ = select_anchor(raw, anchor)
            except (ValueError, KeyError, OSError):
                row['reason'] = 'pointer does not resolve'
                residue.append(row)
                continue
            row['upstream'] = selected
            anchor['sha256'] = hashlib.sha256(selected.encode()).hexdigest()
            entry['anchor'] = anchor
            if canonical(loads(selected)) == canonical(claim.get('expected')):
                entry['relation'] = 'equal'
            else:
                row['reason'] = 'upstream value differs from expected'
                residue.append(row)
    return out, residue


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--residue', required=True)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args(argv)
    root = Path(a.root).resolve()
    target = loads((root/'verification/manifests/observatory.json').read_text())
    catalog = loads((root/'verification/manifests/sources.json').read_text())
    sources = {s['id']: s for s in catalog['sources']}
    new, residue = propose(target, sources, root)
    candidate = safe_path(root, a.candidate)
    if candidate.exists():
        raise SystemExit('Candidate already exists; use a fresh candidate path')
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(
        {'schema': 'stratum-acquisition-candidate/v1', 'catalog': catalog, 'target': new,
         'results': [], 'review_policy': 'Candidate is NOT an accepted lock. Read every proposed '
                                         'pointer against the cached bytes before promoting.'},
        indent=2) + '\n')
    safe_path(root, a.residue).write_text(json.dumps(residue, indent=2) + '\n')
    print(f'{a.candidate}: proposed anchors; {len(residue)} residue rows -> {a.residue}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
