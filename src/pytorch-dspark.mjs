// Educational adaptations of DeepSeek V4.1 Flash and DeepSpec (MIT).
export const DSPARK_EXAMPLES = [
  {
    id:'dspark-attention', title:'Draft-block visibility and attention sink', stages:['dspark'],
    sources:['dsv41-code','dsv41-kernel'], symbols:['DSparkAttention.forward','get_dspark_topk_idxs','sparse_attn','sparse_attn_kernel'],
    summary:'A dense PyTorch rendering of the sparse attention contraction: each draft slot reads the valid captured-history window and every slot in the current draft block. The learned sink contributes softmax mass with no value vector.',
    limitations:'Inputs are already projected, RMS-normalized and RoPE-rotated. History contains only valid captured target slots, in any common key/value order. This isolates the contraction: ring-buffer writes, FP8 quantization, inverse output RoPE, grouped output projection and the surrounding mHC block are omitted. It is not a complete draft layer.',
    requires:[],
    code:`# Adapted from DeepSeek V4.1 Flash, MIT; Copyright (c) 2023 DeepSeek.
import torch


def dspark_projected_attention(q, history_kv, draft_kv, sink, scale):
    # q: [B, K, H, D]; history_kv: [B, L, D]; draft_kv: [B, K, D].
    # Shared latent vectors serve as both keys and values.
    # L contains only valid captured-history slots; no future target tokens.
    kv = torch.cat((history_kv, draft_kv), dim=1)
    scores = torch.einsum('bkhd,bnd->bkhn', q.float(), kv.float()) * scale
    sink_score = sink.float().view(1, 1, -1, 1).expand(*scores.shape[:-1], 1)
    weights = torch.softmax(torch.cat((scores, sink_score), dim=-1), dim=-1)
    # No triangular mask: even the first draft slot sees the last draft slot.
    output = torch.einsum('bkhn,bnd->bkhd', weights[..., :-1], kv.float())
    return output.to(q.dtype)
`
  },
  {
    id:'dspark-heads', title:'Sequential Markov correction and raw confidence', stages:['dspark'],
    sources:['dsv41-code'], symbols:['DSparkMarkovHead','DSparkConfidenceHead','DSparkBlock.forward_head'],
    summary:'The backbone produces draft positions together. Vocabulary sampling then proceeds sequentially: each position adds a learned bias from the preceding sampled token. Confidence combines the unnormalized collapsed residual with that preceding-token embedding and remains a raw logit.',
    limitations:'Provide collapsed residuals before final RMSNorm and base logits after final RMSNorm plus the shared vocabulary head. This module isolates the output heads; it omits mHC collapse, backbone execution and distributed vocabulary sharding. Sampling uses a positive temperature and PyTorch multinomial instead of the publisher sampler. No checkpoint dimensions or weights are supplied.',
    requires:[],
    code:`# Adapted from DeepSeek V4.1 Flash, MIT; Copyright (c) 2023 DeepSeek.
import torch
from torch import nn


class DSparkOutputHeads(nn.Module):
    def __init__(self, hidden_dim, vocab_size, markov_rank):
        super().__init__()
        self.markov_embed = nn.Embedding(vocab_size, markov_rank)
        self.markov_head = nn.Linear(markov_rank, vocab_size, bias=False)
        self.confidence = nn.Linear(hidden_dim + markov_rank, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden, base_logits, seed, temperature):
        # hidden: [B, K, D], before final norm; base_logits: [B, K, V].
        # seed: [B]. The returned token sequence includes this seed.
        assert temperature > 0
        tokens, logits, embeddings = [seed], [], []
        for position in range(hidden.shape[1]):
            previous = self.markov_embed(tokens[-1])
            corrected = base_logits[:, position] + self.markov_head(previous)
            probs = torch.softmax(corrected.float() / temperature, dim=-1)
            tokens.append(torch.multinomial(probs, 1).squeeze(-1))
            logits.append(corrected)
            embeddings.append(previous)
        markov = torch.stack(embeddings, dim=1)
        confidence_input = torch.cat((hidden, markov), dim=-1).float()
        raw_confidence = self.confidence(confidence_input).squeeze(-1)
        return torch.stack(tokens, dim=1), torch.stack(logits, dim=1), raw_confidence
`
  },
  {
    id:'deepspec-verifier', title:'DeepSpec: prefix verification and reusable state', stages:['dspark'],
    sources:['deepspec-verification','deepspec-draft','deepspec-context','deepspec-sampling'],
    symbols:['verify_draft_tokens','sample_residual','_confident_prefix_length','forward_dspark_draft_block','Qwen3DSparkEvaluator._update','generate_decoding_sample'],
    summary:'The separate DeepSpec evaluator accepts a contiguous prefix by the target/draft probability ratio, samples a correction after rejection or a bonus after full acceptance, and discards speculative state. Its confidence threshold is a fixed offline policy.',
    limitations:'This is DeepSpec’s single-sequence Qwen3/Gemma4 evaluator mechanism, not a published V4.1 production verifier or the paper’s dynamic production scheduler. Probability inputs are already normalized at the chosen temperature; target forward execution is outside this example. It shows a nonterminal round with no EOS/stop token, no output-budget truncation, and dense token-axis-first tensors in place of DynamicCache. Draft cropping occurs immediately after draft forward in DeepSpec; the helper groups the independent state-ownership results for inspection.',
    requires:[],
    code:`# Adapted from DeepSpec, MIT; Copyright (c) 2026 The DeepSpec Authors.
import torch


def deepspec_residual_probabilities(target, draft):
    residual = (target - draft).clamp_min(0)
    # Match the evaluator's fallback for vanishing residual mass.
    residual = torch.where(residual.sum() <= 1e-8, target, residual)
    return residual / residual.sum().clamp_min(1e-8)


def deepspec_verify_probabilities(proposed, draft, target, uniforms):
    # One sequence: proposed [K], draft [K, V], target [K+1, V].
    # Target row i predicts proposal i; its final row supplies the bonus.
    # uniforms [K] are independent U[0, 1) draws, passed in for traceability.
    count = proposed.numel()
    rows = torch.arange(count, device=proposed.device)
    ratio = target[rows, proposed] / draft[rows, proposed].clamp_min(1e-8)
    accepted_mask = (uniforms < ratio.clamp_max(1)).long().cumprod(dim=0)
    accepted = int(accepted_mask.sum())
    if accepted < count:
        next_probs = deepspec_residual_probabilities(target[accepted], draft[accepted])
    else:
        next_probs = target[-1]
    next_token = torch.multinomial(next_probs, 1)
    return torch.cat((proposed[:accepted], next_token)), accepted


def deepspec_confident_prefix(raw_confidence, threshold):
    # Positive fixed threshold; this is not the production scheduler.
    assert 0 < threshold <= 1
    confident = raw_confidence.sigmoid() >= threshold
    return int(confident.long().cumprod(dim=0).sum())


def deepspec_reuse_state(target_cache, draft_cache, verified_features, start, accepted):
    # Educational dense tensors, token axis first; start excludes the seed.
    # Target forward consumed seed + proposals. Keep seed + accepted prefix;
    # the correction/bonus token has not yet entered the target cache.
    target = target_cache[:start + accepted + 1].clone()
    # Draft forward's speculative slots are discarded at the old start.
    draft = draft_cache[:start].clone()
    # This round's verified target features: seed + accepted proposal tokens.
    features = verified_features[:accepted + 1].clone()
    return target, draft, features
`
  }
];
