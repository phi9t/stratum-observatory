// Educational adaptations of DeepSeek V4.1 Flash and DeepSpec (MIT).
export const DSPARK_EXAMPLES = [
  {
    id:'dspark-attention', title:'Draft-block visibility and attention sink', stages:['dspark'],
    sources:['dsv41-code','dsv41-kernel'], symbols:['DSparkAttention.forward','get_dspark_topk_idxs','sparse_attn','sparse_attn_kernel'],
    summary:'A dense PyTorch rendering of the sparse attention contraction: each draft slot reads the valid captured-history window and every slot in the current draft block. The learned sink contributes softmax mass with no value vector.',
    limitations:'Inputs are already projected, RMS-normalized and RoPE-rotated. History contains only valid captured target slots, in any common key/value order. This isolates the contraction: ring-buffer writes, FP8 quantization, inverse output RoPE, grouped output projection and the surrounding mHC block are omitted. It is not a complete draft layer.',
    requires:[],
    code:`# Adapted from DeepSeek V4.1 Flash, MIT; Copyright (c) 2023 DeepSeek.
import math
import torch
from torch import Tensor
from typing import Annotated


def dspark_projected_attention(
    queries: Annotated[Tensor, "batch draft_positions heads head_dim"],
    history_key_values: Annotated[Tensor, "batch history_positions head_dim"],
    draft_key_values: Annotated[Tensor, "batch draft_positions head_dim"],
    sink: Annotated[Tensor, "heads"],
    attention_scale: float,
) -> Annotated[Tensor, "batch draft_positions heads head_dim"]:
    """Read valid history and every draft slot; return [batch, draft, heads, width].

    queries: [batch, draft, heads, width], history_key_values: [batch, history, width],
    draft_key_values: [batch, draft, width], sink: [heads].
    Inputs are projected/rotated latents. Keys and values share the same vectors.
    """
    if queries.ndim != 4 or history_key_values.ndim != 3:
        raise ValueError("queries must be rank four and history rank three")
    batch, draft_length, heads, width = queries.shape
    if draft_key_values.shape != (batch, draft_length, width):
        raise ValueError("draft key/value shape must match query batch, length and width")
    history_shape = (history_key_values.shape[0], history_key_values.shape[-1])
    if history_shape != (batch, width) or sink.shape != (heads,):
        raise ValueError("history width/batch or sink head count does not match queries")
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("attention_scale must be finite and positive")
    # Keep attention accumulation FP32 even under an outer autocast context.
    with torch.autocast(device_type=queries.device.type, enabled=False):
        key_values = torch.cat((history_key_values, draft_key_values), dim=1).float()
        # b=batch, k=draft query, h=head, d=width, n=key position.
        # Contract d: each query/head scores every retained key.
        scores = torch.einsum("bkhd,bnd->bkhn", queries.float(), key_values)
        scores = scores * attention_scale
        # Broadcast one sink per head across batch and draft positions.
        sink_scores = sink.float().view(1, 1, heads, 1).expand(*scores.shape[:-1], 1)
        probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)
        # No triangular mask: the first draft slot sees the final draft slot.
        # Contract key axis n; remove the sink, which owns no value vector.
        output = torch.einsum("bkhn,bnd->bkhd", probabilities[..., :-1], key_values)
    return output.to(queries.dtype)
`
  },
  {
    id:'dspark-heads', title:'Sequential Markov correction and raw confidence', stages:['dspark'],
    sources:['dsv41-code'], symbols:['DSparkMarkovHead','DSparkConfidenceHead','DSparkBlock.forward_head'],
    summary:'The backbone produces draft positions together. Vocabulary sampling then proceeds sequentially: each position adds a learned bias from the preceding sampled token. Confidence combines the unnormalized collapsed residual with that preceding-token embedding and remains a raw logit.',
    limitations:'Provide collapsed residuals before final RMSNorm and base logits after final RMSNorm plus the shared vocabulary head. This module isolates the output heads; it omits mHC collapse, backbone execution and distributed vocabulary sharding. Sampling uses a positive temperature and PyTorch multinomial instead of the publisher sampler. No checkpoint dimensions or weights are supplied.',
    requires:[],
    code:`# Adapted from DeepSeek V4.1 Flash, MIT; Copyright (c) 2023 DeepSeek.
import math
import torch
from torch import Tensor, nn
from typing import Annotated
from torch.nn import functional as F


class DSparkOutputHeads(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        if min(hidden_dim, vocab_size, markov_rank) < 1:
            raise ValueError("hidden, vocabulary and Markov dimensions must be positive")
        self.markov_embed = nn.Embedding(vocab_size, markov_rank)
        self.markov_head = nn.Linear(markov_rank, vocab_size, bias=False)
        self.confidence = nn.Linear(
            hidden_dim + markov_rank, 1, bias=False, dtype=torch.float32)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        hidden: Annotated[Tensor, "batch draft_positions hidden_dim"],
        base_logits: Annotated[Tensor, "batch draft_positions vocabulary"],
        seed: Annotated[Tensor, "batch"],
        temperature: float,
    ) -> tuple[
        Annotated[Tensor, "batch seed_plus_draft_positions"],
        Annotated[Tensor, "batch draft_positions vocabulary"],
        Annotated[Tensor, "batch draft_positions"],
    ]:
        """Return tokens [B,K+1], corrected logits [B,K,V], raw confidence [B,K].

        hidden [B,K,D] is before final RMSNorm; base_logits [B,K,V] is after the
        shared vocabulary head. seed [B] is the preceding token, included in output.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError("hidden must have shape [batch, draft, hidden_dim]")
        batch, draft_length, _ = hidden.shape
        vocab_size = self.markov_embed.num_embeddings
        if draft_length < 1 or base_logits.shape != (batch, draft_length, vocab_size):
            raise ValueError("base_logits must match a nonempty hidden draft block")
        if seed.shape != (batch,) or seed.dtype != torch.long:
            raise ValueError("seed must be int64 with shape [batch]")
        if ((seed < 0) | (seed >= vocab_size)).any():
            raise ValueError("seed contains a token outside the vocabulary")
        token_ids, corrected_logits, previous_embeddings = [seed], [], []
        for position in range(draft_length):
            previous_embedding = self.markov_embed(token_ids[-1])
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                # Publisher vocabulary heads accumulate FP32, including Markov bias.
                markov_bias = F.linear(previous_embedding.float(), self.markov_head.weight.float())
                position_logits = base_logits[:, position].float() + markov_bias
                probabilities = (position_logits / temperature).softmax(dim=-1)
            token_ids.append(torch.multinomial(probabilities, 1).squeeze(-1))
            corrected_logits.append(position_logits)
            previous_embeddings.append(previous_embedding)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            markov_embeddings = torch.stack(previous_embeddings, dim=1).float()
            # Concatenate features, not positions: D hidden + R preceding-token features.
            confidence_input = torch.cat((hidden.float(), markov_embeddings), dim=-1)
            # Explicit cast also works after module.to(bfloat16); gradients remain connected.
            raw_confidence = F.linear(confidence_input, self.confidence.weight.float()).squeeze(-1)
        return (
            torch.stack(token_ids, dim=1),
            torch.stack(corrected_logits, dim=1),
            raw_confidence,
        )
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
import math
import torch
from torch import Tensor
from typing import Annotated


def _deepspec_check_distribution(
    probabilities: Annotated[Tensor, "... vocabulary"],
) -> None:
    """Require finite nonnegative probability rows with a nonempty vocabulary."""
    if probabilities.ndim < 1 or probabilities.shape[-1] < 1:
        raise ValueError("probabilities require a nonempty vocabulary axis")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    row_mass = probabilities.float().sum(dim=-1)
    if not torch.allclose(row_mass, torch.ones_like(row_mass), atol=1e-5, rtol=1e-5):
        raise ValueError("each probability row must sum to one")


def deepspec_residual_probabilities(
    target: Annotated[Tensor, "vocabulary"],
    draft: Annotated[Tensor, "vocabulary"],
) -> Annotated[Tensor, "vocabulary"]:
    """Return the normalized positive residual for probability vectors [vocabulary]."""
    if target.ndim != 1 or draft.shape != target.shape:
        raise ValueError("target and draft must have matching vector shapes")
    _deepspec_check_distribution(target)
    _deepspec_check_distribution(draft)
    residual = (target.float() - draft.float()).clamp_min(0)
    # Match the evaluator's fallback for vanishing residual mass.
    residual = torch.where(residual.sum() <= 1e-8, target.float(), residual)
    return residual / residual.sum().clamp_min(1e-8)


def deepspec_verify_probabilities(
    proposed: Annotated[Tensor, "proposal_positions"],
    draft: Annotated[Tensor, "proposal_positions vocabulary"],
    target: Annotated[Tensor, "proposal_plus_bonus_positions vocabulary"],
    uniforms: Annotated[Tensor, "proposal_positions"],
) -> tuple[Annotated[Tensor, "accepted_plus_correction_or_bonus_positions"], int]:
    """Verify one sequence; return committed tokens [accepted+1] and accepted count.

    proposed [K], draft [K,V], target [K+1,V], uniforms [K]. Target row i predicts
    proposal i; its final row supplies the bonus. Uniform draws must be in [0,1).
    Empty proposals are valid. This round excludes EOS and output-budget stopping.
    """
    if proposed.ndim != 1 or proposed.dtype != torch.long:
        raise ValueError("proposed must be an int64 token vector")
    proposal_count = proposed.numel()
    if target.ndim != 2 or target.shape[0] != proposal_count + 1:
        raise ValueError("target must have proposal_count + 1 probability rows")
    if draft.shape != (proposal_count, target.shape[-1]):
        raise ValueError("draft must have shape [proposal_count, vocabulary]")
    if uniforms.shape != (proposal_count,) or not torch.isfinite(uniforms).all():
        raise ValueError("uniforms must be a finite vector matching the proposal count")
    if ((uniforms < 0) | (uniforms >= 1)).any():
        raise ValueError("uniform draws must lie in [0, 1)")
    if ((proposed < 0) | (proposed >= target.shape[-1])).any():
        raise ValueError("proposal contains a token outside the vocabulary")
    _deepspec_check_distribution(target)
    _deepspec_check_distribution(draft)
    rows = torch.arange(proposal_count, device=proposed.device)
    # Paired advanced indexing selects the proposed token probability at each position.
    selected_target = target[rows, proposed].float()
    selected_draft = draft[rows, proposed].float().clamp_min(1e-8)
    probability_ratio = selected_target / selected_draft
    # Once a rejection creates a zero, cumulative product keeps all later entries zero.
    prefix_mask = (uniforms < probability_ratio.clamp_max(1)).long().cumprod(dim=0)
    accepted = int(prefix_mask.sum())
    if accepted < proposal_count:
        next_probabilities = deepspec_residual_probabilities(target[accepted], draft[accepted])
    else:
        next_probabilities = target[-1].float()
    next_token = torch.multinomial(next_probabilities, 1)
    return torch.cat((proposed[:accepted], next_token)), accepted


def deepspec_confident_prefix(
    raw_confidence: Annotated[Tensor, "draft_positions"], threshold: float,
) -> int:
    """Count the contiguous confident prefix of raw logits [draft_length]."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be finite and in (0, 1]")
    if raw_confidence.ndim != 1 or not torch.isfinite(raw_confidence).all():
        raise ValueError("raw_confidence must be a finite vector")
    # Positive fixed threshold; this is not the production scheduler.
    confident = raw_confidence.float().sigmoid() >= threshold
    return int(confident.long().cumprod(dim=0).sum())


def deepspec_reuse_state(
    target_cache: Annotated[Tensor, "cached_target_positions ..."],
    draft_cache: Annotated[Tensor, "cached_draft_positions ..."],
    verified_features: Annotated[Tensor, "seed_plus_proposal_positions ..."],
    start: int, accepted: int,
) -> tuple[
    Annotated[Tensor, "start_plus_seed_plus_accepted_positions ..."],
    Annotated[Tensor, "start_positions ..."],
    Annotated[Tensor, "seed_plus_accepted_positions ..."],
]:
    """Crop dense token-axis-first tensors after a nonterminal verification round.

    start excludes the seed. Target keeps the seed and accepted prefix; the new
    correction/bonus token has not entered its cache. Draft crops to the old start.
    verified_features contains this round's seed and proposal features.
    """
    if not isinstance(start, int) or not isinstance(accepted, int) or min(start, accepted) < 0:
        raise ValueError("start and accepted must be nonnegative integers")
    if min(target_cache.ndim, draft_cache.ndim, verified_features.ndim) < 1:
        raise ValueError("state tensors must have a leading token axis")
    target_length = start + accepted + 1
    if target_length > len(target_cache) or start > len(draft_cache):
        raise ValueError("requested retained state exceeds the available cache")
    if accepted + 1 > len(verified_features):
        raise ValueError("verified features must contain the seed and accepted prefix")
    # Slice token axis zero; clone releases backing storage for speculative tails.
    target = target_cache[:target_length].clone()
    draft = draft_cache[:start].clone()
    features = verified_features[:accepted + 1].clone()
    return target, draft, features
`
  }
];
