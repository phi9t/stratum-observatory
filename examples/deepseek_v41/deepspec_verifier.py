# Adapted from DeepSpec, MIT; Copyright (c) 2026 The DeepSpec Authors.
"""Single-sequence offline Qwen3/Gemma4 evaluator helpers, not V4.1 serving.

Probability inputs are already normalized at the caller's sampling temperature.
These helpers intentionally do not consume the V4.1 checkpoint configuration.
"""
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
