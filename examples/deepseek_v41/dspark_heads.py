# Adapted from DeepSeek V4.1 Flash, MIT; Copyright (c) 2023 DeepSeek.
import math
import torch
from torch import Tensor, nn
from typing import Annotated
from torch.nn import functional as F

from .config import V41Config


class DSparkOutputHeads(nn.Module):
    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.markov_embed = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_head = nn.Linear(config.markov_rank, config.vocab_size, bias=False)
        self.confidence = nn.Linear(
            config.hidden_dim + config.markov_rank, 1, bias=False, dtype=torch.float32)
        self.hidden_dim = config.hidden_dim

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
