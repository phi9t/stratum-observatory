"""Writes fixture.json: a toy-width GPT-2 from the upstream implementation, with its
weights, one input and the logits and attention weights it produces; then the
next-token loss on that input, the gradient of the loss for every parameter, and the
parameters after three optimizer steps (AdamW and, separately, plain SGD).

The TypeScript reference in model.ts must reproduce these numbers: the forward pass in
tests/reference.test.ts, the loss, gradients and training steps in
tests/reference-grad.test.ts. Run with any Python that has torch and transformers:

    python models/gpt2/reference/make_fixture.py
"""

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

TOY = dict(vocab_size=40, n_positions=16, n_embd=24, n_layer=2, n_head=3)

torch.manual_seed(0)
config = GPT2Config(**TOY, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
config._attn_implementation = "eager"
model = GPT2LMHeadModel(config).eval()
# The default init leaves biases at zero and norms at identity; perturb everything so
# a wrong bias or gain cannot hide.
with torch.no_grad():
    for parameter in model.parameters():
        parameter.add_(0.05 * torch.randn_like(parameter))


def rounded(value: float) -> float:
    """Seven significant digits: float32 precision, and a fixture a third the size."""
    return float(f"{value:.7g}")


def dump(tensor: torch.Tensor) -> dict:
    return {"shape": list(tensor.shape), "data": [rounded(v) for v in tensor.flatten().tolist()]}


def exported(module: GPT2LMHeadModel) -> dict:
    """The model's parameters under their checkpoint names. lm_head.weight is left out:
    it is the same tensor as transformer.wte.weight."""
    return {name: dump(parameter) for name, parameter in module.named_parameters()}


# Train from exactly the values the fixture stores, so rounding cannot separate the two sides.
with torch.no_grad():
    for parameter in model.parameters():
        parameter.copy_(torch.tensor(dump(parameter)["data"]).view_as(parameter))

# Nine tokens of text: the first eight are the input, the last eight the targets.
tokens = torch.tensor([[3, 17, 5, 29, 11, 0, 38, 8, 21]])
input_ids, targets = tokens[:, :-1], tokens[:, 1:]


def next_token_loss(module: GPT2LMHeadModel) -> torch.Tensor:
    logits = module(input_ids).logits
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))


with torch.no_grad():
    output = model(input_ids, output_attentions=True)
weights = exported(model)

# Gradients. The token embedding is used twice (as the input table and as the output
# head), so its gradient is the sum of both uses; autograd does that for a tied tensor.
loss = next_token_loss(model)
loss.backward()
grads = {name: dump(parameter.grad) for name, parameter in model.named_parameters()}
model.zero_grad()


def train(make_optimizer, steps: int = 3) -> dict:
    """Runs a few steps on a copy of the model; records the loss before each step."""
    module = copy.deepcopy(model)
    optimizer = make_optimizer(module.parameters())
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        step_loss = next_token_loss(module)
        step_loss.backward()
        optimizer.step()
        losses.append(rounded(step_loss.item()))
    return {"losses": losses, "weights": exported(module)}


ADAMW = dict(lr=1e-2, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
SGD = dict(lr=0.1)

fixture = {
    "upstream": "transformers.GPT2LMHeadModel",
    "transformers_version": __import__("transformers").__version__,
    "torch_version": torch.__version__,
    "config": TOY,
    "input_ids": input_ids.tolist(),
    "targets": targets.tolist(),
    "weights": weights,
    "logits": dump(output.logits),
    "attention_layer0": dump(output.attentions[0]),
    "loss": rounded(loss.item()),
    "grads": grads,
    "adamw": {"options": ADAMW, **train(lambda parameters: torch.optim.AdamW(parameters, **ADAMW))},
    "sgd": {"options": SGD, **train(lambda parameters: torch.optim.SGD(parameters, **SGD))},
}
path = Path(__file__).with_name("fixture.json")
path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
print(f"wrote {path} ({path.stat().st_size} bytes, {len(weights)} tensors)")
