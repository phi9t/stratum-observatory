# Fidelity: sources, facts, coverage

## Finding the implementations

Start from the checkpoint's Hugging Face repo.

1. `config.json` gives `model_type` and `architectures`. If it has an `auto_map`, the
   repo ships its own `modeling_*.py` / `configuration_*.py`: that is publisher code.
   Many repos also ship reference inference code (`inference/model.py`).
2. transformers: `src/transformers/models/<model_type>/modeling_<model_type>.py`. If
   there is no directory for this `model_type`, find which class actually loads it
   (`architectures`, the repo README, `auto` mappings). Do not assume a sibling
   directory with a similar name is the same architecture; check.
3. vllm: `vllm/model_executor/models/registry.py` maps architecture names to files.
4. sglang: `python/sglang/srt/models/`.
5. Training: torchtitan `torchtitan/models/`, Megatron-LM `megatron/core/transformer/`,
   Megatron-Bridge, NeMo AutoModel.

List directories with the GitHub API
(`https://api.github.com/repos/<repo>/contents/<path>`) or by fetching the tree page;
fetch files from `raw.githubusercontent.com`. `HF_ENDPOINT=https://hf-mirror.com`
works if the hub is slow.

**Choose the reference** in this order: publisher modeling code that ships with the
checkpoint; else the transformers class that loads it; else the publisher's reference
inference code. The reference is what the page is drawn from and what `coverage.json`
is checked against. If the implementation spans several files (attention in one,
MoE in another), add each as its own source with role `extra` and cite them in refs.

**The serving implementation is not decoration.** Read it for what the reference
hides: what is cached between tokens, what is fused or reordered, which tensors are
never materialised, which config fields it reads that the reference ignores. Put the
differences on the page and in `coverage.json` `serving.notes`. If you find none after
actually comparing, write what you compared.

## Facts

- `cfg("a.b", value)`: the gate compares with the pinned `config.json`. Use the value
  exactly as written there, including `null`, strings and lists.
- `code(value, { source, symbol, contains })`: for anything the config does not say —
  which activation, norm placement, a hard-coded constant, which layers differ, the
  scoring function, whether a bias exists. `symbol` must be the class or function whose
  body shows it, and `contains` a short literal from that body that a reader could
  search for. A `code` fact whose `contains` is just the symbol name proves nothing.
- `derive(fn, deps)`: head dimension, parameter counts, cache sizes, shapes, counts of
  layers of each kind. If you catch yourself typing a number you worked out by hand,
  it is a derived fact.
- Defaults matter. If the config omits a field, the value comes from the config class
  default or the code: that is a `code` fact with a locator, not a `cfg` fact.
- Per-layer variation (first N layers dense, attention kind alternating, shared
  indices) must be a function `kindOf(layerIndex)` derived from config and code, used
  by both the layer map and the ledger.

## Parameter ledger

Count parameters from shapes, per module, the way the reference constructs them
(weights and biases, norms, embeddings, tied weights counted once). Group the ledger by
the parts a reader cares about (embedding, attention, FFN or experts, router, norms,
head). Export `parameterTotal`.

The gate compares it with the hub's `safetensors.total`. If they differ, find out why
before writing `parameterNote`: common real reasons are persistent buffers stored in
the checkpoint, untied or duplicated tensors, auxiliary modules (vision tower, MTP
heads, draft models) the page scopes out, or quantised storage. The note must state
the difference as an exact number and what accounts for it. "Approximately equal" is
not an explanation.

## Coverage

`coverage.json` maps every class in the reference file to the section that draws it,
or excludes it with a reason. Good reasons: output dataclass, base class with no
forward computation, a head for a task the checkpoint does not ship, a legacy branch
the config disables (name the field and its value). "Not important" is not a reason.

Coverage is the floor, not the goal. Also walk each `forward` and make sure every
operation in it is an op in some `OpGraph`: every norm, projection, reshape that
changes meaning, activation, residual add, mask, scale, cache read or write.

## When implementations disagree

Different naming or fused operations are not disagreement; describe them. If two
implementations compute different things (an extra norm, a different scale, a
different default), show the reference on the diagram, state the difference in the
op's inspector `note`, and list it in `REPORT.md`. Do not average them.

## What the gates cannot check

The gates prove that config values, symbols, parameter totals and (with a fixture) the
clean reference's arithmetic are right. They do not read your prose, your inspector
shapes or your takeaways. An independent audit of the first eight pages found no error
in architecture or formulas, and found every one of the following more than once. Check
your page for each before you report:

- **Inspector shapes on parallel branches.** An op that reads the block input, not the
  previous box's output, needs its input written out. Do not let a helper default an
  op's input to the previous op's output. Shapes around a reshape must match the code
  line by line: if the norm runs before the transpose, the norm's input still has the
  head axis after the sequence axis.
- **Row-shaped tensors.** Inside expert dispatch the tensors are rows of selected
  tokens, not `[B, S, width]`. Use a symbolic count for the first axis and say what it
  counts.
- **Claims about code that is not pinned.** If the pinned file only calls a helper (a
  cache class, a fused kernel, a serving layer), you may describe the call, not what the
  helper stores or does. Pin the helper as an extra source, or say it is not pinned.
- **Defaults you did not see.** A value that comes from a config-class default is a
  `code` fact with a locator in a pinned file, or it is stated as not pinned.
- **A path the model does not run.** If the code always dispatches to another backend,
  say that the drawn arithmetic is the equivalent eager form.
- **"Learned", "trained", "why it helps".** Toy weights are random: do not call them
  learned. Do not state training behaviour, stability or design intent unless a pinned
  source states it.
- **Superlatives and totals.** "The widest tensor", "the only signal", "everything
  above" are claims; check them against the whole model, including the logits and the
  mask.
- **Conditions.** State the condition the code tests, then say what it reduces to for
  this checkpoint; do not state the reduced form as the rule.
- **Readout labels and units.** A count of parameters is not a count of matrices.
- **Differences in the clean reference.** List in `REPORT.md` every place
  `reference/model.ts` departs from the upstream `forward` (an omitted no-op, an
  unconditional branch, a different tie-break), even when the result is identical.

If you can, have a second agent that has not seen your reasoning audit the finished page
against the pinned sources, claim by claim, before it is published.
