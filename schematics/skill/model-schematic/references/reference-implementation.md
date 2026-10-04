# The clean reference implementation

Every page ships `models/<slug>/reference/model.ts`: the checkpoint's forward pass
written once, cleanly, in TypeScript on `kit/ref/tensor.ts`. It is the page's
authoritative statement of what the model computes, and it does three jobs:

1. **It is readable.** One file, top to bottom, with no framework glue.
2. **Run without data at the real widths** (shape mode) it gives the true shape of
   every operation and an exact parameter count. The page's shapes and
   `parameterTotal` must agree with it.
3. **Run with data at toy widths** (value mode) it computes real numbers: in a test,
   to prove it matches upstream; in the browser, for the page's experiments.

`models/gpt2/reference/` is the worked example: `model.ts`, `make_fixture.py`,
`fixture.json`, and `tests/reference.test.ts`.

## Writing it

- **Distil, do not port.** Upstream code carries cache plumbing, device and dtype
  handling, optional outputs, deprecated arguments, alternative backends and branches
  for configs this checkpoint does not use. Leave all of it out. Keep exactly the
  arithmetic this checkpoint performs, in the order upstream performs it.
- **Take a branch only if this checkpoint takes it.** When upstream has
  `if config.x:`, read the pinned config, keep the taken side, and say in a comment
  which field decided it. Layer-dependent behaviour (dense first layers, alternating
  attention kinds) is real structure: keep it, as a function of the layer index.
- **Use the checkpoint's parameter names** (`model.layers.3.mlp.gate.weight`). Then
  weights exported from upstream load with no mapping, and the test can assert that
  every upstream tensor was consumed and none was invented.
- **One function per upstream module**, named after it, in the same file. Small
  helpers (a norm, a projection) are fine. No classes, no inheritance.
- **Mark the tensors the page shows** with `ctx.mark("<id>", tensor)`. Use ids that
  match the op ids in your `OpGraph`s so shapes can be read from the trace.
- **A config type with only what the forward pass needs**, built from `facts.ts` for
  the real size and from a `TOY` constant for experiments.
- If an op is missing from `kit/ref/tensor.ts`, write it as a small pure function in
  `reference/ops.ts` in the same style (shape mode must work: return
  `{ shape, data: null }` when any input has no data), and test it.
- The first lines of the file say which upstream classes it was distilled from and at
  which pinned source.

Scope it like the page: the text decoder's forward pass to logits. Decode-time
caching, sampling, training losses, vision towers and auxiliary heads are outside
unless the page covers them; say so in the header comment.

## Proving it (required)

`reference/make_fixture.py` builds the **upstream** model at toy widths, perturbs
every parameter (so a wrong bias or gain cannot hide behind a zero or identity init),
runs one input, and writes `fixture.json` with the config, input, weights, logits and
a few intermediate tensors (attention weights, router choices). Keep the toy small
enough that the fixture is under about 500 KB, but keep every structural feature
alive: at least two layers of every layer kind, at least two heads, several experts
with top-k above one, groups if the router has groups.

Run it with a Python that has torch and transformers
(`${SCHEMATICS_PYTHON:-/data03/yph/venv/bin/python}`). Which upstream to instantiate,
in order of preference:

1. the transformers class that loads this checkpoint;
2. the publisher's `modeling_*.py` or `inference/model.py` from the pinned cache,
   imported directly;
3. another pinned implementation (torchtitan, vllm) if it runs on CPU standalone.

`tests/reference.test.ts` then requires, as the GPT-2 test does:

- value mode on the fixture weights reproduces the logits (max abs difference under
  `1e-4`) and the intermediates;
- the set of parameter names the clean code asked for equals the set in the fixture;
- shape mode at the real config gives `ctx.parameterCount() === parameterTotal` and
  the shapes the page states for a handful of ops.

If no upstream can be run locally, do not fake a fixture. Ship the reference with the
shape-mode test only, and state in `REPORT.md` and on the page that numerical parity
is unverified and why.

## Using it on the page

- A **"Run it" section** runs the reference in the browser at toy width with seeded
  weights (`new ref.Ctx("value", null, seed)`) and shows something computed: attention
  weights as a `Heatmap`, router scores and the experts actually chosen, the norm of
  the stream after each layer. Give it controls (layer, head, seed, a position).
- **Show what is distinctive about this model**, not the experiment the worked example
  happens to use. GPT-2's page shows a causal attention heatmap because that is all
  GPT-2 has. A grouped-query model should show keys shared across query heads; a
  latent-attention model the latent that is cached against the keys rebuilt from it; a
  mixture-of-experts model which experts each token of a toy sequence is sent to and
  with what weights; a sliding-window or sparse model the keys actually reached.
  Write your own sentence and takeaways about what your figure shows.
- Seeded weights can make a toy run look uniform. Pass a `gain` to `ref.Ctx` (GPT-2
  uses 4) until the figure has visible structure.
- Say plainly that the weights are random: the experiment shows how the mechanism
  behaves, never what the trained model does.
- Prefer computed values over synthetic ones everywhere: a router demo should run the
  reference's router on a toy stream, not `routeToken` on made-up logits, once the
  reference exists.
- Keep toy runs small (they execute on every render): a few layers, width under 64,
  at most a dozen tokens.
