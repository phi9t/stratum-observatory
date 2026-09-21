# V4.1 mechanism chapters

`src/mechanism-chapters.mjs` binds each section to an existing scene operator, equations,
source IDs/symbols and a CPU fixture. `src/mechanism-visuals.mjs` renders the saved tensors
and input-position records; it does not recalculate model operations in JavaScript.
The complete scene and its SVG export remain available through Schematic.

The Chapter panel is the default for CSA2, Engram, Inputs, Residual and dSpark. A model
context strip precedes each chapter. Token/cache axes, stream matrices, layer ownership
and speculative timelines use their own visual structures. Equations use offline MathML
and include a plain-text equivalent. A selected operator opens the inspector; examples
otherwise retain the whole reading width. Mobile layouts stack equations below the
visuals and scroll dense tensors without scaling their text.

Chapter selections use the existing sanitized `example` field with a `chap--` prefix.
They survive panel changes, URL restoration and browser history. Selecting a normalization
step or token also selects its scene operation. The Python navigator focuses the relevant
mHC function. Source links are independent of local executable code coverage.

## Reproduction boundaries

- `examples/deepseek_v41/*.py` remains canonical. The build generates Python delivery bytes.
- `tools/export-worked.py` runs reduced-width CPU examples and emits every displayed
  numerical intermediate, including finite Sinkhorn errors. Teaching weights are explicit.
- `tools/export-input-cases.py` executes the pinned publisher formatter, tokenizer and image
  processor against fixed messages and a locally authored chart. The checked-in fixture
  records source hashes, exact serialized tokens and expanded image-position ownership.
- Formatter reproduction runs in the tested macOS Seatbelt boundary with no network or
  filesystem writes. Missing cache, dependencies or sandbox reports BLOCKED. CI checks
  fixture identities and ordering offline; it does not claim to rerun absent publisher code.
- The derived fixture contains no upstream tokenizer assets or weights. All pages remain
  standalone and make no runtime network requests. The gzip-packed application and nested
  fixtures are scanned after decoding as well as in their canonical source form.
- V4-Pro dSpark architecture, V4.1 feature mappings and the reduced verification example
  have separate scopes. CPU success is neither checkpoint parity nor a GPU benchmark.

`tests/mechanism_chapters.py` covers real reader journeys. Node tests bind sections to
actual operators and reject unsupported selections. PyTorch tests check causality,
normalization, hash boundaries, matrix orientation, delayed gates and retained state.
The existing schematic/reference journeys still run explicitly in the Schematic panel.
Capture and aesthetic inventories include interactive chapter states derived from saved
fixture dimensions. Accepted full-scene composition baselines are unchanged.
