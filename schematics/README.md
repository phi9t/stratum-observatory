# Model schematics

Source-faithful, playable architecture pages, one per checkpoint, built from a shared
kit. This subtree is self-contained: it does not touch Stratum's `src/`, `tools/`,
`verification/` or root `package.json`, and Stratum's gates do not cover it.

```
skill/model-schematic/   the skill agents follow (start at SKILL.md)
kit/                     shared components, styles, helpers, and the gallery page
tnsr/                    the tensor library: ops with forward and backward, tape, optimizers
tools/                   new-model, fetch-sources, build, check, shoot, launch-worker
models/<slug>/           one page per checkpoint
runs/                    (ignored) worker transcripts, check reports, screenshots
.sources-cache/          (ignored) pinned upstream files; never committed
dist/                    (ignored) one self-contained HTML per model
```

## Commands (run inside `schematics/`)

```sh
npm ci
npm run new -- <slug> <hf-repo>      # scaffold models/<slug>/
npm run fetch -- <slug>              # pin revisions in sources.json, fill .sources-cache/<slug>/
npm run dev -- <slug>                # vite dev server for one page
npm run build -- <slug>              # dist/<slug>.html, single file
npm run format -- <slug>             # prettier over models/<slug>
npm run check -- <slug>              # every gate; writes runs/check/<slug>/report.json
npm run shoot -- <slug>              # per-section screenshots into runs/shots/<slug>/
```

## Model folder contract

```
models/<slug>/
  index.html        <div id="root"> + <script type="module" src="./main.tsx">
  main.tsx          mounts <Page />
  Page.tsx          <Schematic …> composed of sections; exports `sectionIds: string[]`
  facts.ts          every number the page shows (see below)
  sources.json      checkpoint identity and pinned sources
  coverage.json     reference-class → section map
  reference/        model.ts (clean forward pass), make_fixture.py, fixture.json
  sections/*.tsx    one file per section; no numeric literals other than 0, 1, 2 except layout (below)
  tests/*.test.ts   at least facts.test.ts (scaffolded)
```

### `sources.json`

```json
{
  "checkpoint": { "repo": "org/Name", "revision": "<40-hex commit>" },
  "sources": [
    { "id": "config", "kind": "hf", "repo": "org/Name", "revision": "<sha>", "path": "config.json", "role": "config" },
    { "id": "reference", "kind": "github", "repo": "huggingface/transformers", "revision": "<sha>", "path": "src/transformers/models/x/modeling_x.py", "role": "reference" },
    { "id": "serving", "kind": "github", "repo": "vllm-project/vllm", "revision": "<sha>", "path": "vllm/model_executor/models/x.py", "role": "serving" }
  ]
}
```

- `kind`: `hf` (file in a Hugging Face model repo) or `github`.
- `role`: exactly one `config`; exactly one `reference` (the implementation the page
  is drawn from: publisher code or transformers); at least one `serving` (vllm or
  sglang); optional `training`, `paper`, `extra`.
- `revision` may be written as a branch name; `npm run fetch` replaces it with the
  commit it resolved to. A page is not finished while any revision is not 40 hex.
- Fetched files land in `.sources-cache/<slug>/<id>/<path>`. `fetch` also writes
  `.sources-cache/<slug>/model-info.json` (the Hugging Face model-info response,
  which includes `safetensors.total` when the hub reports it).

### `facts.ts`

```ts
import { cfg, code, derive } from "../../kit/lib/facts";

export const hidden = cfg("hidden_size", 7168);            // config.json key path, value as written
export const scoring = code("sigmoid", { source: "reference", symbol: "DeepseekV3TopkRouter", contains: "sigmoid()" });
export const headDim = derive(() => hidden.value / heads.value, [hidden, heads]);
export const facts = { hidden, scoring, headDim, … };       // every Fact, for the tests
export const parameterTotal = derive(…);                    // required export
```

- `cfg(path, value)`: `path` is a dotted path into the pinned `config.json`
  (`text_config.hidden_size`); the gate fails unless the file holds exactly `value`.
- `code(value, locator)`: a fact read from source. `locator.source` is a source id,
  `symbol` is a class or function name that must be defined in that file, `contains`
  is a short literal that must appear inside that definition.
- `derive(fn, deps)`: computed from other facts; never typed in.
- Section files take numbers only from `facts.ts`.
- Layout is not a fact. The `literals` gate allows numbers in JSX attributes
  (`x={120}`, `size={24}`), in `col:` / `row:` of an op, and in a top-level
  `const LAYOUT… = { … }` (or a name ending `_PX`). Everything else — widths, counts,
  defaults that describe the model — comes from `facts.ts`.

### `coverage.json`

```json
{
  "source": "reference",
  "classes": {
    "DeepseekV3Attention": { "section": "attention" },
    "DeepseekV3PreTrainedModel": { "excluded": "base class with no forward computation" }
  },
  "serving": { "source": "serving", "notes": "differences from the reference, or 'none found'" }
}
```

Every `class` defined in the reference file must appear, mapped to a section id that
exists in `Page.tsx`'s `sectionIds`, or excluded with a reason.

## Gates (`npm run check -- <slug>`)

| Gate | Fails when |
|---|---|
| `pins` | a revision is not a 40-hex commit, or a cached file is missing |
| `typecheck` | `tsc --noEmit` reports an error |
| `literals` | a section file contains a numeric literal outside JSX geometry props |
| `format` | a `.ts`/`.tsx` file in the model folder is not prettier-formatted (`npm run format -- <slug>` fixes it) |
| `tests` | `vitest run models/<slug>` fails; this includes the clean reference's parity test against the upstream fixture |
| `facts` | a `cfg` value differs from `config.json`; a `code` locator does not resolve |
| `coverage` | a reference class is unmapped, or mapped to a missing section |
| `parameters` | `parameterTotal` differs from `safetensors.total` and `facts.ts` exports no `parameterNote` explaining why (skipped, and reported as skipped, when the hub gives no total) |
| `build` | build fails, two builds differ, or the HTML references any external URL as a resource |
| `layout` | in Chrome, light and dark, desktop and mobile: page scrolls sideways, text under 11px, SVG text wider than its box, or text contrast under 4.5:1 |
| `play` | fewer than three distinct controls, a control that does not change the page, or state that does not survive a reload via the URL hash |

The report lists each gate as `pass`, `fail` or `skipped` with details. A gate that
cannot run is `skipped` with the reason, never `pass`.
