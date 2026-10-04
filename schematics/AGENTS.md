# Model schematics: agent guide

This subtree builds one playable, source-faithful architecture page per model
checkpoint. If you are asked to visualize, diagram, explain or "build a schematic
for" a model architecture in this repository, this is where the work goes and the
skill below is how it is done.

## Use the skill

Read `skill/model-schematic/SKILL.md` and follow it, including every file under
`skill/model-schematic/references/` and the screenshots in
`skill/model-schematic/examples/`. It is a complete procedure: pin sources, read the
upstream code, write facts and a clean reference implementation, build the page with
the kit, pass the gates, look at screenshots, fix, report.

```sh
cd schematics
npm ci                               # once
npm run new -- <slug> <hf-repo>      # scaffold models/<slug>/
npm run fetch -- <slug>              # pin and cache upstream sources
npm run format -- <slug>
npm run check -- <slug>              # all gates; must end "ok"
npm run shoot -- <slug>              # screenshots to look at
npm run index                        # rebuild every page and dist/index.html
```

`models/gpt2/` is the worked example. `README.md` defines the folder contract and
what each gate checks. `tnsr/` is the tensor library the clean references are
written on.

## Working rules

- Write only under `schematics/models/<slug>/` for the model you were asked about.
  Do not change `kit/`, `tnsr/`, `tools/` or `skill/` to make a page pass; if one of
  them is wrong, say so in the page's `REPORT.md`.
- Nothing outside `schematics/` is part of this work. The rest of the repository has
  its own contract (the root `AGENTS.md`) and its own verification that these pages
  must not disturb.
- Use the network only to read public source code and model configs from GitHub and
  Hugging Face. Never download model weights.
- Python with torch and transformers for parity fixtures:
  `${SCHEMATICS_PYTHON:-/data03/yph/venv/bin/python}`. Headless Chrome:
  `${CHROME:-/usr/bin/google-chrome}`; `check` and `shoot` drive it.
- Do not commit, push or publish unless the person asking told you to. Leave the work
  in the tree with `REPORT.md` written and the final `npm run check` table shown.
- A gate that cannot run is reported as skipped with the reason, never as passed.
  Parity that could not be run is stated as unverified on the page and in the report.

## Publishing

Pages build to `schematics/dist/` (ignored). The public site is the separate
`stratum-pages` repository, where these pages live under `schematics/`. Publishing
is a deliberate step taken by a person, not part of building a page.
