You are working autonomously in a git worktree. No one will answer questions during
this run, so make reasonable decisions, record them, and finish the task.

Ground rules:

- Work only inside the `schematics/` directory of this worktree. Create and edit files
  only under `schematics/models/<slug>/` (plus the ignored `runs/`, `dist/` and
  `.sources-cache/` that the tools write). Do not edit `kit/`, `tools/`, `skill/`,
  `README.md` or anything outside `schematics/`.
- Do not commit, push, create branches, or change git configuration. Do not install
  global packages or change files in the home directory.
- You have network access for fetching public source code and model configs from
  GitHub and Hugging Face. Do not download model weights.
- Node dependencies are already installed in `schematics/`. Headless Chrome is at
  `/usr/bin/google-chrome`; the `check` and `shoot` tools drive it for you.
- You can read images. After `npm run shoot`, open the PNGs and look at them.
- Follow the skill exactly, including its reference files. When the skill and your own
  habits disagree, the skill wins.
- Finish by writing `models/<slug>/REPORT.md` and printing the final
  `npm run check -- <slug>` table. If something could not be done, say exactly what
  and why rather than working around it.

