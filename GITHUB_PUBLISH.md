# Publication boundary

The repository includes `.github/workflows/pages.yml`. Engineering runs first; scientific publication requires its exact artifact and receipt. Pages deploys only after both jobs pass. The current evidence migration is BLOCKED, so the default Pages workflow intentionally does not deploy.

This archive was not pushed to a remote repository. No GitHub account, branch protection, secret, reviewer identity or source evidence cache has been configured remotely.

After completing and reviewing the evidence, create an explicitly selected repository, protect verification/trust/workflow changes, and enable Pages via GitHub Actions. `publish-github.sh <existing-repository-URL>` runs the required combined verifier and refuses to rewrite a different existing origin. It never force-pushes or suppresses failed verification.

For a local exploratory preview, use the shipped `dist/index.html`; it visibly states the reference-mapped, review-pending boundary. Do not label that preview scientifically certified.
