# Public source distribution

Use `npm run package:public` to produce `.publication/stratum-public.tar.gz`.
The exporter scans the current file set before writing and fails on recognized private
identifiers. It includes current tracked and non-ignored files, including reviewed changes,
and omits deleted files, `.git`, ignored caches, raw runtime captures and local staging.
Archive ownership and timestamps are normalized. It never pushes or changes Git history.

Extract into an empty directory, review the contents, then initialize a new Git repository
using a public author name and email. The pre-scrub history contains private captures and
personal commit metadata. A new commit deleting those files does not erase older commits.
For an existing repository, audit and rewrite every published branch, then push each with
an explicit lease against its previous remote tip. Keep the original recovery bundle and
any unrevised local worktree refs private. Do not upload the whole workspace. GitHub may
retain old pull-request refs and cached commits after branch replacement; resolve those
with GitHub before changing repository visibility.

Raw GPU probes contained network addresses, host identifiers, GPU UUIDs, device serials
and GUIDs. The public distribution withdraws those captures and their runtime receipts.
Original bytes are preserved locally under the ignored `.verification-cache/` directory;
that directory must never be distributed. Scientific source snapshots and trust roots
are unchanged. Runtime tests use temporary synthetic captures; they are not runtime evidence.
New publication-safe real evidence needs a separate capture and review process.

The automated scan covers home paths, IPv4/IPv6 addresses, MAC addresses, GPU UUIDs,
email addresses, recognizable keys/tokens, known private host naming patterns, external
symlinks and unexpected PNG metadata. Loopback addresses, system tool paths, synthetic
hardware fixtures, software versions, public citations and code-owner handles are expected.
It cannot prove arbitrary prose contains no personal information or discover every secret
format. Public source attribution and review ownership are deliberately retained.

Software verification and privacy checks do not certify scientific claims. Publication
remains BLOCKED and runtime experiments remain NOT_RUN until their evidence requirements
are fulfilled. GitHub visibility, branch protection, licensing decisions and deployment
are separate repository-owner actions.
