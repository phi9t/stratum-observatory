#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $# == 1 ]] || { echo "Usage: $0 <existing-GitHub-repository-URL>" >&2; exit 2; }
repo="$1"
if [[ "$repo" != git@github.com:* && "$repo" != https://github.com/* ]]; then
  echo 'Expected an explicit GitHub repository URL.' >&2; exit 2
fi
npm run verify
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then git init -b main; fi
if git remote get-url origin >/dev/null 2>&1; then
  [[ "$(git remote get-url origin)" == "$repo" ]] || { echo 'Existing origin differs; refusing to rewrite it.' >&2; exit 3; }
else
  git remote add origin "$repo"
fi
git add .
if ! git diff --cached --quiet; then git commit -m "feat: STRATUM multi-model observatory and evidence gates"; fi
git push -u origin HEAD
