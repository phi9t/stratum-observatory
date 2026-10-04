#!/usr/bin/env bash
# usage: tools/launch-worker.sh <traecli|codex> <slug> <hf-repo> [--model M] [--timeout-min N] [--dry-run]
#
# Runs one coding agent against the model-schematic skill in its own git
# worktree and records what it did. Run from the schematics/ directory of a
# clean worktree; the current HEAD is the base the worker starts from.
#
#   worktree   <repo>/../stratum-runs/<worker>-<slug>   branch schematics/run/<worker>-<slug>
#   run dir    schematics/runs/workers/<worker>-<slug>/  (git-ignored, in THIS worktree)
#              prompt.md events.jsonl stderr.log last.md run.json diff.txt check.json
#
# env: TRAE_PROFILE (traecli -p). --dry-run prints the commands and launches nothing.
set -uo pipefail

usage() {
  sed -n '2s/^# //p' "$0" >&2
  exit 2
}
die() {
  echo "error: $*" >&2
  exit 1
}

WORKER=""
SLUG=""
REPO=""
MODEL=""
TIMEOUT_MIN=90
DRY_RUN=0
positional=()
while (($# > 0)); do
  case "$1" in
    --model)
      (($# >= 2)) || die "--model needs a value"
      MODEL="$2"
      shift 2
      ;;
    --timeout-min)
      (($# >= 2)) || die "--timeout-min needs a value"
      TIMEOUT_MIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help) usage ;;
    --*) die "unknown option $1" ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done
((${#positional[@]} == 3)) || usage
WORKER="${positional[0]}"
SLUG="${positional[1]}"
REPO="${positional[2]}"

case "$WORKER" in
  traecli | codex) ;;
  *) die "worker must be traecli or codex, not \"$WORKER\"" ;;
esac
[[ "$SLUG" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || die "slug \"$SLUG\" must match ^[a-z0-9][a-z0-9._-]*\$"
[[ "$SLUG" != "gallery" ]] || die "\"gallery\" is reserved for kit/gallery"
[[ "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "hf-repo \"$REPO\" must look like org/Name"
[[ "$TIMEOUT_MIN" =~ ^[1-9][0-9]*$ ]] || die "--timeout-min must be a positive integer, not \"$TIMEOUT_MIN\""

# Environment preconditions. A dry run reports them and carries on; a real run stops.
precondition() {
  if ((DRY_RUN)); then
    echo "dry-run: would fail: $*" >&2
  else
    die "$*"
  fi
}

HERE="$(pwd -P)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git worktree"
[[ "$HERE" == "$REPO_ROOT/schematics" ]] || die "run this from the schematics/ directory of the worktree (cwd is $HERE)"
BASE="$(git rev-parse HEAD)" || die "cannot read HEAD"

NAME="$WORKER-$SLUG"
BRANCH="schematics/run/$NAME"
WORKTREE="$(dirname "$REPO_ROOT")/stratum-runs/$NAME"
WORK_CWD="$WORKTREE/schematics"
RUN_DIR="$HERE/runs/workers/$NAME"
PREAMBLE="$HERE/skill/model-schematic/worker-preamble.md"
PROMPT="$RUN_DIR/prompt.md"
FINAL_LINE="Use the model-schematic skill at $WORK_CWD/skill/model-schematic/SKILL.md to build the schematic for the Hugging Face checkpoint \`$REPO\` as models/$SLUG/."

[[ -z "$(git status --porcelain)" ]] || precondition "the worktree is not clean (commit or stash first; the worker starts from HEAD)"
[[ -f "$PREAMBLE" ]] || precondition "$PREAMBLE is missing"
[[ ! -e "$WORKTREE" ]] || precondition "$WORKTREE already exists"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  precondition "branch $BRANCH already exists"
fi
[[ ! -e "$RUN_DIR/run.json" ]] || precondition "$RUN_DIR already holds a finished run"
command -v "$WORKER" >/dev/null 2>&1 || precondition "$WORKER is not on PATH"

case "$WORKER" in
  traecli)
    agent=(traecli exec --json -o "$RUN_DIR/last.md" -C "$WORK_CWD"
      --permission-mode bypass_permissions --skip-git-repo-check --shell-tool-timeout 10m)
    [[ -z "${TRAE_PROFILE:-}" ]] || agent+=(-p "$TRAE_PROFILE")
    ;;
  codex)
    agent=(codex exec --json -o "$RUN_DIR/last.md" -C "$WORK_CWD"
      --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check)
    ;;
esac
[[ -z "$MODEL" ]] || agent+=(-m "$MODEL")
agent+=(-)
# setsid puts timeout and the agent in a session of their own, so the TERM/KILL
# that timeout sends on expiry reaches the agent's whole process group.
launch=(setsid --wait timeout --signal=TERM --kill-after=30 "${TIMEOUT_MIN}m" "${agent[@]}")

if ((DRY_RUN)); then
  quoted() { printf '%q ' "$@"; }
  cat <<EOF
base        $BASE
worktree    $WORKTREE  (branch $BRANCH)
run dir     $RUN_DIR

git worktree add -b $(quoted "$BRANCH" "$WORKTREE" "$BASE")
(cd $(quoted "$WORK_CWD")&& npm ci)
mkdir -p $(quoted "$RUN_DIR")
cat $(quoted "$PREAMBLE")> $(quoted "$PROMPT")  # plus the final line:
#   $FINAL_LINE
(cd $(quoted "$WORK_CWD")&& $(quoted "${launch[@]}")< $(quoted "$PROMPT")> $(quoted "$RUN_DIR/events.jsonl")2> $(quoted "$RUN_DIR/stderr.log"))
# then, whatever the exit code:
#   write $RUN_DIR/run.json (exit code, wall time)
#   write $RUN_DIR/diff.txt (git status --short, git diff --stat, paths outside schematics/models/$SLUG/)
(cd $(quoted "$WORK_CWD")&& npm run check -- $(quoted "$SLUG"))
cp $(quoted "$WORK_CWD/runs/check/$SLUG/report.json" "$RUN_DIR/check.json")
EOF
  exit 0
fi

mkdir -p "$(dirname "$WORKTREE")" "$RUN_DIR" || die "cannot create $RUN_DIR"
git worktree add -b "$BRANCH" "$WORKTREE" "$BASE" || die "git worktree add failed"
[[ -d "$WORK_CWD" ]] || die "$WORK_CWD does not exist at $BASE (is schematics/ committed?)"
(cd "$WORK_CWD" && npm ci) || die "npm ci failed in $WORK_CWD"

{
  cat "$PREAMBLE"
  printf '\n%s\n' "$FINAL_LINE"
} >"$PROMPT" || die "cannot write $PROMPT"

echo "launching $WORKER for $SLUG ($REPO), timeout ${TIMEOUT_MIN}m; events: $RUN_DIR/events.jsonl"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started="$(date +%s)"
(cd "$WORK_CWD" && "${launch[@]}" <"$PROMPT" >"$RUN_DIR/events.jsonl" 2>"$RUN_DIR/stderr.log")
exit_code=$?
wall_seconds=$(($(date +%s) - started))
timed_out=false
if ((exit_code == 124 || exit_code == 137)); then
  timed_out=true
fi
echo "$WORKER exited with $exit_code after ${wall_seconds}s"

json_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}
cat >"$RUN_DIR/run.json" <<EOF
{
  "worker": $(json_string "$WORKER"),
  "slug": $(json_string "$SLUG"),
  "repo": $(json_string "$REPO"),
  "model": $(json_string "$MODEL"),
  "base": $(json_string "$BASE"),
  "branch": $(json_string "$BRANCH"),
  "worktree": $(json_string "$WORKTREE"),
  "started_at": $(json_string "$started_at"),
  "timeout_minutes": $TIMEOUT_MIN,
  "exit_code": $exit_code,
  "timed_out": $timed_out,
  "wall_seconds": $wall_seconds
}
EOF

# What the worker changed, measured against the base (so its own commits count too).
{
  echo "## git status --short"
  git -C "$WORKTREE" status --short
  echo
  echo "## git diff --stat HEAD"
  git -C "$WORKTREE" diff --stat HEAD
  if [[ "$(git -C "$WORKTREE" rev-parse HEAD)" != "$BASE" ]]; then
    echo
    echo "## git log --oneline $BASE..HEAD"
    git -C "$WORKTREE" log --oneline "$BASE..HEAD"
    echo
    echo "## git diff --stat $BASE"
    git -C "$WORKTREE" diff --stat "$BASE"
  fi
  echo
  echo "## changed paths outside schematics/models/$SLUG/"
  outside="$({
    git -C "$WORKTREE" diff --name-only "$BASE"
    git -C "$WORKTREE" ls-files --others --exclude-standard
  } | sort -u | grep -v "^schematics/models/$SLUG/" || true)"
  if [[ -z "$outside" ]]; then
    echo "(none)"
  else
    echo "$outside"
  fi
} >"$RUN_DIR/diff.txt" 2>&1

(cd "$WORK_CWD" && npm run check -- "$SLUG") >"$RUN_DIR/check.log" 2>&1
check_code=$?
if [[ -f "$WORK_CWD/runs/check/$SLUG/report.json" ]]; then
  cp "$WORK_CWD/runs/check/$SLUG/report.json" "$RUN_DIR/check.json"
  echo "check exited with $check_code; report: $RUN_DIR/check.json"
else
  echo "check exited with $check_code and wrote no report; see $RUN_DIR/check.log" >&2
fi
echo "run recorded in $RUN_DIR"
exit "$exit_code"
