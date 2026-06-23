#!/usr/bin/env bash
set -uo pipefail

GUARD_LINE='Do NOT read or print generated files (.next/, dist/, build/, node_modules/, coverage/, *.min.*, compiled CSS). Reason about source only; never cat build output.'

MODE="build"
BASE="beta"
TASK_ID=""
SPEC_FILE=""
SPEC_CONTENT=""
TEST_CMD=""
TEST_CMD_SET=0
TEARDOWN=0

OVERALL_RESULT="PASS"
TEST_RESULT="NOT RUN"
FAILING_COMMAND=""

usage() {
  cat >&2 <<'USAGE'
usage: bash scripts/codex_loop.sh --mode <build|guard|duel> [--task <id>] [--base <branch>] [--test '<cmd>'] [--spec <file>] [--teardown]

Modes:
  build   Codex implements, then Codex reviews the diff for Claude.
  guard   Claude already edited; Codex reviews hard.
  duel    Codex implements in an isolated worktree for comparison.
USAGE
}

die() {
  echo "codex_loop: $*" >&2
  exit 2
}

require_value() {
  [ $# -ge 2 ] && [ -n "$2" ] || die "$1 requires a value"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      require_value "$@"
      MODE="$2"
      shift 2
      ;;
    --task)
      require_value "$@"
      TASK_ID="$2"
      shift 2
      ;;
    --base)
      require_value "$@"
      BASE="$2"
      shift 2
      ;;
    --test)
      require_value "$@"
      TEST_CMD="$2"
      TEST_CMD_SET=1
      shift 2
      ;;
    --spec)
      require_value "$@"
      SPEC_FILE="$2"
      shift 2
      ;;
    --teardown)
      TEARDOWN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

case "$MODE" in
  build|guard|duel) ;;
  *) die "--mode must be build, guard, or duel" ;;
esac

if [ "$TEST_CMD_SET" -eq 1 ] && [ -z "$TEST_CMD" ]; then
  die "--test cannot be empty"
fi

REPO_ROOT_RAW="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
REPO_ROOT="$(cd "$REPO_ROOT_RAW" 2>/dev/null && pwd -P)" || die "cannot resolve repo root"
CURRENT_DIR="$(pwd -P)"

if [ "$CURRENT_DIR" != "$REPO_ROOT" ]; then
  die "run from repo root: $REPO_ROOT"
fi

if [ -z "$TASK_ID" ]; then
  HEAD_SHORT="$(git rev-parse --short HEAD 2>/dev/null)" || die "cannot resolve HEAD"
  TASK_ID="task-${HEAD_SHORT}-$$"
fi

case "$TASK_ID" in
  ""|*/*|*\\*|*..*|*[!A-Za-z0-9._-]*)
    die "--task must contain only letters, numbers, dot, underscore, or hyphen"
    ;;
esac

TASK_DIR_REL="tmp/codex/tasks/$TASK_ID"
TASK_DIR="$REPO_ROOT/$TASK_DIR_REL"
BRIDGE_DIR="$REPO_ROOT/tmp/codex"
DUEL_BRANCH="codex/duel-$TASK_ID"
DUEL_WORKTREE="../quasar-duel-$TASK_ID"

if [ "$TEARDOWN" -eq 1 ]; then
  [ "$MODE" = "duel" ] || die "--teardown is only valid with --mode duel"
  echo "=== teardown ==="
  teardown_rc=0
  if [ -e "$DUEL_WORKTREE" ]; then
    git worktree remove --force "$DUEL_WORKTREE" || teardown_rc=$?
  fi
  if git show-ref --verify --quiet "refs/heads/$DUEL_BRANCH"; then
    git branch -D "$DUEL_BRANCH" || teardown_rc=$?
  fi
  exit "$teardown_rc"
fi

git rev-parse --verify "$BASE^{commit}" >/dev/null 2>&1 || die "base not found: $BASE"
mkdir -p "$TASK_DIR" "$BRIDGE_DIR" || die "cannot create task directories"

read_spec() {
  if [ -n "$SPEC_FILE" ]; then
    [ -f "$SPEC_FILE" ] || die "spec file not found: $SPEC_FILE"
    SPEC_CONTENT="$(cat "$SPEC_FILE")" || die "cannot read spec file: $SPEC_FILE"
  else
    [ ! -t 0 ] || die "spec required via --spec <file> or stdin"
    SPEC_CONTENT="$(cat)" || die "cannot read spec from stdin"
  fi

  [ -n "$SPEC_CONTENT" ] || die "spec is empty"
}

prompt_consult() {
  printf 'Critique this implementation spec before any edits. Identify risks, edge cases, missing tests, and alternatives. Do not modify files.\n\n'
  printf '%s\n\n' "$GUARD_LINE"
  printf 'Do not run npm, next, install commands, dev servers, or full builds.\n\n'
  printf 'SPEC:\n%s\n' "$SPEC_CONTENT"
}

prompt_build() {
  printf 'Implement the spec AND address your own critique above. Edit only source files.\n\n'
  printf '%s\n\n' "$GUARD_LINE"
  printf 'Do not run npm, next, install commands, dev servers, or full builds.\n\n'
  printf 'SPEC:\n%s\n' "$SPEC_CONTENT"
}

prompt_review() {
  printf 'Review ONLY the source diff vs %s for correctness bugs, missing tests, and spec deviations.\n\n' "$BASE"
  printf '%s\n\n' "$GUARD_LINE"
  if [ -n "${SPEC_CONTENT:-}" ]; then
    printf 'SPEC:\n%s\n' "$SPEC_CONTENT"
  fi
}

prompt_guard_findings() {
  printf 'Go deeper: name concrete correctness bugs, race/edge cases, and any missing test for the diff vs %s. List each as a checkbox finding.\n\n' "$BASE"
  printf '%s\n' "$GUARD_LINE"
}

copy_last_message() {
  dest="$1"
  if [ -f "$BRIDGE_DIR/last_message.md" ]; then
    cp "$BRIDGE_DIR/last_message.md" "$dest"
  else
    printf 'No tmp/codex/last_message.md was produced.\n' > "$dest"
  fi
}

run_bridge() {
  bridge_mode="$1"
  dest="$2"
  log="$3"
  prompt_fn="$4"
  shift 4

  : > "$dest"
  "$prompt_fn" | CODEX_BRIDGE_DIR="$BRIDGE_DIR" bash scripts/codex_bridge.sh "$bridge_mode" "$@" > "$log" 2>&1
  rc=$?
  copy_last_message "$dest"
  return "$rc"
}

run_review() {
  out="$1"
  log="$2"
  prompt_fn="$3"

  : > "$out"
  "$prompt_fn" | codex exec review --base "$BASE" \
    -c sandbox_mode=read-only \
    -c windows.sandbox=unelevated \
    -c model_reasoning_effort=xhigh \
    -m gpt-5.5 \
    -o "$out" \
    - > "$log" 2>&1
}

run_test() {
  log="$1"
  : > "$log"

  bash -lc "$TEST_CMD" > >(tee "$log" >/dev/null) 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    TEST_RESULT="PASS"
  else
    TEST_RESULT="FAIL"
    OVERALL_RESULT="FAIL (test)"
    FAILING_COMMAND="$TEST_CMD"
  fi
  return "$rc"
}

write_status() {
  changed_files="$(git diff --name-only "$BASE" -- 2>/dev/null)"
  changed_rc=$?

  {
    printf 'mode: %s\n' "$MODE"
    printf 'base: %s\n' "$BASE"
    printf 'result: %s\n' "$OVERALL_RESULT"
    printf '\nchanged files:\n'
    if [ "$changed_rc" -ne 0 ]; then
      printf -- '- (unable to compute git diff --name-only %s)\n' "$BASE"
    elif [ -n "$changed_files" ]; then
      while IFS= read -r changed_file; do
        printf -- '- %s\n' "$changed_file"
      done <<EOF_CHANGED
$changed_files
EOF_CHANGED
    else
      printf -- '- (none)\n'
    fi
    printf '\ntest result: %s\n' "$TEST_RESULT"
    if [ -n "$FAILING_COMMAND" ]; then
      printf 'failing command: %s\n' "$FAILING_COMMAND"
    fi
  } > "$TASK_DIR/status.md"
}

print_build_summary() {
  echo "SUMMARY"
  echo "mode: $MODE"
  echo "artifacts: $TASK_DIR_REL"
  echo "status: $OVERALL_RESULT"
  if [ -n "$FAILING_COMMAND" ]; then
    echo "failing command: $FAILING_COMMAND"
  fi
  echo "NEXT: Claude reviews \`git diff $BASE\` and adjudicates Codex findings in $TASK_DIR_REL/review.md"
}

print_guard_summary() {
  echo "SUMMARY"
  echo "mode: guard"
  echo "artifacts: $TASK_DIR_REL"
  echo "status: $OVERALL_RESULT"
  if [ -n "$FAILING_COMMAND" ]; then
    echo "failing command: $FAILING_COMMAND"
  fi
  echo "NEXT: Claude reconciles findings in $TASK_DIR_REL/findings.md, fixes or waives each with reason"
}

print_duel_summary() {
  echo "SUMMARY"
  echo "mode: duel"
  echo "artifacts: $TASK_DIR_REL"
  echo "status: $OVERALL_RESULT"
  if [ -n "$FAILING_COMMAND" ]; then
    echo "failing command: $FAILING_COMMAND"
  fi
  echo "Codex's attempt: $DUEL_WORKTREE (diff vs $BASE: \`git -C $DUEL_WORKTREE diff $BASE\`). NEXT: Claude implements its own attempt in main; compare both diffs vs $BASE and pick/merge."
}

run_consult_build() {
  echo "=== consult ==="
  if ! run_bridge consult "$TASK_DIR/consult.md" "$TASK_DIR/consult.log" prompt_consult; then
    OVERALL_RESULT="FAIL (consult)"
    FAILING_COMMAND="bash scripts/codex_bridge.sh consult"
    return 1
  fi

  echo "=== build ==="
  if ! run_bridge build "$TASK_DIR/build.md" "$TASK_DIR/build.log" prompt_build --resume; then
    OVERALL_RESULT="FAIL (build)"
    FAILING_COMMAND="bash scripts/codex_bridge.sh build --resume"
    return 1
  fi

  return 0
}

run_build_mode() {
  read_spec

  if ! run_consult_build; then
    write_status
    print_build_summary
    return 1
  fi

  echo "=== review ==="
  if ! run_review "$TASK_DIR/review.md" "$TASK_DIR/review.log" prompt_review; then
    OVERALL_RESULT="FAIL (review)"
    FAILING_COMMAND="codex exec review --base $BASE"
    write_status
    print_build_summary
    return 1
  fi

  if [ "$TEST_CMD_SET" -eq 1 ]; then
    echo "=== test ==="
    if ! run_test "$TASK_DIR/test.log"; then
      write_status
      print_build_summary
      return 1
    fi
  fi

  write_status
  print_build_summary
  return 0
}

run_guard_mode() {
  git diff --quiet "$BASE" --
  diff_rc=$?
  if [ "$diff_rc" -eq 0 ]; then
    echo "guard mode: no changes to review" >&2
    return 2
  elif [ "$diff_rc" -ne 1 ]; then
    die "cannot compute diff vs $BASE"
  fi

  echo "=== review ==="
  if ! run_review "$TASK_DIR/review.md" "$TASK_DIR/review.log" prompt_review; then
    OVERALL_RESULT="FAIL (review)"
    FAILING_COMMAND="codex exec review --base $BASE"
    write_status
    print_guard_summary
    return 1
  fi

  echo "=== findings ==="
  if ! run_bridge consult "$TASK_DIR/findings.md" "$TASK_DIR/findings.log" prompt_guard_findings --resume; then
    OVERALL_RESULT="FAIL (findings)"
    FAILING_COMMAND="bash scripts/codex_bridge.sh consult --resume"
    write_status
    print_guard_summary
    return 1
  fi

  write_status
  print_guard_summary
  return 0
}

run_duel_mode() {
  read_spec

  echo "=== worktree ==="
  if ! git worktree add -b "$DUEL_BRANCH" "$DUEL_WORKTREE" "$BASE" > "$TASK_DIR/worktree.log" 2>&1; then
    OVERALL_RESULT="FAIL (worktree)"
    FAILING_COMMAND="git worktree add -b $DUEL_BRANCH $DUEL_WORKTREE $BASE"
    write_status
    print_duel_summary
    return 1
  fi

  if ! pushd "$DUEL_WORKTREE" >/dev/null; then
    OVERALL_RESULT="FAIL (worktree)"
    FAILING_COMMAND="cd $DUEL_WORKTREE"
    write_status
    print_duel_summary
    return 1
  fi

  run_consult_build
  duel_rc=$?
  write_status
  popd >/dev/null || return 1

  print_duel_summary
  return "$duel_rc"
}

case "$MODE" in
  build) run_build_mode; exit $? ;;
  guard) run_guard_mode; exit $? ;;
  duel) run_duel_mode; exit $? ;;
esac
