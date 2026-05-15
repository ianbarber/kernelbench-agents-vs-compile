#!/usr/bin/env bash
# Wraps `codex exec` for non-interactive use in the agent loop.
# Args: --prompt-file <path> --cwd <path> --log-file <path>
set -u
set -o pipefail

PROMPT_FILE=""
CWD=""
LOG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --cwd) CWD="$2"; shift 2 ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 64 ;;
    esac
done

if [[ -z "$PROMPT_FILE" || -z "$CWD" || -z "$LOG_FILE" ]]; then
    echo "usage: $0 --prompt-file <p> --cwd <d> --log-file <l>" >&2
    exit 64
fi

PROJECT_ROOT="/home/ianbarber/Projects/KernelBench"

# Source NVM so the `codex` shim resolves.
export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh" --no-use
nvm use v24.13.0 >/dev/null 2>&1

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"

PROMPT_CONTENT="$(cat "$PROMPT_FILE")"

mkdir -p "$(dirname "$LOG_FILE")"

cd "$CWD" || exit 70

# --dangerously-bypass-approvals-and-sandbox: local sandbox.
# --cd: working root.
# --skip-git-repo-check: sandbox isn't a git repo.
codex exec \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --cd "$CWD" \
    "$PROMPT_CONTENT" \
    2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
