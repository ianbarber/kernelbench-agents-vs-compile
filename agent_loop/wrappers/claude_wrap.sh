#!/usr/bin/env bash
# Wraps `claude` for non-interactive use in the agent loop.
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
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"

PROMPT_CONTENT="$(cat "$PROMPT_FILE")"

mkdir -p "$(dirname "$LOG_FILE")"

cd "$CWD" || exit 70

# --dangerously-skip-permissions: local sandbox, OK.
# --add-dir: ensure tool access to sandbox dir.
# --permission-mode bypassPermissions: belt-and-braces.
claude \
    -p "$PROMPT_CONTENT" \
    --output-format text \
    --dangerously-skip-permissions \
    --add-dir "$CWD" \
    2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
