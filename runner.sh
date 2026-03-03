#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_URL="https://github.com/diegoscbr/test-dispatcher.git"
REPO_URL="$DEFAULT_REPO_URL"

if [[ $# -gt 0 && "$1" != --* ]]; then
  REPO_URL="$1"
  shift
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

git clone --depth 1 "$REPO_URL" "$TMP_DIR/repo" >/dev/null 2>&1
cd "$TMP_DIR/repo"

if [[ ! -f "prompts.py" ]]; then
  echo "File not found in repo: prompts.py" >&2
  exit 1
fi

FOUND_PATH="$(pwd)/prompts.py"
DEST_PATH="$HOME/prompts.py"

echo "Found prompts.py at: $FOUND_PATH"
mv "$FOUND_PATH" "$DEST_PATH"
echo "Moved prompts.py to: $DEST_PATH"
