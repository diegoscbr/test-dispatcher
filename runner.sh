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

FILES_TO_MOVE=("prompts.py" "query.py" "follow_up_prompts.py" "phase2.py")

for file_name in "${FILES_TO_MOVE[@]}"; do
  if [[ ! -f "$file_name" ]]; then
    echo "File not found in repo: $file_name" >&2
    exit 1
  fi
done

for file_name in "${FILES_TO_MOVE[@]}"; do
  found_path="$(pwd)/$file_name"
  dest_path="$HOME/$file_name"
  echo "Found $file_name at: $found_path"
  mv "$found_path" "$dest_path"
  echo "Moved $file_name to: $dest_path"
done
