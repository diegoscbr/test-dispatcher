#!/usr/bin/env bash
set -euo pipefail

for f in prompts.csv context_prompts.csv responses.csv; do
  echo "===== $f ====="
  if [[ -f "$f" ]]; then
    cat "$f"
  else
    echo "[missing] $f"
  fi
  echo
done
