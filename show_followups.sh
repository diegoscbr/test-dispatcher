#!/usr/bin/env bash
set -euo pipefail

echo "===== Phase15_Responses.csv ====="
if [[ -f "Phase15_Responses.csv" ]]; then
  cat "Phase15_Responses.csv"
else
  echo "[missing] Phase15_Responses.csv"
fi
echo
