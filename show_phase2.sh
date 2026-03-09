#!/usr/bin/env bash
set -euo pipefail

CSV="${1:-phase2_responses.csv}"

if [[ ! -f "$CSV" ]]; then
  echo "File not found: $CSV"
  exit 1
fi

TOTAL=$(tail -n +2 "$CSV" | wc -l | tr -d ' ')
BORDER="$(printf '=%.0s' {1..80})"
THIN="$(printf -- '-%.0s' {1..80})"

echo "$BORDER"
echo "  Phase 2 Responses  --  $CSV  ($TOTAL rows)"
echo "$BORDER"
echo

python3 - "$CSV" <<'PYEOF'
import csv, sys, textwrap

path = sys.argv[1]
wrap = lambda t: "\n".join(textwrap.fill(line, width=76, initial_indent="    ", subsequent_indent="    ") for line in (t or "").splitlines()) or "    (empty)"
thin = "-" * 80

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, 1):
        prompt_id   = row.get("prompt_id", "")
        brand       = row.get("brand", "")
        model       = row.get("model", "")
        category    = row.get("category", "")
        p2_prompt   = row.get("p2_prompt", "")
        p2_response = row.get("p2_response", "")
        p2_category = row.get("p2_category", "")

        print(thin)
        print(f"  #{idx:<4d}  {prompt_id:<24s}  brand: {brand}")
        print(f"        model: {model}")
        print(f"        category: {category:<20s}  p2_category: {p2_category}")
        print(thin)
        print("  USER:")
        print(wrap(p2_prompt))
        print()
        print("  ASSISTANT:")
        print(wrap(p2_response))
        print()
PYEOF

echo "$BORDER"
echo "  Total: $TOTAL rows"
echo "$BORDER"
