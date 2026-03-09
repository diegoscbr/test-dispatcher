#!/usr/bin/env bash
set -euo pipefail

CSV="${1:-phase2_responses.csv}"

if [[ ! -f "$CSV" ]]; then
  echo "File not found: $CSV"
  exit 1
fi

python3 - "$CSV" <<'PYEOF'
import csv, sys, textwrap

path = sys.argv[1]

# Read CSV, strip BOM from header
with open(path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []
    rows = list(reader)

total = len(rows)
if not total:
    print(f"  {path}: empty")
    sys.exit(0)

# Find the longest header name for label alignment
label_width = max(len(h) for h in headers)
sep = "=" * 80

print(f"\n  {path}  ({total} rows)")
print(sep)

for idx, row in enumerate(rows, 1):
    print(f"  Row {idx}/{total}")
    print("-" * 80)
    for h in headers:
        val = (row.get(h) or "").strip()
        label = f"  {h:>{label_width}}: "
        if "\n" in val or len(val) > (78 - label_width):
            # Multi-line: print label then indented content
            print(label)
            indent = " " * 4
            for line in val.splitlines():
                wrapped = textwrap.fill(line, width=76, initial_indent=indent, subsequent_indent=indent)
                print(wrapped)
        else:
            print(f"{label}{val}")
    print(sep)

print(f"  {total} rows\n")
PYEOF
