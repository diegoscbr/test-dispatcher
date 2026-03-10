#!/usr/bin/env bash
set -euo pipefail

FILES=(
  "analysis.csv"
  "weights.csv"
  "aivis_final.csv"
)

for csv in "${FILES[@]}"; do
  if [[ ! -f "$csv" ]]; then
    echo "  [skip] $csv not found"
    echo
    continue
  fi

  python3 - "$csv" <<'PYEOF'
import csv, sys, textwrap, unicodedata

def sanitize(text):
    """Strip emoji and normalize unicode to ASCII-safe output."""
    text = unicodedata.normalize("NFKC", text)
    reps = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...", "\u00a0": " ",
        "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    }
    for old, new in reps.items():
        text = text.replace(old, new)
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
        elif unicodedata.category(ch).startswith(("L", "N", "P", "Z")):
            out.append(ch)
    return "".join(out)

path = sys.argv[1]

with open(path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []
    rows = list(reader)

total = len(rows)
if not total:
    print(f"  {path}: empty")
    sys.exit(0)

label_width = max(len(h) for h in headers)
sep = "=" * 80

print(f"\n  {path}  ({total} rows)")
print(sep)

for idx, row in enumerate(rows, 1):
    print(f"  Row {idx}/{total}")
    print("-" * 80)
    for h in headers:
        val = sanitize((row.get(h) or "").strip())
        label = f"  {h:>{label_width}}: "
        if "\n" in val or len(val) > (78 - label_width):
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
done
