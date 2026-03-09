#!/usr/bin/env bash
set -euo pipefail

FILES=(
  "consumer_ai_integrations.csv"
  "key.csv"
  "phase2_responses_enriched.csv"
)

for csv in "${FILES[@]}"; do
  if [[ ! -f "$csv" ]]; then
    echo "  ⏭  $csv not found, skipping"
    echo
    continue
  fi

  python3 - "$csv" <<'PYEOF'
import csv, sys, shutil

path = sys.argv[1]
term_width = shutil.get_terminal_size((120, 40)).columns

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader, None)
    rows = list(reader)

if not headers:
    print(f"  {path}: empty")
    sys.exit(0)

total = len(rows)
num_cols = len(headers)

col_widths = [len(h) for h in headers]
for row in rows:
    for i, val in enumerate(row):
        if i < num_cols:
            first_line = val.replace("\r", "").split("\n")[0]
            col_widths[i] = max(col_widths[i], len(first_line))

border_space = num_cols + 1 + num_cols * 2
available = term_width - border_space

total_width = sum(col_widths)
if total_width > available:
    min_w = 6
    max_single = max(int(available * 0.6), min_w)
    col_widths = [min(w, max_single) for w in col_widths]
    total_width = sum(col_widths)
    if total_width > available:
        ratio = available / total_width
        col_widths = [max(min_w, int(w * ratio)) for w in col_widths]

def trunc(text, width):
    first_line = text.replace("\r", "").split("\n")[0]
    if len(first_line) <= width:
        return first_line.ljust(width)
    return first_line[:width - 1] + "…"

def sep(l, m, r, f):
    return l + m.join(f * (w + 2) for w in col_widths) + r

def row_str(vals):
    return "│" + "│".join(" " + trunc(v, w) + " " for v, w in zip(vals, col_widths)) + "│"

print(f"\n  {path}  ({total} rows)\n")
print(sep("┌", "┬", "┐", "─"))
print(row_str(headers))
print(sep("├", "┼", "┤", "─"))
for r in rows:
    padded = r + [""] * (num_cols - len(r))
    print(row_str(padded))
print(sep("└", "┴", "┘", "─"))
print(f"\n  {total} rows\n")
PYEOF
done
