#!/usr/bin/env bash
set -euo pipefail

CSV="${1:-phase2_responses.csv}"

if [[ ! -f "$CSV" ]]; then
  echo "File not found: $CSV"
  exit 1
fi

python3 - "$CSV" <<'PYEOF'
import csv, sys, os, shutil

path = sys.argv[1]

term_width = shutil.get_terminal_size((120, 40)).columns

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader)
    rows = list(reader)

if not headers:
    print("Empty CSV.")
    sys.exit(0)

total = len(rows)
num_cols = len(headers)

# Compute max width per column (header vs data)
col_widths = [len(h) for h in headers]
for row in rows:
    for i, val in enumerate(row):
        if i < num_cols:
            # For table display, truncate cell content; use first line only
            first_line = val.replace("\r", "").split("\n")[0]
            col_widths[i] = max(col_widths[i], len(first_line))

# Cap each column to a reasonable max so the table fits the terminal
# Reserve space for borders: num_cols + 1 pipe chars + 2 spaces per col
border_space = num_cols + 1 + num_cols * 2
available = term_width - border_space

# If total width exceeds terminal, shrink columns proportionally
total_width = sum(col_widths)
if total_width > available:
    # Set a minimum column width
    min_w = 6
    # First, cap any column that's wider than 60% of available
    max_single = max(int(available * 0.6), min_w)
    col_widths = [min(w, max_single) for w in col_widths]
    total_width = sum(col_widths)

    # If still too wide, shrink proportionally
    if total_width > available:
        ratio = available / total_width
        col_widths = [max(min_w, int(w * ratio)) for w in col_widths]

def truncate(text, width):
    """Truncate text to width, adding ellipsis if needed."""
    first_line = text.replace("\r", "").split("\n")[0]
    if len(first_line) <= width:
        return first_line.ljust(width)
    return first_line[:width - 1] + "…"

def make_separator(widths, left, mid, right, fill):
    parts = [fill * (w + 2) for w in widths]
    return left + mid.join(parts) + right

# Table borders (box-drawing characters)
top_border = make_separator(col_widths, "┌", "┬", "┐", "─")
header_sep  = make_separator(col_widths, "├", "┼", "┤", "─")
bottom_border = make_separator(col_widths, "└", "┴", "┘", "─")

def make_row(values, widths):
    cells = []
    for val, w in zip(values, widths):
        cells.append(" " + truncate(val, w) + " ")
    return "│" + "│".join(cells) + "│"

# Print title
print(f"\n  {path}  ({total} rows)\n")

# Print table
print(top_border)
print(make_row(headers, col_widths))
print(header_sep)

for row in rows:
    # Pad row if it has fewer columns than headers
    padded = row + [""] * (num_cols - len(row))
    print(make_row(padded, col_widths))

print(bottom_border)
print(f"\n  {total} rows\n")
PYEOF
