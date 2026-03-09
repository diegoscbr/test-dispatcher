#!/usr/bin/env bash
set -euo pipefail

for f in prompts.csv context_prompts.csv responses.csv Phase15_Responses.csv phase2_responses.csv; do
  echo "===== $f ====="
  if [[ -f "$f" ]]; then
    python3 -c "
import sys, re, unicodedata

with open(sys.argv[1], encoding='utf-8-sig') as f:
    text = f.read()

# Normalize unicode (NFKC collapses compatibility chars)
text = unicodedata.normalize('NFKC', text)

# Replace common unicode with ASCII equivalents
replacements = {
    '\u2018': \"'\", '\u2019': \"'\",   # smart single quotes
    '\u201c': '\"', '\u201d': '\"',     # smart double quotes
    '\u2013': '-', '\u2014': '--',      # en/em dash
    '\u2026': '...',                     # ellipsis
    '\u00a0': ' ',                       # non-breaking space
    '\u200b': '', '\u200c': '', '\u200d': '', '\ufeff': '',  # zero-width
}
for old, new in replacements.items():
    text = text.replace(old, new)

# Strip emoji and other non-ASCII symbols (keep basic latin + common punctuation)
def strip_non_ascii(s):
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
        elif unicodedata.category(ch).startswith(('L', 'N', 'P', 'Z')):
            # Keep letters, numbers, punctuation, separators from other scripts
            out.append(ch)
        # Drop symbols (So = emoji, Sk, Sc, Sm) and marks
    return ''.join(out)

text = strip_non_ascii(text)

sys.stdout.write(text)
" "$f"
  else
    echo "[missing] $f"
  fi
  echo
done
