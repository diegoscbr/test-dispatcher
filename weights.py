#!/usr/bin/env python3
"""
GEO AI Visibility Scoring (V1)

Reads `analysis.csv` (or specify another path) as input and writes `weights.csv` by default.

Writes:
  aivisfinal_MMDDYYYY.csv

Adds ONE column at the end:
  AIS

Notes / assumptions (based on the methodology PDF):
- AIS is a *line-item* weighted contribution score (not the final aggregated 0-100 index).
- Two responses exist per row: p1 (initial search) and p2 (follow-up).
  p1 is weighted more heavily than p2 via configurable multipliers.
- Prompt weight Wp = W_funnel * W_intent (intent == `category` column).
- Model weight Wm can be supplied via an optional JSON file; otherwise defaults to 1.0.
- Source URLs are taken ONLY from p*_sources columns.
- Authority values are on a 0–10 scale (normalized to 0–1 by dividing by 10).
  If there are no sources, authority is ignored and default A=0.5 is used.
- Columns such as p*_position, p*_frequency, p*_authority, p*_sentiment may be
  either single values (for the row brand) OR comma-separated lists aligned to p*_position.
  This script supports both.

Usage:
  python weights.py analysis_02122026.csv
  python weights.py --clients pmg_client.json --model-weights model_weights.json analysis_02122026.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import numpy as np


# -----------------------------
# Configurable constants
# -----------------------------

FUNNEL_WEIGHTS: Dict[str, float] = {
    "awareness": 1.0,
    "consideration": 1.5,
    "conversion": 2.0,
    "post-purchase": 1.2,
    "postpurchase": 1.2,
    "post purchase": 1.2,
}

INTENT_WEIGHTS: Dict[str, float] = {
    "compare & decide": 1.2,
    "recommendations": 1.15,
    "price & value": 1.1,
    "reviews & social proof": 1.05,
    "how-to & setup": 1.0,
    "how to & setup": 1.0,
    "fix & troubleshoot": 0.95,
    "learn & understand": 1.0,
}

# Relative importance of p1 vs p2 (p1 is the initial query; p2 is follow-up).
DEFAULT_P1_MULT: float = 1.0
DEFAULT_P2_MULT: float = 0.7


# -----------------------------
# Utility helpers
# -----------------------------

def _norm_text(x: Any) -> str:
    """Lowercase, strip, collapse whitespace. Safe for None/NaN."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_blank(x: Any) -> bool:
    return _norm_text(x) == ""


def _split_list(val: Any) -> List[str]:
    """
    Split a potentially comma/pipe/semicolon-separated string into a list of trimmed items.
    If val is already a list-like, returns stringified items.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, (list, tuple)):
        return [str(v).strip() for v in val if str(v).strip() != ""]
    s = str(val).strip()
    if s == "":
        return []
    # Choose a delimiter. Prefer "|", then ";", then ",".
    delim = None
    for d in ["|", ";", ","]:
        if d in s:
            delim = d
            break
    if delim is None:
        return [s]
    return [part.strip() for part in s.split(delim) if part.strip() != ""]


_URL_RE = re.compile(r"https?://[^\s,]+")

def _extract_urls(s: Any) -> List[str]:
    """
    Extract URLs from a cell. Uses regex to be robust to comma-separated URL lists.
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    txt = str(s)
    urls = _URL_RE.findall(txt)
    # Strip trailing punctuation
    cleaned = [u.rstrip(").,;]\"'") for u in urls]
    return [u for u in cleaned if u]


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mention_strength(count: int) -> float:
    """V1 mention-strength mapping."""
    if count <= 0:
        return 0.0
    if count == 1:
        return 0.7
    if count in (2, 3):
        return 0.85
    return 1.0


def _sentiment_multiplier(val: Any) -> float:
    """
    Convert a Universal Sentiment Score (-10 to 10) to a multiplier between 0.5 and 1.0.

    If the cell holds a comma-separated list, the caller should already pick the aligned value.
    Non-numeric values default to the neutral multiplier of 0.75.
    """
    default = 0.75
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default

    if isinstance(val, str):
        normalized = val.strip()
        if normalized == "":
            return default
        try:
            x = float(normalized)
        except ValueError:
            return default
    else:
        try:
            x = float(val)
        except Exception:
            return default

    clamped = max(-10.0, min(10.0, x))
    normalized_position = (clamped + 10.0) / 20.0  # maps [-10,10] -> [0,1]
    return 0.5 + 0.5 * normalized_position


def _pick_aligned_value(raw: Any, idx: Optional[int]) -> Optional[Any]:
    """
    If raw is a comma-separated list and idx is provided, returns the idx item.
    If raw is scalar, returns raw.
    """
    if idx is None:
        # no index: if scalar return raw else None
        if isinstance(raw, str) and any(d in raw for d in [",", "|", ";"]):
            return None
        return raw

    # scalar numeric
    if isinstance(raw, (int, float)) and not (isinstance(raw, float) and math.isnan(raw)):
        return raw

    items = _split_list(raw)
    if not items:
        return None
    if idx < 0 or idx >= len(items):
        return None
    return items[idx]


def _to_int_safe(x: Any) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    try:
        return int(float(str(x).strip()))
    except Exception:
        return 0


def _to_float_safe(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


# -----------------------------
# Owned-domain loading
# -----------------------------

_DOMAIN_CANDIDATE_KEYS = {
    "domain",
    "domains",
    "root_domain",
    "root_domains",
    "website",
    "websites",
    "url",
    "urls",
    "site",
    "sites",
    "owned_domains",
    "ownedDomains",
}


def _extract_domains_from_value(v: Any) -> Set[str]:
    domains: Set[str] = set()
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return domains
    values: List[str] = []
    if isinstance(v, str):
        values = _split_list(v)
    elif isinstance(v, (list, tuple)):
        values = [str(x) for x in v]
    else:
        values = [str(v)]

    for item in values:
        item = item.strip()
        if not item:
            continue
        if item.startswith("http://") or item.startswith("https://"):
            d = _domain_from_url(item)
        else:
            # treat as domain-like
            d = item.lower().strip()
            d = d.replace("www.", "")
            d = d.split("/")[0]
        if d:
            domains.add(d)
    return domains


def load_owned_domains_map(pmg_client_path: Optional[Path]) -> Dict[str, Set[str]]:
    """
    Attempts to load a map: normalized_brand_name -> set(domains)

    The JSON structure is not assumed; common patterns are supported:
    - list[ { "name": ..., "domains": [...] } ]
    - { "clients": [ ... ] }
    - { "brands": [ ... ] }
    """
    if pmg_client_path is None:
        return {}
    if not pmg_client_path.exists():
        return {}

    try:
        data = json.loads(pmg_client_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Locate list of client objects
    clients: List[dict] = []
    if isinstance(data, list):
        clients = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict):
        for key in ("clients", "brands", "accounts", "data"):
            if key in data and isinstance(data[key], list):
                clients = [x for x in data[key] if isinstance(x, dict)]
                break
        if not clients:
            # maybe dict is itself a single client record
            if "name" in data:
                clients = [data]
    else:
        return {}

    out: Dict[str, Set[str]] = {}
    for c in clients:
        name = c.get("name") or c.get("brand") or c.get("client") or c.get("Company") or ""
        name_key = _norm_text(name)
        if not name_key:
            continue

        domains: Set[str] = set()
        for k, v in c.items():
            if _norm_text(k) in _DOMAIN_CANDIDATE_KEYS:
                domains |= _extract_domains_from_value(v)

        # If no explicit domains, keep empty; we'll fall back to heuristic later.
        out[name_key] = domains

    return out


def _heuristic_owned_domain_tokens(brand: str) -> Set[str]:
    """
    Fallback heuristic when pmg_client.json doesn't provide domains:
    treat a citation as owned if the domain contains a significant brand token.

    This is intentionally conservative to reduce false positives.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", _norm_text(brand)) if t]
    tokens = [t for t in tokens if len(t) >= 4]  # avoid short tokens like "ai"
    return set(tokens)


def is_owned_citation(brand: str, urls: Sequence[str], owned_domains_map: Dict[str, Set[str]]) -> bool:
    """
    Determine whether any URL is first-party owned for this brand.
    Priority:
      1) pmg_client.json domain mapping (domain suffix match)
      2) heuristic token match
    """
    brand_key = _norm_text(brand)
    owned_domains = owned_domains_map.get(brand_key, set())

    url_domains = [_domain_from_url(u) for u in urls if u]
    url_domains = [d for d in url_domains if d]

    # 1) explicit mapping
    if owned_domains:
        for d in url_domains:
            for owned in owned_domains:
                owned = owned.lower().replace("www.", "")
                if d == owned or d.endswith("." + owned) or d.endswith(owned):
                    return True

    # 2) heuristic fallback
    tokens = _heuristic_owned_domain_tokens(brand)
    if tokens:
        for d in url_domains:
            sld = d.split(".")[0] if d else ""
            # match either in full domain or SLD
            for t in tokens:
                if t in d or t in sld:
                    return True

    return False


# -----------------------------
# Scoring
# -----------------------------

def _find_brand_index(brand: str, positions_raw: Any) -> Optional[int]:
    positions = _split_list(positions_raw)
    if not positions:
        return None

    brand_key = _norm_text(brand)
    for i, p in enumerate(positions):
        if _norm_text(p) == brand_key:
            return i

    # fallback: substring match if exact match fails
    for i, p in enumerate(positions):
        if brand_key and brand_key in _norm_text(p):
            return i

    return None


def compute_turn_sadj(
    row: pd.Series,
    prefix: str,
    brand: str,
    owned_domains_map: Dict[str, Set[str]],
) -> float:
    """
    Compute S_adj,response for either p1 or p2.

    Uses:
      - {prefix}_position
      - {prefix}_frequency
      - {prefix}_authority
      - {prefix}_sentiment
      - {prefix}_sources
    """
    pos_col = f"{prefix}_position"
    freq_col = f"{prefix}_frequency"
    auth_col = f"{prefix}_authority"
    sent_col = f"{prefix}_sentiment"
    src_col = f"{prefix}_sources"
    mention_col = f"{prefix}_mention"  # optional

    idx = _find_brand_index(brand, row.get(pos_col))

    # Mention count
    mention_count = 0
    freq_val = row.get(freq_col)
    if idx is not None:
        aligned = _pick_aligned_value(freq_val, idx)
        mention_count = _to_int_safe(aligned)
    else:
        # If not in positions, we can fall back to mention flag
        if mention_col in row.index:
            mention_count = 1 if _to_int_safe(row.get(mention_col)) > 0 else 0

    brand_mentioned = mention_count > 0 or idx is not None

    # Ranking score
    if idx is None:
        R = 0.0
    else:
        rank = idx + 1
        R = 1.0 / float(rank)

    # Mention strength
    M = _mention_strength(mention_count)

    # Sources / citations (ONLY p*_sources)
    urls = _extract_urls(row.get(src_col))
    citations_exist = len(urls) > 0

    # Citation score C
    if not brand_mentioned:
        C = 0.0
    elif citations_exist:
        C = 1.0 if is_owned_citation(brand, urls, owned_domains_map) else 0.5
    else:
        C = 0.0

    core = 0.20 * M + 0.35 * R + 0.45 * C

    # Authority modifier
    if not brand_mentioned:
        authority_mult = 1.0  # doesn't matter; core will be 0
    else:
        if citations_exist:
            auth_raw = row.get(auth_col)
            auth_aligned = _pick_aligned_value(auth_raw, idx)
            auth_val = _to_float_safe(auth_aligned)

            # authority is 0-10 -> normalize to 0-1
            if auth_val is None:
                A = 0.5
            else:
                A = _clamp01(auth_val / 10.0)
        else:
            # Ignore authority if no sources
            A = 0.5

        authority_mult = 0.7 + 0.3 * A

    # Sentiment modifier
    if not brand_mentioned:
        sent_mult = 1.0
    else:
        sent_raw = row.get(sent_col)
        sent_aligned = _pick_aligned_value(sent_raw, idx)
        sent_mult = _sentiment_multiplier(sent_aligned)

    s_adj = core * authority_mult * sent_mult
    return float(s_adj)


def _get_weight(mapping: Dict[str, float], key: Any, default: float = 1.0) -> float:
    k = _norm_text(key)
    if not k:
        return default
    return float(mapping.get(k, default))


def load_model_weights(path: Optional[Path]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Optional model weighting file.

    Supported formats:
    1) JSON object with either:
       - {"by_model": {...}, "by_owner": {...}}
       - or just {"gpt-...": 1.1, "gemini-...": 1.0} (treated as by_model)
    2) If missing, returns empty dicts (default Wm=1.0).
    """
    if path is None or (not path.exists()):
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    if isinstance(data, dict):
        if "by_model" in data or "by_owner" in data:
            by_model = data.get("by_model") if isinstance(data.get("by_model"), dict) else {}
            by_owner = data.get("by_owner") if isinstance(data.get("by_owner"), dict) else {}
            # normalize keys
            by_model_n = {_norm_text(k): float(v) for k, v in by_model.items()}
            by_owner_n = {_norm_text(k): float(v) for k, v in by_owner.items()}
            return by_model_n, by_owner_n

        # treat as by_model
        by_model_n = {_norm_text(k): float(v) for k, v in data.items()}
        return by_model_n, {}

    return {}, {}


def get_model_weight(row: pd.Series, by_model: Dict[str, float], by_owner: Dict[str, float]) -> float:
    model = _norm_text(row.get("model"))
    owner = _norm_text(row.get("Model_Owner"))
    if model and model in by_model:
        return float(by_model[model])
    if owner and owner in by_owner:
        return float(by_owner[owner])
    return 1.0


def compute_row_ais(
    row: pd.Series,
    owned_domains_map: Dict[str, Set[str]],
    model_weights_by_model: Dict[str, float],
    model_weights_by_owner: Dict[str, float],
    p1_mult: float,
    p2_mult: float,
) -> float:
    brand = row.get("brand", "")
    if _is_blank(brand):
        return 0.0

    # Funnel weight (shared)
    wfunnel = _get_weight(FUNNEL_WEIGHTS, row.get("funnel"), default=1.0)

    # Intent weights: p1 uses `category`; p2 uses `p2_category` when present, otherwise `category`
    wintent_p1 = _get_weight(INTENT_WEIGHTS, row.get("category"), default=1.0)
    wintent_p2 = _get_weight(INTENT_WEIGHTS, row.get("p2_category") if "p2_category" in row.index else row.get("category"), default=wintent_p1)

    wp1 = wfunnel * wintent_p1
    wp2 = wfunnel * wintent_p2

    wm = get_model_weight(row, model_weights_by_model, model_weights_by_owner)

    s1 = compute_turn_sadj(row, "p1", brand, owned_domains_map)
    s2 = compute_turn_sadj(row, "p2", brand, owned_domains_map)

    ais = wm * (p1_mult * wp1 * s1 + p2_mult * wp2 * s2)
    return float(ais)


# -----------------------------
# I/O orchestration
# -----------------------------

def resolve_input_path(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    default_path = Path("analysis.csv")
    if default_path.exists():
        return default_path.resolve()
    raise SystemExit("No input provided and analysis.csv not found in the current directory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GEO AIS line-item scores (adds AIS column).")
    parser.add_argument("input_csv", nargs="?", help="Input CSV file (analysis_MMDDYYYY.csv). If omitted, auto-detect in CWD.")
    parser.add_argument("--clients", default="pmg_client.json", help="Path to pmg_client.json (owned domains). Default: pmg_client.json")
    parser.add_argument("--model-weights", default="model_weights.json", help="Optional model weights JSON. Default: model_weights.json")
    parser.add_argument("--p1-mult", type=float, default=DEFAULT_P1_MULT, help=f"Multiplier for p1 (initial search). Default: {DEFAULT_P1_MULT}")
    parser.add_argument("--p2-mult", type=float, default=DEFAULT_P2_MULT, help=f"Multiplier for p2 (follow-up). Default: {DEFAULT_P2_MULT}")
    parser.add_argument("--output", default=None, help="Optional explicit output path. If omitted, uses weights.csv in the current directory.")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = Path("weights.csv").expanduser().resolve()

    clients_path = Path(args.clients).expanduser().resolve()
    owned_domains_map = load_owned_domains_map(clients_path)

    model_weights_path = Path(args.model_weights).expanduser().resolve()
    model_by_model, model_by_owner = load_model_weights(model_weights_path)

    df = pd.read_csv(input_path)

    # Compute AIS
    df["AIS"] = df.apply(
        lambda r: compute_row_ais(
            r,
            owned_domains_map=owned_domains_map,
            model_weights_by_model=model_by_model,
            model_weights_by_owner=model_by_owner,
            p1_mult=float(args.p1_mult),
            p2_mult=float(args.p2_mult),
        ),
        axis=1,
    )

    if not df.empty:
        top_row = df.iloc[0].to_dict()
        print("Top row detail for debug:")
        for key in sorted(top_row):
            print(f"  - {key}: {top_row[key]}")
    df.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

