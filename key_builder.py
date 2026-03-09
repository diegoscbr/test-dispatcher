#!/usr/bin/env python3
"""Fetch model slugs from all providers and AA intelligence scores in one pass.

Fetches concurrently:
  - Artificial Analysis API  → aa_slug → intelligence_index lookup
  - OpenAI API               → provider slugs
  - OpenRouter API           → provider slugs
  - Gemini API               → provider slugs

Matches each provider slug to an AA canonical slug (same normalization/matching
logic as map_model_slugs.py), attaches the intelligence index inline, and writes:
  - key.csv  (provider, provider_slug, aa_slug, match_type, intelligence_index)
"""

import csv
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

# ── Manual overrides ──────────────────────────────────────────────────────────
MANUAL_OVERRIDES = {
    ("openai", "gpt-5-chat-latest"): "gpt-5-non-reasoning",
    ("gemini", "gemini-flash-latest"): "gemini-2-5-flash",
    ("gemini", "gemini-flash-lite-latest"): "gemini-2-5-flash-lite",
    ("gemini", "gemini-pro-latest"): "gemini-2-5-pro",
    (
        "openrouter",
        "anthropic/claude-3.7-sonnet:thinking",
    ): "claude-3-7-sonnet-thinking",
    ("openrouter", "meta-llama/llama-3.1-405b"): "llama-3-1-instruct-405b",
}  # type: Dict[Tuple[str, str], str]

# ── Skip patterns (no AA equivalent expected) ─────────────────────────────────
SKIP_PATTERNS = [
    "*-audio-*",
    "*-native-audio-*",
    "*-transcribe*",
    "whisper-*",
    "*-tts*",
    "tts-*",
    "dall-e-*",
    "gpt-image-*",
    "sora-*",
    "imagen-*",
    "veo-*",
    "*-image-generation",
    "chatgpt-image-*",
    "text-embedding-*",
    "*-embedding-*",
    "gemini-embedding-*",
    "omni-moderation-*",
    "babbage-002",
    "davinci-002",
    "aqa",
    "nano-banana-*",
    "*-robotics-*",
    "deep-research-*",
    "gpt-realtime*",
    "*-guard-*",
    "openrouter/*",
]  # type: List[str]


def should_skip(slug: str) -> bool:
    return any(fnmatch(slug, pat) for pat in SKIP_PATTERNS)


def normalize(slug: str, provider: str = "") -> str:
    s = slug
    if provider == "openrouter":
        if "/" in s:
            s = s.split("/", 1)[1]
        if ":" in s:
            s = s.split(":")[0]
    s = s.replace(".", "-")
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
    s = re.sub(r"-chat-latest$", "-non-reasoning", s)
    s = re.sub(r"^(gemma-.*)-it$", r"\1", s)
    return s


def _strip_suffix(s: str, suffix: str) -> str:
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s


def try_match(candidate: str, aa_set: Set[str]) -> Optional[Tuple[str, str]]:
    if candidate in aa_set:
        return candidate, "normalized"
    date_stripped = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", candidate)
    if date_stripped != candidate and date_stripped in aa_set:
        return date_stripped, "date_stripped"
    for suffix in ["-001", "-preview", "-latest", "-exp"]:
        stripped = _strip_suffix(candidate, suffix)
        if stripped != candidate and stripped in aa_set:
            return stripped, "suffix_stripped"
    if "-preview" in candidate:
        no_preview = candidate.replace("-preview", "")
        if no_preview in aa_set:
            return no_preview, "suffix_stripped"
    m = re.match(r"^(llama-\d+-\d+)-(\d+b)-instruct$", candidate)
    if m:
        reordered = "{}-instruct-{}".format(m.group(1), m.group(2))
        if reordered in aa_set:
            return reordered, "reordered"
    return None


# ── API fetchers ──────────────────────────────────────────────────────────────


def fetch_aa() -> Dict[str, str]:
    """Returns {aa_slug: intelligence_index} for all AA models."""
    api_key = os.environ.get("AI_ANALYSIS_KEY")
    if not api_key:
        print(
            "Warning: AI_ANALYSIS_KEY not set, skipping Artificial Analysis",
            file=sys.stderr,
        )
        return {}
    resp = requests.get(
        "https://artificialanalysis.ai/api/v2/data/llms/models",
        headers={"x-api-key": api_key},
    )
    resp.raise_for_status()
    result = {}
    for m in resp.json()["data"]:
        slug = (m.get("slug") or "").strip()
        evals = m.get("evaluations") or {}
        score = evals.get("artificial_analysis_intelligence_index")
        if slug:
            result[slug] = str(score) if score is not None else ""
    return result


def fetch_openai() -> List[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set, skipping OpenAI", file=sys.stderr)
        return []
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json()["data"])


def fetch_openrouter() -> List[str]:
    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if not api_key:
        print("Warning: OPEN_ROUTER_KEY not set, skipping OpenRouter", file=sys.stderr)
        return []
    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json()["data"])


def fetch_gemini() -> List[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set, skipping Gemini", file=sys.stderr)
        return []
    resp = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key, "pageSize": 1000},
    )
    resp.raise_for_status()
    return sorted(m["name"].removeprefix("models/") for m in resp.json()["models"])


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Fetch all 4 APIs concurrently
    tasks = {
        "aa": fetch_aa,
        "openai": fetch_openai,
        "openrouter": fetch_openrouter,
        "gemini": fetch_gemini,
    }
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                count = len(results[name])
                print(f"  {name}: {count} models")
            except Exception as exc:
                results[name] = {} if name == "aa" else []
                print(f"  {name}: ERROR - {exc}", file=sys.stderr)

    aa_scores: Dict[str, str] = results["aa"]  # slug → intelligence_index
    aa_slugs: Set[str] = set(aa_scores.keys())

    print(f"\nAA slugs loaded: {len(aa_slugs)}")

    # Match provider slugs → AA slugs, attach scores inline
    output_rows = []
    unmatched = []

    for provider in ("openai", "gemini", "openrouter"):
        seen: Set[str] = set()
        for slug in results[provider]:
            if not slug or slug in seen:
                continue
            seen.add(slug)

            bare = (
                slug.split("/", 1)[1]
                if provider == "openrouter" and "/" in slug
                else slug
            )

            if should_skip(bare):
                output_rows.append(
                    {
                        "provider": provider,
                        "provider_slug": slug,
                        "aa_slug": "",
                        "match_type": "skipped",
                        "intelligence_index": "",
                    }
                )
                continue

            key = (provider, slug)
            if key in MANUAL_OVERRIDES:
                aa_slug = MANUAL_OVERRIDES[key]
                output_rows.append(
                    {
                        "provider": provider,
                        "provider_slug": slug,
                        "aa_slug": aa_slug,
                        "match_type": "manual",
                        "intelligence_index": aa_scores.get(aa_slug, ""),
                    }
                )
                continue

            if slug in aa_slugs:
                output_rows.append(
                    {
                        "provider": provider,
                        "provider_slug": slug,
                        "aa_slug": slug,
                        "match_type": "exact",
                        "intelligence_index": aa_scores.get(slug, ""),
                    }
                )
                continue

            candidate = normalize(slug, provider)
            match = try_match(candidate, aa_slugs)
            if match:
                aa_slug, match_type = match
                output_rows.append(
                    {
                        "provider": provider,
                        "provider_slug": slug,
                        "aa_slug": aa_slug,
                        "match_type": match_type,
                        "intelligence_index": aa_scores.get(aa_slug, ""),
                    }
                )
            else:
                output_rows.append(
                    {
                        "provider": provider,
                        "provider_slug": slug,
                        "aa_slug": "",
                        "match_type": "unmatched",
                        "intelligence_index": "",
                    }
                )
                unmatched.append((provider, slug))

    # Write output
    output_file = Path(__file__).parent / "key.csv"
    fieldnames = [
        "provider",
        "provider_slug",
        "aa_slug",
        "match_type",
        "intelligence_index",
    ]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Summary
    counts = Counter(r["match_type"] for r in output_rows)
    print(f"\nWrote {len(output_rows)} rows to {output_file.name}\n")
    print("Match type breakdown:")
    for mtype in [
        "exact",
        "normalized",
        "date_stripped",
        "suffix_stripped",
        "reordered",
        "manual",
        "skipped",
        "unmatched",
    ]:
        if counts[mtype]:
            print(f"  {mtype:20s} {counts[mtype]:4d}")

    if unmatched:
        print(f"\n── Unmatched ({len(unmatched)}) ── review for manual overrides ──")
        for provider, slug in unmatched:
            print(
                f"  [{provider:10s}] {slug:45s} (normalized: {normalize(slug, provider)})"
            )


if __name__ == "__main__":
    main()
