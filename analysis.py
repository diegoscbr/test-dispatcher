
#!/usr/bin/env python3
"""
Analyze LLM responses for brand presence, sources, and basic perception signals.

Input:
  - phase2_responses_enriched.csv (Phase 1 export that now carries Phase 2 prompts/responses plus persona/brand metadata)
  - brand.csv or pmgclient.json (must include brand_id/data or id/slug plus brand/name; domain or source URL is optional)

Output:
  - analysis.csv with columns:
    base metadata (response_id, brand_id, brand, prompt_id, category)
    plus per-stage metrics (p1_*, p2_*) covering prompt, response, mention, snippet, page, position,
    word count, authority, frequency, sentiment, sources, and model.

Notes / heuristics:
  - "citation" captures the URL that follows the brand mention (or brand site itself) when present; multiple matches are comma-separated.
  - "sources" carries the comma-separated URLs extracted from the response when available.
  - "page" is the path component of the brand's `source` metadata (e.g., `/pages/contact-us`) when the brand is mentioned.
  - "position" is the current brand's rank compared to competitors based on how prominently each brand appears.
    If only the brand is mentioned -> 1. If the brand is absent -> blank.
  - "word_count" sums the words from the sentences that cite or mention our brand, isolating only the text attributed to those mentions/citations.
  - "authority" mirrors word_count/frequency by returning comma-separated scores for each brand in the response order. Each score maps to the rated authority of the cited content for that brand; brands without cited content return a blank slot.
  - "frequency" now counts how often the brand is mentioned or cited within that response (1 per mention/citation).

  - Phase 1 metrics prefer `p1c_response` when supplied but otherwise use the original `response`.
  - `p1c_prompt` and `p1c_response` are retained as top-level columns to surface the customer-provided follow-up.
  - "mention" is 1 when the brand text appears in the response, regardless of citations.
  - "authority" is a comma-separated list of 1–10 scores following each detected brand (10 = strongest); entries stay blank for brands without cited content.
  - "sentiment" is mapped to an integer -10..10 scale (10 = most positive). Both are left blank if the brand is absent from the response.
  - All columns from the input CSV are preserved in `analysis.csv` so you can trace each metric row back to the original phase2_responses_enriched payload.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Pattern, Sequence, Set, Tuple, Union

from urllib.parse import urlparse

import openai
import pandas as pd
from model_usage_logger import ModelUsageLogger, extract_usage_from_response

ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# -----------------------------
# Source typing (citation categorization)
# -----------------------------

# Keep taxonomy small + practical.
SOURCE_CATEGORIES: List[str] = [
    "direct",
    "social",
    "affiliate",
    "marketplace",
    "forum",
    "publisher",
    "documentation",
    "aggregator",
    "blog",
    "unknown",
]

SOCIAL_DOMAINS = {
    "x.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "linkedin.com",
    "youtube.com",
    "youtu.be",
    "pinterest.com",
    "threads.net",
    "reddit.com",  # also a forum, but treat as social by default here
}

FORUM_DOMAINS = {
    "stackoverflow.com",
    "stackexchange.com",
    "quora.com",
    "discourse.org",
}

MARKETPLACE_DOMAINS = {
    "amazon.com",
    "amzn.to",
    "walmart.com",
    "target.com",
    "ebay.com",
    "etsy.com",
    "bestbuy.com",
    "homedepot.com",
    "lowes.com",
    "appstore.apple.com",
    "apps.apple.com",
    "play.google.com",
}

DOC_DOMAINS = {
    "wikipedia.org",
    "developer.mozilla.org",
    "mdn.dev",
    "support.google.com",
    "support.apple.com",
    "learn.microsoft.com",
    "docs.microsoft.com",
    "github.com",
    "gitlab.com",
    "readthedocs.io",
}

# Small starter list; expand as needed.
PUBLISHER_DOMAINS = {
    "nytimes.com",
    "wsj.com",
    "washingtonpost.com",
    "cnn.com",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "forbes.com",
    "techcrunch.com",
    "theverge.com",
    "wired.com",
}

# Affiliate networks + tracking systems.
AFFILIATE_DOMAINS = {
    "impact.com",
    "impactradius.com",
    "cj.com",
    "linksynergy.com",
    "shareasale.com",
    "awin1.com",
    "awin.com",
    "rakutenadvertising.com",
    "rakuten.com",
    "partnerize.com",
    "skimlinks.com",
    "viglink.com",
}

AFFILIATE_PARAM_RE = re.compile(r"(?i)(?:\bref\b=|\baff\b=|affiliate|utm_source=affiliate|utm_medium=affiliate|tag=)")

AGGREGATOR_HINT_RE = re.compile(r"(?i)(review|reviews|rating|ratings|compare|comparison|best-|top-?\d+|listicle|roundup)")

DEFAULT_EMBEDDING_MODEL = os.environ.get("AVIS_EMBEDDING_MODEL", "text-embedding-3-large")
_USAGE_LOGGER = ModelUsageLogger(Path(__file__), identifier_label="text")


def _norm_host(host: str) -> str:
    host = (host or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_from_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
        return _norm_host(parsed.netloc)
    except Exception:
        return ""


def _normalize_domain_host(value: str) -> str:
    if not value:
        return ""
    host = _host_from_url(value)
    if host:
        return host
    return _norm_host(value)


def _normalize_brand_key(value: str) -> str:
    clean = (value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", clean)


def _is_subdomain_or_equal(host: str, domain: str) -> bool:
    host = _norm_host(host)
    domain = _norm_host(domain)
    if not host or not domain:
        return False
    return host == domain or host.endswith("." + domain)


def _domain_hosts_for_urls(urls: Sequence[str], domain: str) -> str:
    domain_host = _normalize_domain_host(domain)
    if not domain_host:
        return ""
    matches: List[str] = []
    seen: Set[str] = set()
    for url in urls or []:
        host = _host_from_url(url)
        if not host:
            continue
        if _is_subdomain_or_equal(host, domain_host) and host not in seen:
            seen.add(host)
            matches.append(host)
    return ", ".join(matches)


def _classify_url_source_type(url: str, brand_domain: str = "") -> str:
    """Return one SOURCE_CATEGORIES label for a URL."""
    host = _host_from_url(url)
    url_lc = (url or "").lower()

    # 1) Direct (brand-owned) — highest priority
    if brand_domain and _is_subdomain_or_equal(host, brand_domain):
        return "direct"

    # 2) Social
    if host in SOCIAL_DOMAINS or any(host.endswith("." + d) for d in SOCIAL_DOMAINS):
        return "social"

    # 3) Forum
    if host in FORUM_DOMAINS or any(host.endswith("." + d) for d in FORUM_DOMAINS):
        return "forum"

    # 4) Marketplace
    if host in MARKETPLACE_DOMAINS or any(host.endswith("." + d) for d in MARKETPLACE_DOMAINS):
        return "marketplace"

    # 5) Documentation / reference
    if host in DOC_DOMAINS or any(host.endswith("." + d) for d in DOC_DOMAINS):
        return "documentation"

    # 6) Affiliate (networks or tracking parameters)
    if host in AFFILIATE_DOMAINS or any(host.endswith("." + d) for d in AFFILIATE_DOMAINS):
        return "affiliate"
    if AFFILIATE_PARAM_RE.search(url_lc):
        return "affiliate"

    # 7) Publisher / media
    if host in PUBLISHER_DOMAINS or any(host.endswith("." + d) for d in PUBLISHER_DOMAINS):
        return "publisher"

    # 8) Aggregator / review signals
    if AGGREGATOR_HINT_RE.search(host) or AGGREGATOR_HINT_RE.search(url_lc):
        return "aggregator"

    # 9) Blog (default fallback for normal websites)
    if host:
        return "blog"

    return "unknown"


def bucket_urls_by_source_type(urls: List[str], brand_domain: str = "") -> Dict[str, List[str]]:
    """Bucket URLs into {category: [urls...]} using SOURCE_CATEGORIES ordering."""
    buckets: Dict[str, List[str]] = {k: [] for k in SOURCE_CATEGORIES}
    for u in urls or []:
        if not u:
            continue
        cat = _classify_url_source_type(u, brand_domain=brand_domain)
        if cat not in buckets:
            cat = "unknown"
        buckets[cat].append(u)
    return buckets


# -----------------------------
# Regex + lightweight helpers
# -----------------------------

URL_RE = re.compile(r"(https?://[^\s\)\]\}<>\"']+)", re.IGNORECASE)
DEFAULT_RESPONSES_PATH = "phase2_responses_enriched.csv"

BASE_OUTPUT_FIELDS = [
    "date",
    "response_id",
    "brand_id",
    "brand",
    "prompt_id",
    "category",
    "model",
    "relevance",
]
P1C_FIELDS = ["p1c_prompt", "p1c_response"]

FOLLOW_UP_PATTERNS = [
    re.compile(r"\bneed to know\b", re.IGNORECASE),
    re.compile(r"\bin order to\b", re.IGNORECASE),
    re.compile(r"\bcould you tell\b", re.IGNORECASE),
    re.compile(r"\bplease (?:tell|share)\b", re.IGNORECASE),
    re.compile(r"\bwhat is your\b", re.IGNORECASE),
    re.compile(r"\bhow old are you\b", re.IGNORECASE),
    re.compile(r"\bwhere (?:are|do)\b.*\byou\b", re.IGNORECASE),
    re.compile(r"\bmay i\b", re.IGNORECASE),
    re.compile(r"\bwould you\b", re.IGNORECASE),
    re.compile(r"\byour (?:age|location|address)\b", re.IGNORECASE),
]


# Sentiment lexicons (lightweight, deterministic)
SENT_POS = {
    "best",
    "great",
    "excellent",
    "reliable",
    "trusted",
    "recommended",
    "strong",
    "top",
    "leading",
    "high-quality",
    "high",
    "quality",
    "good",
    "impressive",
    "easy",
    "useful",
    "love",
    "liked",
    "fast",
    "accurate",
}
SENT_NEG = {
    "bad",
    "poor",
    "terrible",
    "unreliable",
    "avoid",
    "scam",
    "fake",
    "worst",
    "complaint",
    "complaints",
    "issue",
    "issues",
    "problem",
    "problematic",
    "disappointing",
    "hard",
    "difficult",
    "expensive",
    "overpriced",
    "lawsuit",
    "slow",
    "inaccurate",
}

HEDGING_TERMS = {
    "may",
    "might",
    "possibly",
    "could",
    "often",
    "generally",
    "typically",
    "sometimes",
    "seems",
    "appears",
    "likely",
    "unlikely",
}

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by",
    "with","from","as","is","are","was","were","be","been","being","it","this","that","these","those","you","your",
    "they","their","we","our","i","me","my","he","she","his","her","them","who","what","where","why","how","can",
    "will","would","should","could","may","might","do","does","did","done","have","has","had","having","not","no","yes",
}


def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(1).rstrip(".,;:)]}>") for m in URL_RE.finditer(text)]


def _url_spans(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    return [(m.start(), m.end()) for m in URL_RE.finditer(text)]


def _is_index_in_spans(idx: int, spans: Sequence[Tuple[int, int]]) -> bool:
    for start, end in spans:
        if start <= idx < end:
            return True
    return False


def _text_without_urls(text: str) -> str:
    if not text:
        return ""
    return URL_RE.sub(" ", text)


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Very simple sentence splitter
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# --- Deterministic helpers for tokenization and cosine similarity ---
def _tokenize(text: str) -> List[str]:
    # Lowercase alnum tokenization; keeps simple, deterministic behavior.
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _tf_vector(tokens: List[str]) -> Dict[str, int]:
    tf: Dict[str, int] = {}
    for t in tokens:
        if not t or t in STOPWORDS:
            continue
        tf[t] = tf.get(t, 0) + 1
    return tf


def _cosine_sim(tf_a: Dict[str, int], tf_b: Dict[str, int]) -> float:
    if not tf_a or not tf_b:
        return 0.0
    # Dot product
    dot = 0.0
    for k, va in tf_a.items():
        vb = tf_b.get(k)
        if vb:
            dot += float(va) * float(vb)
    # Norms
    na = sum(float(v) * float(v) for v in tf_a.values()) ** 0.5
    nb = sum(float(v) * float(v) for v in tf_b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _cosine_similarity_vectors(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = sum(float(a) * float(a) for a in vec_a) ** 0.5
    norm_b = sum(float(b) * float(b) for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _build_embedding_client(api_key: Optional[str]):
    if not api_key:
        return None
    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=api_key)
    openai.api_key = api_key
    return openai


def _request_embeddings(
    texts: List[str],
    client: object,
    api_key: Optional[str],
    model: str,
    identifier_value: str = "",
) -> List[List[float]]:
    if not texts:
        return []
    print(f"DEBUG: Requesting embeddings | texts={len(texts)} | model={model}")
    start = time.perf_counter()
    response = None
    try:
        if client is not None:
            response = client.embeddings.create(model=model, input=texts)
            result = [item.embedding for item in response.data]
            return result
        if api_key:
            openai.api_key = api_key
        response = openai.Embedding.create(model=model, input=texts)
        result = [item["embedding"] for item in response["data"]]
        return result
    except Exception as e:
        print("ERROR: Embedding request failed:", repr(e))
        return []
    finally:
        duration = time.perf_counter() - start
        _USAGE_LOGGER.record(
            model,
            duration,
            extract_usage_from_response(response),
            identifier_value=identifier_value,
        )


def _get_cached_embedding(
    text: str,
    cache: Dict[str, Optional[List[float]]],
    client: object,
    api_key: Optional[str],
    model: str,
) -> Optional[List[float]]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return None
    if normalized in cache:
        print("DEBUG: Embedding cache hit")
        return cache[normalized]
    print("DEBUG: Embedding cache miss; requesting embedding")
    embeddings = _request_embeddings(
        [normalized],
        client,
        api_key,
        model,
        identifier_value=normalized,
    )
    vector = embeddings[0] if embeddings else None
    cache[normalized] = vector
    return vector


def _compute_prompt_summary_relevance(
    prompt: str,
    summary: str,
    cache: Dict[str, Optional[List[float]]],
    client: object,
    api_key: Optional[str],
    model: str,
) -> Optional[float]:
    if not api_key:
        return None
    prompt_text = (prompt or "").strip()
    summary_text = (summary or "").strip()
    print("DEBUG: Computing relevance")
    print("  prompt length:", len(prompt_text))
    print("  summary length:", len(summary_text))
    if not prompt_text or not summary_text:
        return None
    prompt_vec = _get_cached_embedding(prompt_text, cache, client, api_key, model)
    summary_vec = _get_cached_embedding(summary_text, cache, client, api_key, model)
    if not prompt_vec or not summary_vec:
        if not prompt_vec:
            print("DEBUG: prompt embedding is None")
        if not summary_vec:
            print("DEBUG: summary embedding is None")
        return None
    cosine = _cosine_similarity_vectors(prompt_vec, summary_vec)
    score = (cosine + 1.0) * 5.0
    return round(max(0.0, min(10.0, score)), 2)


def _normalize_brand_sentence(sentence: str) -> str:
    """Clean a sentence so it only contains the words attributed to a brand mention."""
    normalized = sentence
    for url in extract_urls(sentence):
        normalized = normalized.replace(url, "")
    normalized = re.sub(r"\s+", " ", normalized).strip(" \t\n\r.,;:()[]<>\"'")
    return normalized


def snippet_for_brand_mention(text: str, brand: str) -> str:
    """Return the sentence containing the brand (minus inline citations)."""
    variants = [variant.lower() for variant in _brand_detection_variants(brand)]
    if not text or not variants:
        return ""
    for sentence in split_sentences(text):
        low_sentence = sentence.lower()
        if not any(variant and variant in low_sentence for variant in variants):
            continue
        return _normalize_brand_sentence(sentence)
    return ""


def count_words_for_brand_mentions(text: str, brand: str) -> int:
    """Sum the words from each sentence that directly mentions the brand."""
    variants = [variant.lower() for variant in _brand_detection_variants(brand) if variant]
    if not text or not variants:
        return 0
    total = 0
    for sentence in split_sentences(text):
        low_sentence = sentence.lower()
        if not any(variant in low_sentence for variant in variants):
            continue
        normalized = _normalize_brand_sentence(sentence)
        if not normalized:
            continue
        words = re.findall(r"\b[\w']+\b", normalized)
        total += len(words)
    return total


def count_words_for_brand_cited_mentions(text: str, brand: str, brand_domain: str) -> int:
    """Count words only when the brand sentence cites the provided brand-owned domain."""
    domain = (brand_domain or "").strip()
    if not domain:
        return 0

    domain_host = _host_from_url(domain)
    if not domain_host:
        domain_host = _norm_host(domain)
    if not domain_host:
        return 0

    variants = [variant.lower() for variant in _brand_detection_variants(brand) if variant]
    if not text or not variants:
        return 0

    total = 0
    for sentence in split_sentences(text):
        low_sentence = sentence.lower()
        if not any(variant in low_sentence for variant in variants):
            continue

        urls = extract_urls(sentence)
        has_owned_url = False
        for url in urls:
            url_host = _host_from_url(url)
            if url_host and _is_subdomain_or_equal(url_host, domain_host):
                has_owned_url = True
                break
        if not has_owned_url:
            continue

        normalized = _normalize_brand_sentence(sentence)
        if not normalized:
            continue

        words = re.findall(r"\b[\w']+\b", normalized)
        total += len(words)
    return total


def detects_follow_up_question(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    for pattern in FOLLOW_UP_PATTERNS:
        if pattern.search(text):
            return True
    if "?" in text:
        # treat question as follow-up when it clearly asks for personal/user info
        if re.search(r"\b(your|you)\b", lower):
            return True
    return False


def _find_brand_variant_index(text: str, brand: str) -> int:
    """Return first index of the brand or its alias within `text` (case-insensitive)."""
    if not text or not brand:
        return -1
    low = text.lower()
    for variant in _brand_detection_variants(brand):
        idx = low.find(variant.lower())
        if idx >= 0:
            return idx
    return -1


def find_brand_mentions(text: str, brands: List[str]) -> Dict[str, int]:
    """
    Return map of brand -> first character index mention in text (case-insensitive).
    Uses word-boundary-ish matching; still tolerant for punctuation.
    """
    mentions: Dict[str, int] = {}
    if not text:
        return mentions

    for b in brands:
        b2 = (b or "").strip()
        if not b2:
            continue
        idx = _find_brand_variant_index(text, b2)
        if idx >= 0:
            mentions[b2] = idx
    return mentions

def brand_order_from_response(text: str, brands: Sequence[str]) -> List[str]:
    """Return the brand names in the order they first appear in `text`."""
    mentions = find_brand_mentions(text, [b for b in brands if b])
    if not mentions:
        return []
    ordered = sorted(mentions.items(), key=lambda pair: pair[1])
    return [name for name, _ in ordered]


def _merge_brand_candidates(base_brands: Sequence[str], competitor_brands: Sequence[str]) -> List[str]:
    """Combine a base brand list with competitors while preserving first-seen order."""
    seen: Set[str] = set()
    merged: List[str] = []

    for name in base_brands:
        cleaned = (name or "").strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)

    for name in competitor_brands:
        cleaned = (name or "").strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)

    return merged


def word_counts_for_brand_sequence(
    text: str, brands: Sequence[str], brand_domains: Optional[Mapping[str, str]] = None
) -> str:
    """Return comma-separated word counts for each brand, based on domain-attributed sentences."""
    if not text or not brands:
        return ""
    brand_domains = brand_domains or {}
    counts: List[str] = []
    for brand in brands:
        domain = (brand_domains.get(brand) or "").strip()
        if not domain:
            counts.append("0")
            continue
        count = count_words_for_brand_cited_mentions(text, brand, domain)
        counts.append(str(count))
    return ", ".join(counts)


def _brand_sentences(text: str, brand: str) -> list[str]:
    """Collect the sentences that mention a given brand."""
    if not text or not brand:
        return []
    variants = [variant.lower() for variant in _brand_detection_variants(brand) if variant]
    if not variants:
        return []
    sentences: list[str] = []
    for sentence in split_sentences(text):
        low_sentence = sentence.lower()
        if any(variant in low_sentence for variant in variants):
            sentences.append(sentence)
    return sentences


def authority_scores_for_brand_sequence(
    text: str,
    brands: Sequence[str],
    brand_domains: Optional[Mapping[str, str]] = None,
) -> str:
    """Return comma-separated authority scores for each brand (blank when no citations)."""
    if not text or not brands:
        return ""
    brand_domains = brand_domains or {}
    scores: list[str] = []
    for brand in brands:
        citations = find_citations_after_brand(text, brand)
        if not citations:
            scores.append("")
            continue
        sentences = _brand_sentences(text, brand)
        if not sentences:
            scores.append("")
            continue
        domain = (brand_domains.get(brand) or "").strip()
        block_text = " ".join(sentences)
        score = authority_uas_score(block_text, citations, brand_domain=domain)
        scores.append(str(score))
    return ", ".join(scores)


def sentiment_scores_for_brand_sequence(text: str, brands: Sequence[str]) -> str:
    """Return comma-separated Universal Sentiment Scores for each brand in order."""
    if not text or not brands:
        return ""
    scores: list[str] = []
    for brand in brands:
        clean_brand = (brand or "").strip()
        if not clean_brand:
            scores.append("")
            continue
        score = sentiment_uss_score(text, clean_brand)
        scores.append(str(score))
    return ", ".join(scores)


def frequency_for_brand_sequence(
    text: str, brands: Sequence[str], brand_domains: Optional[Mapping[str, str]] = None
) -> str:
    """Return comma-separated occurrence counts for the brand order provided."""
    if not text or not brands:
        return ""
    brand_domains = brand_domains or {}
    counts: List[str] = []
    for brand in brands:
        domain = (brand_domains.get(brand) or "").strip()
        counts.append(str(count_brand_occurrences(text, brand, brand_domain=domain)))
    return ", ".join(counts)


# --- New helper: extract_rank_candidates ---
def extract_rank_candidates(text: str) -> List[Tuple[str, int]]:
    """Extract ordered candidate item names (product/brand mentions) for ranking.

    This is designed for common LLM list formats, including:
      - Markdown bold headings: **Product Name**
      - Short title-case lines (often followed by description)

    Returns a list of (candidate_text, start_index) in appearance order.
    """
    if not text:
        return []

    candidates: List[Tuple[str, int]] = []

    # 1) Markdown bold headings: **Name**
    for m in re.finditer(r"\*\*(.+?)\*\*", text, flags=re.DOTALL):
        name = re.sub(r"\s+", " ", (m.group(1) or "").strip())
        # Guardrails: avoid extremely short / generic captures
        if 3 <= len(name) <= 140 and not name.endswith(":"):
            candidates.append((name, m.start()))

    # 2) Title-like lines (non-bulleted or bulleted) that look like product names
    #    Example: "NuFACE Trinity+ Starter Kit" followed by description lines.
    lines = text.splitlines()
    cursor = 0
    for line in lines:
        raw = line
        line_len = len(raw)
        line_start = cursor
        cursor += line_len + 1  # +1 for the newline

        s = raw.strip()
        if not s:
            continue

        # Strip common bullets / numbering
        s2 = re.sub(r"^\s*(?:[-•\*]|\d+[\)\.]|\u2022)\s+", "", s)
        s2 = s2.strip()

        # Skip lines that are clearly sentences
        if s2.endswith(".") or s2.endswith("?") or s2.endswith("!"):
            continue

        # Heuristic: title-case-ish (many capitalized words) and not too long
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\+&'\-]*", s2)
        if not (2 <= len(words) <= 14):
            continue

        cap_words = sum(1 for w in words if re.match(r"^[A-Z0-9]", w))
        if cap_words < max(1, int(0.5 * len(words))):
            continue

        # Avoid generic headings
        low = s2.lower()
        if low in {"overview", "summary", "conclusion", "pros", "cons"}:
            continue
        
        


        # Record
        if 3 <= len(s2) <= 140:
            candidates.append((s2, line_start))

    # De-dupe by normalized candidate text, preserve earliest occurrence
    seen: set[str] = set()
    out: List[Tuple[str, int]] = []
    for name, pos in sorted(candidates, key=lambda x: x[1]):
        key = re.sub(r"\s+", " ", name.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((name, pos))

    return out



class BrandMentionDetector:
    def __init__(self, client: object, api_key: Optional[str], model: str, threshold: float = 0.70):
        self.client = client
        self.api_key = api_key
        self.model = model
        self.threshold = threshold
        self.cache: Dict[str, Optional[List[float]]] = {}

    def _embed(self, text: str) -> Optional[List[float]]:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return None
        if normalized in self.cache:
            return self.cache[normalized]
        vectors = _request_embeddings(
            [normalized],
            self.client,
            self.api_key,
            self.model,
            identifier_value=normalized,
        )
        vec = vectors[0] if vectors else None
        self.cache[normalized] = vec
        return vec

    def is_mention(self, brand: str, text: str) -> bool:
        brand_vec = self._embed(brand)
        text_vec = self._embed(text)
        if not brand_vec or not text_vec:
            return False
        score = _cosine_similarity_vectors(brand_vec, text_vec)
        return score >= self.threshold


_MENTION_DETECTOR: Optional[BrandMentionDetector] = None


def _brand_tokens(brand: str) -> List[str]:
    normalized = (brand or "").lower()
    return [tok for tok in re.findall(r"[a-z0-9]+", normalized) if tok]

GENERIC_BRAND_SUFFIXES: Set[str] = {
    "service",
    "services",
    "solution",
    "solutions",
    "platform",
    "platforms",
    "app",
    "apps",
    "studio",
    "studios",
    "lab",
    "labs",
}

def _strip_generic_brand_suffixes(name: str) -> str:
    parts = [part for part in re.split(r"\s+", (name or "").strip()) if part]
    while parts:
        tail = parts[-1].strip(".,;:()[]{}").lower()
        if tail in GENERIC_BRAND_SUFFIXES:
            parts.pop()
            continue
        break
    if not parts:
        return (name or "").strip()
    return " ".join(parts)

def _brand_detection_variants(brand: str) -> List[str]:
    normalized = (brand or "").strip()
    if not normalized:
        return []
    variants: List[str] = []
    seen: Set[str] = set()

    def _add(candidate: str) -> None:
        candidate = (candidate or "").strip()
        if not candidate:
            return
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(candidate)

    _add(normalized)
    base = _strip_generic_brand_suffixes(normalized)
    _add(base)
    return variants


def _text_matches_tokens(text: str, tokens: List[str]) -> bool:
    if not tokens or len(tokens) < 2:
        return False
    normalized_text = text or ""
    pattern = (
        r"(?i)(?<![A-Za-z0-9])"
        + r"\W+".join(re.escape(tok) for tok in tokens)
        + r"(?![A-Za-z0-9])"
    )
    return bool(re.search(pattern, normalized_text))


def _brand_literal_regex(brand: str) -> Optional[re.Pattern]:
    normalized = (brand or "").strip()
    if not normalized:
        return None
    escaped = re.escape(normalized)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _brand_variant_matches_text(variant: str, text: str) -> bool:
    literal_pattern = _brand_literal_regex(variant)
    if literal_pattern and literal_pattern.search(text):
        return True
    tokens = _brand_tokens(variant)
    if tokens and _text_matches_tokens(text, tokens):
        return True
    if _MENTION_DETECTOR:
        try:
            return _MENTION_DETECTOR.is_mention(variant, text)
        except Exception:
            pass
    return False


def detect_brand_presence(brand: str, text: str) -> bool:
    if not brand or not text:
        return False
    if get_brand_positions(text, brand):
        return True
    sanitized = _text_without_urls(text)
    if not sanitized:
        return False
    for variant in _brand_detection_variants(brand):
        if _brand_variant_matches_text(variant, sanitized):
            return True
    return False


def get_brand_positions(text: str, brand: str) -> List[int]:
    """Return all start indexes of brand mentions (case-insensitive)."""
    if not text or not brand:
        return []
    positions: Set[int] = set()
    spans = _url_spans(text)
    for variant in _brand_detection_variants(brand):
        if not variant:
            continue
        pattern = re.compile(r"(?i)(?<![A-Za-z0-9])" + re.escape(variant) + r"(?![A-Za-z0-9])")
        for match in pattern.finditer(text):
            if _is_index_in_spans(match.start(), spans):
                continue
            positions.add(match.start())
    return sorted(positions)


def find_citations_after_brand(text: str, brand: str) -> List[str]:
    positions = get_brand_positions(text, brand)
    if not positions:
        return []
    citations: List[str] = []
    url_matches = list(URL_RE.finditer(text))
    for pos in positions:
        for match in url_matches:
            if match.start() >= pos:
                citation = match.group(1).rstrip(".,;:)]}>")
                if citation and citation not in citations:
                    citations.append(citation)
                break
    return citations


# --- Helper: count_brand_occurrences ---
def count_brand_occurrences(text: str, brand: str, brand_domain: str = "") -> int:
    """Count brand "occurrences" using mentions and brand-owned citations."""
    # Rule:
    #   - Each brand mention counts as 1 occurrence.
    #   - If a mention is immediately followed by a citation URL, that citation is considered part of
    #     the same occurrence (not an extra).
    #   - Any remaining URLs in the response that are not paired to a brand mention can add
    #     occurrences (rare, but handles edge cases where citations exist without an explicit mention).
    if not text or not brand:
        return 0

    mention_positions = sorted(get_brand_positions(text, brand))
    if not brand_domain:
        return len(mention_positions)

    domain_host = _host_from_url(brand_domain)
    if not domain_host:
        domain_host = _norm_host(brand_domain)
    if not domain_host:
        return len(mention_positions)

    url_matches = list(URL_RE.finditer(text))
    owned_matches = [
        match
        for match in url_matches
        if match and _is_subdomain_or_equal(_host_from_url(match.group(1)), domain_host)
    ]
    if not mention_positions and not owned_matches:
        return 0

    paired_starts: Set[int] = set()
    for idx, mention_pos in enumerate(mention_positions):
        next_pos = mention_positions[idx + 1] if idx + 1 < len(mention_positions) else None
        for match in owned_matches:
            match_start = match.start()
            if match_start < mention_pos:
                continue
            if next_pos is not None and match_start >= next_pos:
                break
            paired_starts.add(match_start)
            break

    unmatched_owned = sum(1 for match in owned_matches if match.start() not in paired_starts)
    return len(mention_positions) + unmatched_owned



def rank_position(text: str, brand: str, competitor_brands: List[str]) -> Optional[int]:
    """
    Rank the current brand within the supplied competitors based on how often
    each brand is mentioned or cited, with earliest mention breaking ties.
    """
    if not text or not brand:
        return None

    normalized_brand = brand.strip()
    if not normalized_brand:
        return None

    seen: Set[str] = {normalized_brand.lower()}
    competitors_clean: List[str] = []
    for competitor in competitor_brands:
        comp_clean = (competitor or "").strip()
        if not comp_clean:
            continue
        comp_key = comp_clean.lower()
        if comp_key in seen:
            continue
        seen.add(comp_key)
        competitors_clean.append(comp_clean)

    brands_to_score = [normalized_brand] + competitors_clean
    counts = {name: count_brand_occurrences(text, name) for name in brands_to_score}
    brand_count = counts.get(normalized_brand, 0)
    if brand_count == 0:
        return None

    first_mentions = find_brand_mentions(text, brands_to_score)

    def mention_position(name: str) -> int:
        return first_mentions.get(name, float("inf"))

    ranked = sorted(
        ((name, counts[name]) for name in brands_to_score if counts[name] > 0),
        key=lambda pair: (-pair[1], mention_position(pair[0])),
    )

    for idx, (name, _) in enumerate(ranked, start=1):
        if name == normalized_brand:
            return idx

    return None



# -----------------------------
# Universal scoring (Sentiment USS: -10..10, Authority UAS: 1..10)
# -----------------------------

# Authority citation quality weights (q_j). Keep 0..1.
CITATION_QUALITY_BY_CATEGORY = {
    "direct": 1.00,
    "documentation": 0.90,
    "publisher": 0.80,
    "marketplace": 0.60,
    "aggregator": 0.50,
    "blog": 0.50,
    "social": 0.40,
    "forum": 0.40,
    "affiliate": 0.30,
    "unknown": 0.20,
}

# UAS weights (must sum to 1)
UAS_WEIGHTS = {
    "C": 0.30,  # Citation strength
    "S": 0.20,  # Structural confidence
    "V": 0.20,  # Verifiability
    "D": 0.15,  # Domain alignment
    "L": 0.15,  # Linguistic certainty
}


def sentiment_uss_score(text: str, brand: str, w_p: float = 1.0, w_n: float = 1.0) -> int:
    """
    Universal Sentiment Score (USS)

        Sentiment_USS = sum_i( p_i*w_p - n_i*w_n ) / sum_i t_i

    where t_i = p_i + n_i (sentiment-bearing tokens) per sentence.

    Output is scaled to integer in [-10, 10].
    """
    if not text or not brand:
        return 0

    bl = brand.strip().lower()
    tl = text.lower()
    if bl not in tl:
        return 0

    sentences = [s for s in split_sentences(text) if bl in s.lower()]
    if not sentences:
        return 0

    num = 0.0
    denom = 0.0

    for s in sentences:
        toks = _tokenize(s)
        # Count occurrences of lexicon tokens (simple membership counts)
        p_i = sum(1 for t in toks if t in SENT_POS)
        n_i = sum(1 for t in toks if t in SENT_NEG)
        t_i = p_i + n_i
        if t_i == 0:
            continue
        num += (p_i * w_p) - (n_i * w_n)
        denom += float(t_i)

    if denom == 0.0:
        return 0

    polarity = num / denom
    # Normalize to [-1, 1] then scale to [-10, 10]
    polarity = max(-1.0, min(1.0, polarity))
    score = int(round(polarity * 10.0))
    return max(-10, min(10, score))


def authority_uas_score(response_text: str, urls: List[str], brand_domain: str = "") -> int:
    """
    Universal Authority Score (UAS)

        Authority_UAS = a*C + b*S + g*V + d*D + e*L

    with a+b+g+d+e = 1 and each component in [0, 1].

    Output is scaled to integer in [1, 10].
    """
    text = response_text or ""
    sents = split_sentences(text)
    total_sent = max(1, len(sents))

    # --- C: Citation strength ---
    uniq_urls = [u for u in dict.fromkeys(urls or []) if u]
    if uniq_urls:
        # Use source-type classifier to get category -> quality weight
        qs: List[float] = []
        for u in uniq_urls:
            cat = _classify_url_source_type(u, brand_domain=brand_domain)
            qs.append(float(CITATION_QUALITY_BY_CATEGORY.get(cat, 0.20)))
        C = sum(qs) / float(len(qs))
    else:
        C = 0.0

    # --- S: Structural confidence ---
    # Facts: sentences containing numbers/percents/dates; Definitions: is/are/means/refers to; Enumerated: bullet/numbered.
    def_re = re.compile(r"\b(is|are|means|refers to)\b", re.IGNORECASE)
    fact_re = re.compile(r"\b\d+(?:[\.,]\d+)?\b|%|\b(19\d\d|20\d\d)\b")
    enum_re = re.compile(r"^\s*(?:\d+[\)\.]|[-•])")

    facts = sum(1 for s in sents if fact_re.search(s))
    defs = sum(1 for s in sents if def_re.search(s))
    enums = sum(1 for s in sents if enum_re.search(s))
    S = (facts + defs + enums) / float(total_sent)
    S = max(0.0, min(1.0, S))

    # --- V: Verifiability ---
    # Claims: sentences with assertive verbs; Verifiable: has URL in sentence OR has numbers.
    claim_re = re.compile(r"\b(is|are|has|have|offers|provides|supports|uses|includes|enables|allows)\b", re.IGNORECASE)
    url_re = URL_RE

    claims = [s for s in sents if claim_re.search(s)]
    total_claims = len(claims)
    if total_claims == 0:
        V = 0.0
    else:
        verifiable = 0
        for s in claims:
            if url_re.search(s) or fact_re.search(s):
                verifiable += 1
        V = verifiable / float(total_claims)
        V = max(0.0, min(1.0, V))

    # --- D: Domain alignment ---
    # Cosine similarity between response topic tokens and source-domain tokens.
    # Topic tokens: response text; Domain tokens: URL hosts + paths.
    topic_tf = _tf_vector(_tokenize(text))

    domain_blob_parts: List[str] = []
    for u in uniq_urls:
        try:
            p = urlparse(u)
            host = _norm_host(p.netloc)
            path = (p.path or "").replace("/", " ")
            domain_blob_parts.append(f"{host} {path}")
        except Exception:
            continue
    domain_blob = " ".join(domain_blob_parts)
    domain_tf = _tf_vector(_tokenize(domain_blob))

    D = _cosine_sim(topic_tf, domain_tf)
    D = max(0.0, min(1.0, D))

    # --- L: Linguistic certainty ---
    # L = 1 - hedging_sentences / total_sentences
    hedged = 0
    for s in sents:
        stoks = set(_tokenize(s))
        if stoks.intersection(HEDGING_TERMS):
            hedged += 1
    L = 1.0 - (hedged / float(total_sent))
    L = max(0.0, min(1.0, L))

    # Weighted sum
    a = UAS_WEIGHTS["C"]
    b = UAS_WEIGHTS["S"]
    g = UAS_WEIGHTS["V"]
    d = UAS_WEIGHTS["D"]
    e = UAS_WEIGHTS["L"]

    uas = (a * C) + (b * S) + (g * V) + (d * D) + (e * L)
    uas = max(0.0, min(1.0, uas))

    # Map [0,1] -> [1,10]
    score = 1 + int(round(uas * 9.0))
    return max(1, min(10, score))


def extract_page_path(source_url: str) -> str:
    """Return the path component of a URL (e.g., '/pages/contact-us')."""
    if not source_url:
        return ""
    parsed = urlparse(source_url.strip())
    path = parsed.path or ""
    if not path:
        return ""
    return path if path.startswith("/") else f"/{path}"


def citation_page_sequence(urls: Sequence[str]) -> str:
    """Return comma-separated page paths corresponding to each citation URL."""
    if not urls:
        return ""
    paths = [extract_page_path(u) for u in urls]
    return ", ".join(paths)


# -----------------------------
# Brand data
# -----------------------------

@dataclass
class BrandRow:
    brand_id: str
    brand: str
    domain: str = ""
    source: str = ""


def load_brands_from_csv(path: str) -> Tuple[Dict[str, BrandRow], List[str]]:
    """
    Flexible loader for brand.csv.

    Supported header variants:
      - brand_id: brand_id | id | brandId
      - brand: brand | name | brand_name | brandName
      - domain (optional): domain | website | url

    If no brand_id column exists, a synthetic brand_id is created from the brand string.
    If no brand column exists, the first column is treated as brand.
    """
    df = pd.read_csv(path)

    cols = {c: c.lower().strip() for c in df.columns}

    def find_col(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            for c in df.columns:
                if cols[c] == cand.lower():
                    return c
        return None

    brand_id_col = find_col(["brand_id", "id", "brandid", "brandId"])
    brand_col = find_col(["brand", "name", "brand_name", "brandname", "brandName"])
    domain_col = find_col(["domain", "website", "url"])
    source_col = find_col(["source", "source_url"])

    # If brand column is missing, fall back to the first column
    if brand_col is None:
        if len(df.columns) == 0:
            raise ValueError(f"{path} has no columns")
        brand_col = df.columns[0]

    # If brand_id is missing, synthesize from brand
    if brand_id_col is None:
        brand_id_col = "__synthetic_brand_id__"
        df[brand_id_col] = (
            df[brand_col]
            .astype(str)
            .fillna("")
            .map(lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_"))
        )

    by_id: Dict[str, BrandRow] = {}
    all_brands: List[str] = []

    for _, r in df.iterrows():
        bid = str(r[brand_id_col]) if pd.notna(r.get(brand_id_col)) else ""
        b = str(r[brand_col]) if pd.notna(r.get(brand_col)) else ""
        d = ""
        if domain_col is not None:
            d = str(r[domain_col]) if pd.notna(r.get(domain_col)) else ""
        s = ""
        if source_col is not None:
            s = str(r[source_col]) if pd.notna(r.get(source_col)) else ""

        if not b:
            continue

        if not bid:
            bid = re.sub(r"[^a-z0-9]+", "_", b.lower()).strip("_")

        by_id[bid] = BrandRow(brand_id=bid, brand=b, domain=d, source=s)
        if b not in all_brands:
            all_brands.append(b)

    return by_id, all_brands


def load_brands_from_json(path: str) -> Tuple[Dict[str, BrandRow], List[str]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    entries: List[Dict[str, Union[str, int, float]]] = []
    if isinstance(payload, list):
        entries = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        for key in ("results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                entries = [item for item in value if isinstance(item, dict)]
                break
        if not entries:
            entries = [payload]

    by_id: Dict[str, BrandRow] = {}
    all_brands: List[str] = []
    for entry in entries:
        raw_name = entry.get("name") or entry.get("brand") or ""
        brand_name = str(raw_name).strip()
        if not brand_name:
            continue

        raw_id = entry.get("brand_id") or entry.get("id") or entry.get("slug") or ""
        brand_id = str(raw_id).strip()
        if not brand_id:
            brand_id = re.sub(r"[^a-z0-9]+", "_", brand_name.lower()).strip("_")

        domain = str(entry.get("domain") or entry.get("website") or entry.get("url") or "").strip()
        source = str(entry.get("source") or entry.get("source_url") or "").strip()

        by_id[brand_id] = BrandRow(
            brand_id=brand_id, brand=brand_name, domain=domain, source=source
        )
        if brand_name not in all_brands:
            all_brands.append(brand_name)

    return by_id, all_brands


def load_brands(path: str) -> Tuple[Dict[str, BrandRow], List[str]]:
    if path.lower().endswith(".json"):
        return load_brands_from_json(path)
    return load_brands_from_csv(path)


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _parse_competitor_brands(value: Optional[str]) -> List[str]:
    """Parse comma-separated competitor names from CLI input."""
    if not value:
        return []
    return [segment.strip() for segment in value.split(",") if segment.strip()]


def _matching_brand_keys(row_core_key: str, allowed_brand_keys: Optional[Set[str]]) -> List[str]:
    if not row_core_key:
        return []
    if not allowed_brand_keys:
        return [row_core_key]
    matches: List[str] = []
    seen: Set[str] = set()
    for key in allowed_brand_keys:
        if not key or key in seen:
            continue
        if (
            row_core_key == key
            or row_core_key.startswith(key)
            or key.startswith(row_core_key)
        ):
            seen.add(key)
            matches.append(key)
    return matches


def _load_competitor_data_from_csv(
    path: Path,
    allowed_brand_keys: Optional[Set[str]] = None,
    lob_pattern: Optional[Pattern[str]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Return competitor lists plus core-brand hosts from `competitors.csv`."""
    if not path.exists():
        return {}, {}
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return {}, {}

    columns = {str(col).strip().lower(): col for col in df.columns if col}
    brand_column = None
    for candidate in ("core_brand", "core brand", "brand", "brand_name", "brandname"):
        if candidate in columns:
            brand_column = columns[candidate]
            break
    competitor_column = None
    for candidate in ("competitor", "competitors"):
        if candidate in columns:
            competitor_column = columns[candidate]
            break
    domain_column = None
    for candidate in ("core_brand_domain", "core brand domain", "domain", "website", "url"):
        if candidate in columns:
            domain_column = columns[candidate]
            break
    lob_column = None
    for candidate in ("lob", "line_of_business", "line-of-business", "line of business", "lob_name"):
        if candidate in columns:
            lob_column = columns[candidate]
            break

    competitors_map: Dict[str, List[str]] = {}
    seen_per_key: Dict[str, Set[str]] = defaultdict(set)
    domain_map: Dict[str, str] = {}

    for _, row in df.iterrows():
        core_brand = str(row.get(brand_column) or "").strip() if brand_column else ""
        core_key = _normalize_brand_key(core_brand)
        match_keys = _matching_brand_keys(core_key, allowed_brand_keys)
        if not match_keys:
            continue

        if lob_column and lob_pattern:
            lob_value = str(row.get(lob_column) or "").strip()
            if not lob_pattern.search(lob_value):
                continue

        if domain_column:
            domain_value = str(row.get(domain_column) or "").strip()
            if domain_value:
                host = _normalize_domain_host(domain_value)
                if host:
                    for match_key in match_keys:
                        if match_key not in domain_map:
                            domain_map[match_key] = host

        if not competitor_column:
            continue
        competitor = str(row.get(competitor_column) or "").strip()
        if not competitor:
            continue
        normalized = competitor.lower()
        for match_key in match_keys:
            seen = seen_per_key.setdefault(match_key, set())
            if normalized in seen:
                continue
            seen.add(normalized)
            competitors_map.setdefault(match_key, []).append(competitor)

    return competitors_map, domain_map


def _primary_core_brand_name(brand_by_id: Mapping[str, BrandRow]) -> str:
    for row in brand_by_id.values():
        if row.brand:
            return row.brand
    return ""


def _competitor_list_for_brand(
    brand_name: str, competitor_map: Mapping[str, List[str]]
) -> List[str]:
    if not brand_name:
        return []
    key = _normalize_brand_key(brand_name)
    return competitor_map.get(key, [])


def build_stage_entries(row_idx: Union[int, str], row: pd.Series) -> List[dict]:
    brand = _safe_text(row.get("brand"))
    brand_id = _safe_text(row.get("brand_id"))
    prompt_id = _safe_text(row.get("prompt_id"))
    model_val = _safe_text(row.get("model"))
    brand_lc = brand.lower().strip()

    def _collect_stage(
        stage_name: str, prompt_text: str, response_text: str
    ) -> Optional[dict]:
        text = response_text.strip()
        if not text:
            return None
        urls = extract_urls(text)
        has_source = bool(urls)
        brand_present = detect_brand_presence(brand, text)
        brand_sourced = brand_present and has_source
        return {
            "orig_index": row_idx,
            "stage": stage_name,
            "brand_id": brand_id,
            "brand": brand,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "response": text,
            "model": model_val,
            "urls": urls,
            "has_source": has_source,
            "brand_present": brand_present,
            "brand_sourced": brand_sourced,
            "follow_up": detects_follow_up_question(text),
        }

    entries: list[dict] = []
    phase1_prompt = _safe_text(row.get("prompt"))
    phase1_response = _safe_text(row.get("response"))
    p1c_response = _safe_text(row.get("p1c_response"))
    if p1c_response.strip():
        phase1_response = p1c_response

    stage1_entry = _collect_stage("p1", phase1_prompt, phase1_response)
    if stage1_entry:
        entries.append(stage1_entry)

    p2_prompt = _safe_text(row.get("p2_prompt"))
    p2_response = _safe_text(row.get("p2_response"))
    stage2_entry = _collect_stage("p2", p2_prompt, p2_response)
    if stage2_entry:
        entries.append(stage2_entry)
    return entries


def build_stage_metrics(
    prefix: str,
    entry: Mapping[str, object],
    brand_by_id: Mapping[str, BrandRow],
    all_brand_names: List[str],
    competitor_brands: Sequence[str],
    competitor_domain_map: Mapping[str, str],
) -> Dict[str, object]:
    brand = entry["brand"] or ""
    brand_id = entry["brand_id"] or ""
    response = entry["response"] or ""
    detection_brands = _merge_brand_candidates([brand] + all_brand_names, competitor_brands)
    brand_sequence_list = brand_order_from_response(response, detection_brands)
    brand_sequence = ", ".join(brand_sequence_list)
    brand_row = brand_by_id.get(brand_id) or BrandRow(brand_id=brand_id, brand=brand)
    brand_key = _normalize_brand_key(brand)
    domain_override = competitor_domain_map.get(brand_key, "")
    source_domain_input = domain_override or (brand_row.domain or "")
    effective_domain = _normalize_domain_host(source_domain_input)
    if not effective_domain and brand_row.source:
        effective_domain = _normalize_domain_host(brand_row.source)
    brand_domains: Dict[str, str] = {brand: effective_domain} if effective_domain else {}
    freq_sequence = frequency_for_brand_sequence(response, brand_sequence_list, brand_domains=brand_domains)
    citations = find_citations_after_brand(response, brand)
    brand_present = bool(entry.get("brand_present"))
    authority_sequence = authority_scores_for_brand_sequence(
        response, brand_sequence_list, brand_domains=brand_domains
    )
    sentiment_sequence = sentiment_scores_for_brand_sequence(response, brand_sequence_list) if brand_sequence_list else ""
    sources = ", ".join(entry["urls"])

    # Bucket sources by type/category (comma-separated URLs per category)
    source_buckets = bucket_urls_by_source_type(entry["urls"], brand_domain=effective_domain)
    source_category_fields: Dict[str, str] = {}
    for cat in SOURCE_CATEGORIES:
        source_category_fields[f"{prefix}_sources_{cat}"] = ", ".join(source_buckets.get(cat, []))
    mention = 1 if brand_present else 0
    page = citation_page_sequence(citations) if brand_present else ""
    snippet = snippet_for_brand_mention(response, brand) if brand_present else ""

    brand_citation = _domain_hosts_for_urls(citations, effective_domain) if brand_present else ""

    cite_count = 0
    brand_source = brand_row.source.strip()
    if brand_source:
        normalized_source_url = brand_source.rstrip('/').lower()
        source_host = _host_from_url(brand_source)
        for citation in citations:
            citation_host = _host_from_url(citation)
            if source_host and citation_host == source_host:
                cite_count += 1
            elif citation.rstrip('/').lower() == normalized_source_url:
                cite_count += 1

    return {
        f"{prefix}_prompt": entry["prompt"],
        f"{prefix}_response": response,
        f"{prefix}_mention": mention,
        f"{prefix}_snippet": snippet,
        f"{prefix}_page": page,
        f"{prefix}_position": brand_sequence,
        f"{prefix}_word_count": word_counts_for_brand_sequence(
            response, brand_sequence_list, brand_domains=brand_domains
        ),
        f"{prefix}_authority": authority_sequence,
        f"{prefix}_frequency": freq_sequence,
        f"{prefix}_sentiment": sentiment_sequence,
        f"{prefix}_citation": ", ".join(citations),
        f"{prefix}_brand_citation": brand_citation,
        f"{prefix}_cite_count": cite_count,
        f"{prefix}_sources": sources,
        **source_category_fields,
    }


# -----------------------------
# Main analysis
# -----------------------------

def _run_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--responses",
        default=DEFAULT_RESPONSES_PATH,
        help="Input file for phase responses (defaults to phase2_responses_enriched.csv).",
    )
    ap.add_argument(
        "--brands",
        default="pmgclient.json",
        help="Input brand metadata (pmgclient.json with id/name mappings).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path (defaults to analysis_<mmddyyyy>.csv).",
    )
    ap.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key used for relevance scoring (defaults to OPENAI_API_KEY).",
    )
    ap.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model name for relevance scoring (defaults to AVIS_EMBEDDING_MODEL or text-embedding-3-small).",
    )
    ap.add_argument(
        "--position-competitors",
        default=None,
        help="Comma-separated competitor brands to include when building position/word-count metrics (defaults to the core-brand list from competitors.csv).",
    )
    ap.add_argument(
        "--trace",
        type=int,
        default=0,
        metavar="N",
        help="Print detailed ranking/scoring trace for the first N prompts.",
    )
    ap.add_argument(
        "--lob",
        default=None,
        help="Regex filter applied to the competitors.csv LOB column (case-insensitive).",
    )
    args = ap.parse_args()
    competitors_arg_value = args.position_competitors
    lob_arg_value = args.lob
    today = datetime.utcnow().strftime("%m%d%Y")
    default_out = Path(f"analysis_{today}.csv")
    args.out_path = Path(args.out) if args.out else default_out

    responses_path = Path(args.responses)
    brands_path = Path(args.brands)
    inputs = [
        ("responses", responses_path),
        ("brands", brands_path),
    ]
    for label, path in inputs:
        status = "found" if path.exists() else "missing"
        print(f"Need {label}: {path} ({status})")

    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        source = "--openai-api-key" if args.openai_api_key else "OPENAI_API_KEY"
        print(f"OpenAI API key: set via {source}.")
    else:
        print("OpenAI API key: missing (relevance scoring will be skipped).")
    openai_package_version = getattr(openai, "__version__", "unknown")
    print(f"OpenAI package version: {openai_package_version}")
    embedding_model = args.embedding_model or DEFAULT_EMBEDDING_MODEL
    print(f"Embedding model for relevance: {embedding_model}")

    if not brands_path.exists():
        raise SystemExit(f"{brands_path} not found.")
    brand_by_id, all_brand_names = load_brands(str(brands_path))

    normalized_brand_keys: Set[str] = {
        _normalize_brand_key(name) for name in all_brand_names if name
    }
    allowed_brand_keys: Optional[Set[str]] = normalized_brand_keys or None

    competitor_csv_path = Path.cwd() / "competitors.csv"
    lob_pattern: Optional[Pattern[str]] = None
    if lob_arg_value:
        try:
            lob_pattern = re.compile(lob_arg_value, re.IGNORECASE)
            print(f"Filtering competitors.csv rows to LOB regex {lob_arg_value!r}.")
        except re.error as exc:
            print(f"Invalid --lob regex {lob_arg_value!r}: {exc}; ignoring LOB filter.")

    competitor_map, competitor_domain_map = _load_competitor_data_from_csv(
        competitor_csv_path,
        allowed_brand_keys=allowed_brand_keys,
        lob_pattern=lob_pattern,
    )

    if competitors_arg_value:
        competitor_brands = _parse_competitor_brands(competitors_arg_value)
    else:
        core_brand_name = _primary_core_brand_name(brand_by_id)
        competitor_brands = _competitor_list_for_brand(core_brand_name, competitor_map)
        if competitor_brands:
            print(
                f"Loaded {len(competitor_brands)} competitors "
                f"for {core_brand_name or 'the core brand'} from {competitor_csv_path}."
            )
        elif competitor_map:
            brand_label = core_brand_name or "the core brand"
            print(
                f"No competitor entries matched {brand_label} in {competitor_csv_path}; "
                "skipping competitor ranking."
            )
        else:
            print(f"No competitors found in {competitor_csv_path}; position metrics will ignore extra brands.")

    if not responses_path.exists():
        raise SystemExit(f"{responses_path} not found.")
    responses_df = pd.read_csv(responses_path)
    if "type" in responses_df.columns:
        responses_df = responses_df.drop(columns=["type"])
    print(f"Loaded {len(responses_df)} rows from {responses_path} with {len(responses_df.columns)} columns.")

    date_column = "date"
    today_value = datetime.utcnow().strftime("%Y-%m-%d")
    if date_column not in responses_df.columns:
        responses_df[date_column] = today_value
        print(f"Added missing {date_column} column set to {today_value}.")
    else:
        is_blank = responses_df[date_column].isna() | responses_df[date_column].astype(str).str.strip().eq("")
        blank_count = int(is_blank.sum())
        if blank_count:
            responses_df.loc[is_blank, date_column] = today_value
            print(f"Filled {blank_count} blank {date_column} rows with {today_value}.")

    required = {
        "brand_id",
        "brand",
        "prompt_id",
        "prompt",
        "response",
        "p2_prompt",
        "p2_response",
    }
    missing = sorted(required - set(responses_df.columns))
    if missing:
        raise SystemExit(
            f"{args.responses} is missing required columns and will not be auto-renamed: {missing}"
        )

    embedding_client = _build_embedding_client(openai_api_key)

    print("=== DEBUG: embedding client ===")
    print("Embedding client type:", type(embedding_client))
    print("Has openai.OpenAI:", hasattr(openai, "OpenAI"))
    print("Embedding model resolved to:", embedding_model)
    print("================================")
    embedding_cache: Dict[str, Optional[List[float]]] = {}
    global _MENTION_DETECTOR
    if openai_api_key:
        try:
            _MENTION_DETECTOR = BrandMentionDetector(embedding_client, openai_api_key, embedding_model)
        except Exception as exc:
            print("Failed to initialize mention detector:", repr(exc))
            _MENTION_DETECTOR = None

    def _row_relevance(row: pd.Series) -> Optional[float]:
        prompt_text = _safe_text(row.get("prompt"))
        summary_text = _safe_text(row.get("summary"))  # optional; relevance will be blank if missing
        return _compute_prompt_summary_relevance(
            prompt_text,
            summary_text,
            embedding_cache,
            embedding_client,
            openai_api_key,
            embedding_model,
        )

    responses_df["relevance"] = responses_df.apply(_row_relevance, axis=1)
    relevance_nonempty = responses_df["relevance"].notna().sum()
    if openai_api_key:
        print(f"Relevance column: computed {relevance_nonempty} non-empty rows.")
        print("DEBUG: Relevance value counts:")
        print(responses_df["relevance"].value_counts(dropna=False).head(10))
    else:
        print("Relevance column: skipped (all values blank without OpenAI API key).")

    # Back-compat: if response_id is missing, synthesize one
    if "response_id" not in responses_df.columns:
        if "prompt_id" in responses_df.columns:
            responses_df["response_id"] = (
                responses_df["brand_id"].astype(str).fillna("")
                + "_"
                + responses_df["prompt_id"].astype(str).fillna("")
            )
        else:
            responses_df["response_id"] = [str(uuid.uuid4()) for _ in range(len(responses_df))]

    stage_entries: List[dict] = []
    for idx, row in responses_df.iterrows():
        stage_entries.extend(build_stage_entries(idx, row))

    if not stage_entries:
        raise SystemExit("No stage responses found in input.")

    entries_by_index: Dict[Union[int, str], Dict[str, dict]] = defaultdict(dict)
    for entry in stage_entries:
        idx = entry["orig_index"]
        entries_by_index[idx][entry["stage"]] = entry

    stage_prefixes = [("p1", "p1"), ("p2", "p2")]
    stage_field_defaults = {
        "prompt": "",
        "response": "",
        "mention": 0,
        "snippet": "",
        "page": "",
        "position": "",
        "word_count": "",
        "authority": "",
        "frequency": "",
        "sentiment": "",
        "citation": "",
        "brand_citation": "",
        "sources": "",
    }
    # Add one column per source category (per stage), with comma-separated URLs
    for cat in SOURCE_CATEGORIES:
        stage_field_defaults[f"sources_{cat}"] = ""
    stage_template: Dict[str, object] = {}
    stage_columns: List[str] = []
    for _, prefix in stage_prefixes:
        for field, default in stage_field_defaults.items():
            column_name = f"{prefix}_{field}"
            stage_template[column_name] = default
            stage_columns.append(column_name)

    p1_columns = [c for c in stage_columns if c.startswith("p1_")]
    p2_columns = [c for c in stage_columns if c.startswith("p2_")]
    stage_column_set = set(stage_columns)
    extra_input_columns = [
        col
        for col in responses_df.columns
        if col not in BASE_OUTPUT_FIELDS + P1C_FIELDS
        and col not in stage_column_set
        and col not in {"prompt", "response"}
    ]


    trace_count = args.trace
    traced = 0

    output_rows: List[Dict[str, object]] = []
    for idx, row in responses_df.iterrows():
        response_id = _safe_text(row.get("response_id"))
        if not response_id:
            fallback_id = f"{_safe_text(row.get('brand_id'))}_{_safe_text(row.get('prompt_id'))}"
            response_id = fallback_id or str(uuid.uuid4())

        row_data: Dict[str, object] = {field: _safe_text(row.get(field)) for field in BASE_OUTPUT_FIELDS}
        stage_data = stage_template.copy()
        entry_map = entries_by_index.get(idx, {})
        for stage, prefix in stage_prefixes:
            entry = entry_map.get(stage)
            if not entry:
                continue
            stage_data.update(
                build_stage_metrics(
                    prefix,
                    entry,
                    brand_by_id,
                    all_brand_names,
                    competitor_brands,
                    competitor_domain_map,
                )
            )

            # --trace: print ranking breakdown for first N rows
            if traced < trace_count and stage == "p1":
                brand = entry["brand"] or ""
                response = entry["response"] or ""
                detection_brands = _merge_brand_candidates(
                    [brand] + all_brand_names, competitor_brands
                )
                brand_seq = brand_order_from_response(response, detection_brands)
                mentions = find_brand_mentions(response, detection_brands)
                brand_row = brand_by_id.get(entry["brand_id"] or "") or BrandRow(
                    brand_id=entry["brand_id"] or "", brand=brand
                )
                trace_brand_key = _normalize_brand_key(brand)
                trace_override = competitor_domain_map.get(trace_brand_key, "")
                trace_domain_input = trace_override or (brand_row.domain or "")
                trace_domain_host = _normalize_domain_host(trace_domain_input)
                if not trace_domain_host and brand_row.source:
                    trace_domain_host = _normalize_domain_host(brand_row.source)
                brand_domains_trace: Dict[str, str] = {brand: trace_domain_host} if trace_domain_host else {}

                print(f"\n{'='*60}")
                print(f"TRACE [{traced+1}/{trace_count}] row={idx}  brand=\"{brand}\"")
                print(f"  prompt: {_safe_text(row.get('prompt'))[:120]}...")
                print(f"  model: {_safe_text(row.get('model'))}")
                print(f"  brand domain: {trace_domain_host or '(none)'}")
                print(f"  response length: {len(response)} chars")
                print(f"\n  --- Brand detection ---")
                print(f"  candidates checked: {len(detection_brands)}")
                print(f"  brands found (appearance order): {brand_seq}")
                print(f"  first-mention indexes: ", end="")
                for b in brand_seq:
                    print(f"{b}@{mentions.get(b, '?')}  ", end="")
                print()
                print(f"\n  --- Frequency (occurrence counts) ---")
                for b in brand_seq:
                    count = count_brand_occurrences(response, b, brand_domain=brand_domains_trace.get(b, ""))
                    print(f"    {b}: {count}")
                print(f"\n  --- Sentiment (lexicon USS, -10..10) ---")
                for b in brand_seq:
                    score = sentiment_uss_score(response, b)
                    print(f"    {b}: {score}")
                print(f"\n  --- Output values ---")
                print(f"  p1_position: {stage_data.get('p1_position', '')}")
                print(f"  p1_frequency: {stage_data.get('p1_frequency', '')}")
                print(f"  p1_sentiment: {stage_data.get('p1_sentiment', '')}")
                print(f"  p1_mention: {stage_data.get('p1_mention', '')}")
                print(f"  p1_authority: {stage_data.get('p1_authority', '')}")
                print(f"  p1_citation: {stage_data.get('p1_citation', '')}")
                print(f"{'='*60}\n")
                traced += 1

        row_data.update(stage_data)
        row_data.update({field: _safe_text(row.get(field)) for field in P1C_FIELDS})
        input_data = {col: _safe_text(row.get(col)) for col in extra_input_columns}
        row_data.update(input_data)
        output_rows.append(row_data)

    output_fieldnames = BASE_OUTPUT_FIELDS + p1_columns + P1C_FIELDS + p2_columns + extra_input_columns

    out_df = pd.DataFrame(output_rows, columns=output_fieldnames)
    out_path = args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    alternate_output = out_path.parent / "analysis.csv"
    out_df.to_csv(alternate_output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {out_path} ({len(out_df)} rows)")
    print(f"Wrote alternate copy {alternate_output}")


def main() -> None:
    start = time.perf_counter()
    try:
        _run_main()
    finally:
        _USAGE_LOGGER.flush(time.perf_counter() - start)


if __name__ == "__main__":
    main()

