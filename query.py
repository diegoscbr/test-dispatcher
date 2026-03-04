#!/usr/bin/env python3
"""
Run prompts from prompts.csv through a single OpenRouter model and write responses.csv.

Input:  prompts.csv and optional context_prompts.csv
Output: responses.csv

Required output columns:
persona_id, brand_id, brand, prompt_id, prompt, category, type,
response, model
"""

import os
import time
import json
import argparse
import re
import concurrent.futures
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urlparse

import pandas as pd
import openai
import requests
import unicodedata

# --- Double-encoding mojibake repair helpers ---
def _looks_better(candidate: str, original: str) -> bool:
    """Heuristic: prefer candidate when it reduces suspicious mojibake markers."""
    if not isinstance(candidate, str) or not isinstance(original, str):
        return False

    bad_markers = ("Ã", "â", "Â", "�", "‚", "Ä")

    def score(s: str) -> int:
        return sum(s.count(b) for b in bad_markers)

    try:
        return score(candidate) < score(original)
    except Exception:
        return False


def _repair_double_encoded(text: str) -> str:
    """Best-effort repair for common latin1 <-> utf-8 double-encoding mojibake.

    Attempts two common transforms and accepts a transform only if it appears
    to reduce mojibake markers (using _looks_better).
    """
    if not text or not isinstance(text, str):
        return text

    # Quick heuristic: if text contains common mojibake bytes, attempt repair.
    suspects = ("Ã", "â", "Â", "�", "‚", "Ä")
    if not any(ch in text for ch in suspects):
        return text

    # Try latin1 -> utf-8 repair
    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        if _looks_better(repaired, text):
            return repaired
    except Exception:
        pass

    # Try utf-8 -> latin1 repair (less common)
    try:
        repaired = text.encode("utf-8", errors="ignore").decode("latin-1", errors="ignore")
        if _looks_better(repaired, text):
            return repaired
    except Exception:
        pass

    return text


def normalize_text_human_readable(text: str) -> str:
    """Normalize text to be human-readable and CSV-safe.

    - Repair common double-encoding mojibake
    - Normalize Unicode (NFKC)
    - Replace smart punctuation with ASCII equivalents
    - Remove zero-width / non-printing characters
    - Collapse weird whitespace
    """
    # First, best-effort repair of double-encoded mojibake (latin1<->utf-8 issues)
    text = _repair_double_encoded(text)
    if not text or not isinstance(text, str):
        return text

    # Unicode normalization
    t = unicodedata.normalize("NFKC", text)

    # Replace common smart punctuation
    replacements = {
        "“": '"', "”": '"', "„": '"',
        "’": "'", "‘": "'",
        "–": "-", "—": "-", "−": "-",
        "…": "...",
        "•": "-",
        " ": " ",  # non-breaking space
    }
    for k, v in replacements.items():
        t = t.replace(k, v)

    # Remove zero-width and control chars (except newlines/tabs)
    cleaned = []
    for ch in t:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\t"):
            continue
        cleaned.append(ch)
    t = "".join(cleaned)

    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()

# Cache for resolving Gemini grounding redirect URLs (vertexaisearch grounding-api-redirect).
GROUNDING_URL_CACHE: Dict[str, str] = {}

_VERTEX_REDIRECT_HOST = "vertexaisearch.cloud.google.com"

def resolve_grounding_redirect(url: str, timeout: int = 10) -> str:
    """Resolve a Gemini grounding redirect URL to its final destination.

    Gemini grounding often returns URLs like:
      https://vertexaisearch.cloud.google.com/grounding-api-redirect/...

    We attempt to follow redirects and return the final URL.
    If resolution fails (network restrictions, timeouts), we fall back to the original URL.

    NOTE: This is best-effort and safe for restricted environments.
    """
    u = (url or "").strip()
    if not u:
        return ""
    if u in GROUNDING_URL_CACHE:
        return GROUNDING_URL_CACHE[u]

    try:
        # Prefer HEAD to save bandwidth; some hosts may block HEAD.
        r = requests.head(
            u,
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        final_url = (r.url or u).strip()
        GROUNDING_URL_CACHE[u] = final_url
        return final_url
    except Exception:
        try:
            r = requests.get(
                u,
                allow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            final_url = (r.url or u).strip()
            GROUNDING_URL_CACHE[u] = final_url
            return final_url
        except Exception:
            GROUNDING_URL_CACHE[u] = u
            return u


def _resolve_vertex_redirect(url: str) -> str:
    """Return the final URL for Vertex grounding redirects, or the original if not a wrapper."""
    if not url or not isinstance(url, str):
        return url
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        host = ""
    if host != _VERTEX_REDIRECT_HOST:
        return url
    return resolve_grounding_redirect(url)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _collect_nested_dicts(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            out.append(value)
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    visit(obj)
    return out


def extract_openai_network_data(resp: Any) -> Tuple[List[str], List[str]]:
    if not isinstance(resp, dict):
        return [], []
    queries: List[str] = []
    urls: List[str] = []
    for item in _collect_nested_dicts(resp):
        action = item.get("action") or {}
        if not action:
            action = item.get("arguments") or {}
        for key in ("queries", "query", "searchQueries", "webSearchQueries"):
            value = action.get(key)
            if isinstance(value, str):
                queries.append(value)
            elif isinstance(value, list):
                for q in value:
                    if isinstance(q, str):
                        queries.append(q)
        for key in ("sources", "searchSources"):
            for src in (action.get(key) or []):
                if isinstance(src, dict):
                    u = src.get("url") or src.get("uri")
                    if isinstance(u, str):
                        urls.append(u)
    return _dedupe_preserve_order(queries), _dedupe_preserve_order(urls)


# --- OpenAI source extraction helpers ---
from typing import Tuple as _Tuple

def extract_openai_sources(resp: Any) -> List[_Tuple[str, str]]:
    """Extract (title, url) pairs from OpenAI Responses web_search_call sources.

    Best-effort across both raw dict payloads and nested variants.
    """
    if not isinstance(resp, dict):
        return []

    pairs: List[_Tuple[str, str]] = []
    seen: set[str] = set()

    for item in _collect_nested_dicts(resp):
        action = item.get("action") or {}
        if not action:
            action = item.get("arguments") or {}

        for key in ("sources", "searchSources"):
            for src in (action.get(key) or []):
                if not isinstance(src, dict):
                    continue
                url = src.get("url") or src.get("uri")
                if not isinstance(url, str) or not url.strip():
                    continue
                url = url.strip()
                if url in seen:
                    continue
                seen.add(url)
                title = src.get("title") or src.get("name") or src.get("positionTitle") or ""
                title = str(title).strip() if title is not None else ""
                pairs.append((title, url))

    return pairs


def append_sources_section(text: str, sources: List[_Tuple[str, str]], heading: str = "Sources") -> str:
    """Append a Sources section to `text`.

    Keeps links human-visible in the output, similar to Gemini chat UX.
    """
    base = (text or "").strip()
    if not sources:
        return base

    lines: List[str] = [base, "", f"{heading}:"] if base else [f"{heading}:"]
    for i, (title, url) in enumerate(sources, start=1):
        label = title.strip() if title else url
        lines.append(f"[{i}] {label} - {url}")
    return "\n".join(lines).strip()


def extract_gemini_network_data(body: Any) -> Tuple[List[str], List[str]]:
    if not isinstance(body, dict):
        return [], []

    candidates = body.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return [], []

    queries: List[str] = []
    urls: List[str] = []

    # Scan all candidates; Gemini can place grounding metadata in different locations.
    for cand in candidates:
        if not isinstance(cand, dict):
            continue

        gm = cand.get("groundingMetadata") or cand.get("grounding_metadata")
        if not isinstance(gm, dict):
            content = cand.get("content")
            if isinstance(content, dict):
                gm = content.get("groundingMetadata") or content.get("grounding_metadata")
        if not isinstance(gm, dict):
            continue

        # Queries (support common key variants)
        for key in ("webSearchQueries", "web_search_queries", "searchQueries", "search_queries"):
            qv = gm.get(key)
            if isinstance(qv, str):
                queries.append(qv)
            elif isinstance(qv, list):
                for q in qv:
                    if isinstance(q, str):
                        queries.append(q)

        # Chunks / sources
        chunks = gm.get("groundingChunks") or gm.get("grounding_chunks") or []
        if not isinstance(chunks, list):
            chunks = []
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            web = ch.get("web") or {}
            if isinstance(web, dict):
                uri = web.get("uri")
                if isinstance(uri, str):
                    resolved = _resolve_vertex_redirect(uri)
                    urls.append(resolved or uri)

    return _dedupe_preserve_order(queries), _dedupe_preserve_order(urls)

URL_RE = re.compile(r"(https?://[^\s\)\]\}<>\"']+)", re.IGNORECASE)

GEMINI_TOOLS_PRIMARY = [{"google_search": {}}]
GEMINI_TOOLS_FALLBACK = [{"google_search_retrieval": {}}]
OPENAI_WEB_SEARCH_TOOL = {"type": "web_search"}
SEARCH_CACHE: Dict[str, str] = {}
SERPAPI_SEARCH_URL = "https://serpapi.com/search"
GOOGLE_CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(1).rstrip(".,;:)]}>") for m in URL_RE.finditer(text)]


def _iter_contents(item: Any) -> List[Any]:
    if not item:
        return []
    if isinstance(item, dict):
        return item.get("content") or []
    contents = getattr(item, "content", None)
    if contents is None:
        return []
    return contents


def extract_text_from_output(item: Any) -> str:
    if not item:
        return ""
    texts: List[str] = []
    for content in _iter_contents(item):
        text_value = None
        if isinstance(content, dict):
            text_value = content.get("text") or content.get("value")
        elif isinstance(content, str):
            text_value = content
        else:
            text_value = getattr(content, "text", None)
        if text_value:
            texts.append(str(text_value))
    if not texts:
        fallback = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        if fallback:
            texts.append(str(fallback))
    return "\n".join([t for t in texts if t]).strip()



def _response_used_web_search(response: Any) -> bool:
    """Return True if the OpenAI Responses API output contains a web_search_call item.

    NOTE: In openai-python, items in `response.output` may be dicts or typed objects.
    """
    outputs = list(getattr(response, "output", []) or [])
    for item in outputs:
        if isinstance(item, dict):
            if item.get("type") == "web_search_call":
                return True
        else:
            if getattr(item, "type", None) == "web_search_call":
                return True
    return False


# Gemini: detect if the response used Google Search grounding.
def _gemini_response_used_search(body: Optional[Dict[str, Any]]) -> bool:
    """Best-effort detection of whether Gemini returned Google Search grounding.

    With the Generative Language API, grounded responses typically include
    `groundingMetadata` on the candidate (sometimes nested) with chunks/support.
    We treat presence of grounding metadata/chunks as evidence that search was used.
    """
    if not body or not isinstance(body, dict):
        return False

    candidates = body.get("candidates") or []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue

        # Common locations
        gm = cand.get("groundingMetadata") or cand.get("grounding_metadata")
        if isinstance(gm, dict):
            # Typical evidence fields (names may vary)
            for k in (
                "groundingChunks",
                "grounding_chunks",
                "web",
                "search",
                "citations",
                "supports",
                "groundingSupports",
                "grounding_supports",
            ):
                v = gm.get(k)
                if v:
                    return True
            # If metadata exists at all, count as used (conservative)
            return True

        # Some variants nest grounding under the candidate's content
        content = cand.get("content")
        if isinstance(content, dict):
            gm2 = content.get("groundingMetadata") or content.get("grounding_metadata")
            if isinstance(gm2, dict) and gm2:
                return True

    return False


# Inline Gemini grounding citations, Gemini-chat style.
def _gemini_render_citations_in_text(text: str, body: Optional[Dict[str, Any]]) -> str:
    """Render Gemini grounding citations directly into the response text.

    Gemini API returns grounding in `candidates[0].groundingMetadata`:
      - groundingChunks: list of sources (web.title, web.uri)
      - groundingSupports: list of segments with startIndex/endIndex and groundingChunkIndices

    We mimic Gemini chat UX by:
      - inserting bracketed numeric citations like [1] into the text (when supports provide indices)
      - appending a Sources section listing numbered sources

    Best-effort and degrades gracefully when indices are missing.

    IMPORTANT: We always include *all* groundingChunks in the Sources list (even if supports are sparse),
    so you don't end up with only a single vertexaisearch redirect link.
    """
    base_text = (text or "").strip()
    if not base_text or not body or not isinstance(body, dict):
        return base_text

    candidates = body.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return base_text

    gm = candidates[0].get("groundingMetadata") or candidates[0].get("grounding_metadata")
    if not isinstance(gm, dict):
        return base_text

    chunks = gm.get("groundingChunks") or gm.get("grounding_chunks") or []
    supports = gm.get("groundingSupports") or gm.get("grounding_supports") or []

    if not isinstance(chunks, list) or not chunks:
        return base_text

    def _get_chunk_web(idx: int) -> Tuple[str, str]:
        try:
            ch = chunks[idx]
            if not isinstance(ch, dict):
                return "", ""
            web = ch.get("web")
            if not isinstance(web, dict):
                return "", ""
            title = str(web.get("title") or "").strip()
            uri = str(web.get("uri") or "").strip()
            return title, uri
        except Exception:
            return "", ""

    # Number ALL chunks in order for the Sources list.
    numbered_sources: List[Tuple[int, str, str]] = []
    for idx in range(len(chunks)):
        title, uri = _get_chunk_web(idx)
        if not uri:
            continue
        numbered_sources.append((len(numbered_sources) + 1, title, uri))

    # Map chunk index -> citation number (based on Sources list ordering).
    chunk_num_by_index: Dict[int, int] = {}
    src_idx = 0
    for idx in range(len(chunks)):
        title, uri = _get_chunk_web(idx)
        if not uri:
            continue
        src_idx += 1
        chunk_num_by_index[idx] = src_idx

    # Build insertion operations from groundingSupports when available.
    inserts: List[Tuple[int, str]] = []  # (end_index, marker)
    if isinstance(supports, list):
        for sup in supports:
            if not isinstance(sup, dict):
                continue
            seg = sup.get("segment")
            if not isinstance(seg, dict):
                continue
            end_idx = seg.get("endIndex")
            if not isinstance(end_idx, int):
                continue
            gci = sup.get("groundingChunkIndices") or []
            if not isinstance(gci, list) or not gci:
                continue

            try:
                chunk_idx = int(gci[0])
            except Exception:
                continue

            num = chunk_num_by_index.get(chunk_idx)
            if not num:
                continue

            clamped_end = max(0, min(end_idx, len(base_text)))
            inserts.append((clamped_end, f"[{num}]"))

    rendered = base_text

    # Apply inserts from end -> start so indices remain valid.
    if inserts:
        inserts.sort(key=lambda x: x[0], reverse=True)
        for pos, marker in inserts:
            if pos <= 0 or pos > len(rendered):
                continue

            # Avoid duplicate insertion near the same region.
            window_start = max(0, pos - 12)
            window_end = min(len(rendered), pos + 12)
            if marker in rendered[window_start:window_end]:
                continue

            insert_pos = pos

            # If Gemini's endIndex points into the *indentation* of a new line
            # (e.g., right before a markdown bullet), move insertion to the end
            # of the previous line (just before that newline).
            # Example broken output:
            #   "...sentence.\n [18]   * bullet"
            # We want:
            #   "...sentence. [18]\n   * bullet"
            nl = rendered.rfind("\n", 0, insert_pos)
            if nl != -1:
                between = rendered[nl:insert_pos]
                if between.strip() == "":
                    # We're in leading whitespace after a newline.
                    insert_pos = nl

            # Also handle the common case where endIndex lands right after a newline.
            if insert_pos > 0 and rendered[insert_pos - 1] == "\n":
                j = insert_pos - 1
                while j > 0 and rendered[j - 1] in ("\n", " ", "\t", "\r"):
                    j -= 1
                insert_pos = j

            # Ensure the marker is separated from text when needed.
            prefix = ""
            if insert_pos > 0 and not rendered[insert_pos - 1].isspace():
                prefix = " "

            rendered = rendered[:insert_pos] + prefix + marker + rendered[insert_pos:]

    # Inline citation rendering: replace [n] markers with resolved URLs in parentheses.
    # Build mapping: citation number -> resolved URL
    resolved_uri_by_num = {}
    for num, title, uri in numbered_sources:
        resolved_uri = resolve_grounding_redirect(uri, timeout=10) if uri else uri
        resolved_uri_by_num[num] = resolved_uri or uri

    def _linkify_marker(m: re.Match) -> str:
        try:
            n = int(m.group(1))
        except Exception:
            return m.group(0)
        url = (resolved_uri_by_num.get(n) or "").strip()
        if not url:
            return m.group(0)
        # Replace [n] entirely with the URL in parentheses
        return f"({url})"

    rendered = re.sub(r"\[(\d+)\]", _linkify_marker, rendered)

    # Append Sources section (Gemini-chat-like)
    lines: List[str] = []
    lines.append(rendered.rstrip())
    lines.append("")
    lines.append("Sources:")
    # Resolve grounding redirect URLs to final destinations when possible.
    for num, title, uri in numbered_sources:
        resolved_uri = resolve_grounding_redirect(uri, timeout=10) if uri else uri
        label = title or (resolved_uri or uri)
        lines.append(f"[{num}] {label} - {resolved_uri or uri}")

    return "\n".join(lines).strip()

#
# Default models: use the latest ChatGPT snapshot alias for OpenAI, plus Gemini.
DEFAULT_MODELS = ("gpt-5.2-chat-latest", "gemini-2.0-flash")
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
# Prefer the standard env var name; also support legacy GEMINIAPI_KEY.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_API_KEY_ENV_FALLBACK = "GEMINIAPI_KEY"

# If a requested Gemini model alias is unavailable, fall back to this known-good model.
GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"


DEFAULT_CONTEXT_PROMPTS = "context_prompts.csv"

ESSENTIAL_PROMPT_COLUMNS = [
    "prompt_id",
    "prompt",
]

# We only enforce the columns this script actually reads; any additional schema
# columns (persona_id, brand, category, keywords, etc.) flow through untouched.


def get_env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    return val if val and val.strip() else fallback


def load_env_file(path: Optional[str] = None) -> None:
    """Load key=value pairs from a .env file into os.environ."""
    env_path = Path(path or Path(__file__).resolve().parent / ".env")
    if not env_path.exists():
        return
    try:
        with env_path.open() as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                if key in os.environ and os.environ[key].strip():
                    continue
                os.environ[key] = value
    except Exception:
        pass


load_env_file()


def _summarize_search_items(items: List[Dict[str, Any]], limit: int) -> List[str]:
    snippets: List[str] = []
    seen_links: set[str] = set()
    for item in items:
        if len(snippets) >= limit:
            break
        title = (item.get("title") or item.get("positionTitle") or "").strip()
        link = (item.get("link") or item.get("url") or item.get("displayLink") or "").strip()
        snippet = (item.get("snippet") or item.get("description") or item.get("summary") or "").strip()
        if not (title or snippet or link):
            continue
        if link and link in seen_links:
            continue
        seen_links.add(link)
        parts = []
        if title:
            parts.append(title)
        if snippet:
            parts.append(snippet)
        if link:
            parts.append(f"({link})")
        snippets.append(" — ".join(parts))
    return snippets


def _call_serpapi(query: str, max_results: int, api_key: str) -> List[str]:
    try:
        params = {"q": query, "api_key": api_key, "num": max_results, "output": "json"}
        resp = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = (data.get("organic_results") or [])[: max_results]
        return _summarize_search_items(items, max_results)
    except Exception:
        return []


def _call_google_custom_search(query: str, max_results: int, api_key: str, engine_id: str) -> List[str]:
    try:
        params = {"key": api_key, "cx": engine_id, "q": query, "num": max_results}
        resp = requests.get(GOOGLE_CUSTOM_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = (data.get("items") or [])[: max_results]
        return _summarize_search_items(items, max_results)
    except Exception:
        return []


def build_search_context(query: str, max_results: int = 3) -> str:
    cache_key = (query or "").strip().lower()
    if not cache_key:
        return ""
    if cache_key in SEARCH_CACHE:
        return SEARCH_CACHE[cache_key]

    serp_key = get_env("SERPAPI_API_KEY")
    google_key = get_env("GOOGLE_SEARCH_API_KEY")
    engine_id = get_env("GOOGLE_SEARCH_ENGINE_ID") or get_env("GOOGLE_CUSTOM_SEARCH_CX")
    snippets: List[str] = []

    if serp_key:
        snippets = _call_serpapi(query, max_results, serp_key)
    elif google_key and engine_id:
        snippets = _call_google_custom_search(query, max_results, google_key, engine_id)

    if not snippets:
        SEARCH_CACHE[cache_key] = ""
        return ""

    summary = "Search context:\n" + "\n".join(f"{idx + 1}. {snippet}" for idx, snippet in enumerate(snippets))
    SEARCH_CACHE[cache_key] = summary
    return summary


def normalize_model_env_name(model_name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", model_name).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.upper() if normalized else "MODEL"
    if normalized.startswith("GEMINI"):
        return "GEMINI_API_KEY"
    return f"{normalized}_API_KEY"


def get_model_api_key(
    model_name: str,
    default_key: Optional[str],
    openai_override: Optional[str],
    gemini_override: Optional[str],
) -> Optional[str]:
    normalized = model_name.lower()

    # Gemini must use a Gemini key; never fall back to the OpenAI key.
    if "gemini" in normalized:
        return (
            gemini_override
            or get_env(GEMINI_API_KEY_ENV)
            or get_env(GEMINI_API_KEY_ENV_FALLBACK)
        )

    # OpenAI models use the OpenAI key.
    return openai_override or get_env(DEFAULT_API_KEY_ENV) or default_key


def validate_columns(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in ESSENTIAL_PROMPT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def create_openai_client(api_key: str):
    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=api_key)
    openai.api_key = api_key
    return openai


def is_gemini_model(model_name: str) -> bool:
    return "gemini" in model_name.lower()


# Helper to normalize Gemini model aliases to real model IDs.
def resolve_gemini_model_name(model_name: str) -> str:
    """Normalize Gemini model aliases to real model IDs.

    The Generative Language API requires a concrete model name like
    `gemini-2.0-flash` or `gemini-2.5-pro`.
    """
    name = (model_name or "").strip()
    if not name:
        return name

    # Common shorthand aliases people use.
    lowered = name.lower()
    # Older Gemini 1.5 aliases were commonly used historically, but may be unavailable.
    # Map them to a currently-supported model by default.
    if lowered in {
        "gemini-1.5",
        "gemini-1.5-latest",
        "gemini-1.5-default",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
    }:
        return GEMINI_FALLBACK_MODEL

    # Allow current Gemini 2.x / 2.5 model IDs to pass through.
    if lowered in {"gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"}:
        return lowered

    return name


 # NOTE: Gemini calls are made to the Gemini Developer API (not OpenAI Responses).
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_BASE = get_env("GEMINI_API_BASE", DEFAULT_GEMINI_API_BASE)


def build_gemini_auth_headers(api_key: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    headers = {"Content-Type": "application/json"}
    params: Dict[str, str] = {}
    if not api_key:
        return headers, params
    cleaned = api_key.strip()
    if cleaned.startswith("ya29.") or cleaned.startswith("1/"):
        headers["Authorization"] = f"Bearer {cleaned}"
    else:
        params["key"] = cleaned
    return headers, params


def fetch_openai_batch(
    model_name: str,
    api_key: str,
    batch: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Tuple[Dict[str, Any], str, str, List[str], List[str]]]:
    """Fetch a batch of prompts using an OpenAI-compatible API.

    Supports both:
    - New OpenAI SDK (`openai.OpenAI(...).responses.create`)
    - Legacy OpenAI SDK (`openai==0.28.x` with `openai.ChatCompletion.create`)

    NOTE: When using the legacy SDK, we call the model once per record.
    """

    client = create_openai_client(api_key)

    # --- New SDK path (Responses API) ---
    if hasattr(client, "responses"):
        try:
            results: List[Tuple[Dict[str, Any], str, str, List[str], List[str]]] = []

            # The Responses API expects ONE conversation per call.
            # If batch_size > 1, we execute multiple calls (still parallelized upstream).
            for record in batch:
                prompt_text = str(record.get("prompt", ""))

                create_kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "input": [
                        {
                            "role": "system",
                            "content": (
                                "If web search is helpful, reformulate the user request into 3–5 diverse "
                                "search queries, run the necessary searches, and use the results. "
                                "In your final answer, include a Sources section listing the URLs you relied on."
                            ),
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    "tools": [OPENAI_WEB_SEARCH_TOOL],
                    # Let the model decide when/how often to search so expanded queries can appear.
                    "include": ["web_search_call.action.sources"],
                    "max_output_tokens": args.max_tokens,
                    "timeout": args.timeout,
                }

                # GPT-5 / GPT-5.2 family does not support temperature; only include it for other models.
                if args.temperature is not None and not model_name.lower().startswith("gpt-5"):
                    create_kwargs["temperature"] = args.temperature

                r = client.responses.create(**create_kwargs)
                used_search = _response_used_web_search(r)
                print(
                    f"[{record.get('prompt_id', '')}] OpenAI web_search used: {'yes' if used_search else 'no'}"
                )

                # Prefer the convenience property when present
                response_text = (getattr(r, "output_text", "") or "").strip()
                item = None
                try:
                    outputs = list(getattr(r, "output", []) or [])
                    item = outputs[0] if outputs else None
                except Exception:
                    item = None

                if not response_text:
                    response_text = extract_text_from_output(item)

                raw_payload: Any = {}
                if hasattr(r, "to_dict"):
                    try:
                        raw_payload = r.to_dict()
                    except Exception:
                        raw_payload = {}
                elif isinstance(r, dict):
                    raw_payload = r
                search_queries, safe_urls = extract_openai_network_data(raw_payload)
                oai_sources = extract_openai_sources(raw_payload)
                # Ensure the model output retains visible links even when the model omits them.
                response_text = append_sources_section(response_text, oai_sources, heading="Sources")
                results.append((record, response_text, model_name, search_queries, safe_urls))

            return results
        except Exception as e:  # pragma: no cover
            error_text = f"[ERROR] {type(e).__name__}: {e}"
            return [(record, error_text, model_name, [], []) for record in batch]

    # --- Legacy SDK path (openai==0.28.x) ---
    try:
        results: List[Tuple[Dict[str, Any], str, str, List[str], List[str]]] = []

        for record in batch:
            prompt_text = str(record.get("prompt", ""))

            # Legacy chat completion
            resp = client.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.timeout,
            )

            # Extract assistant text
            response_text = ""
            try:
                choice0 = (resp.get("choices") or [])[0]
                response_text = ((choice0.get("message") or {}).get("content") or "").strip()
            except Exception:
                response_text = ""

            results.append((record, response_text, model_name, [], []))

        return results
    except Exception as e:  # pragma: no cover
        error_text = f"[ERROR] {type(e).__name__}: {e}"
        return [(record, error_text, model_name, [], []) for record in batch]


def fetch_gemini_batch(
    model_name: str,
    api_key: str,
    batch: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Tuple[Dict[str, Any], str, str, List[str], List[str]]]:
    """Call Gemini via the Generative Language API.

    NOTE: The v1beta API uses `:generateContent` (not `:generateText`).
    """
    headers, params = build_gemini_auth_headers(api_key)
    results: List[Tuple[Dict[str, Any], str, str, List[str], List[str]]] = []
    timeout = args.timeout

    resolved_model = resolve_gemini_model_name(model_name)

    for record in batch:
        prompt_text = str(record.get("prompt", ""))
        if not prompt_text.strip():
            results.append((record, "", model_name, [], []))
            continue

        # Only inject BYO search snippets when Gemini tools are disabled.
        search_context = ""
        if getattr(args, "gemini_search", "auto") == "off":
            search_context = build_search_context(prompt_text)

        prompt_body = prompt_text
        if search_context:
            prompt_body = f"{search_context}\n\n{prompt_text}"

        force_search = getattr(args, "gemini_search", "auto") == "force"
        directive = (
            "Use Google Search grounding to answer. Perform the necessary searches before answering.\n\n"
            if force_search
            else ""
        )

        base_payload: dict[str, object] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": directive + prompt_body}],
                }
            ],
            "generationConfig": {
                "temperature": args.temperature,
                # Gemini uses `maxOutputTokens` rather than OpenAI's `max_tokens`.
                "maxOutputTokens": args.max_tokens,
            },
        }

        def build_payload(include_tools: bool, use_fallback_tool: bool = False) -> dict[str, object]:
            """Build Gemini payload.

            Gemini Developer API supports Google Search grounding via `tools`.
            We try `google_search` first; if rejected, we can retry with
            `google_search_retrieval`.
            """
            payload_copy = deepcopy(base_payload)
            if include_tools:
                payload_copy["tools"] = (
                    [dict(t) for t in GEMINI_TOOLS_FALLBACK]
                    if use_fallback_tool
                    else [dict(t) for t in GEMINI_TOOLS_PRIMARY]
                )
            return payload_copy

        def _post(
            model_id: str, payload_body: dict[str, object]
        ) -> Tuple[Optional[requests.Response], Optional[Dict[str, Any]], Optional[Exception]]:
            try:
                _url = f"{GEMINI_API_BASE}/{model_id}:generateContent"
                _resp = requests.post(_url, headers=headers, params=params, json=payload_body, timeout=timeout)
                _resp.raise_for_status()
                # Force UTF-8 JSON decoding to avoid mojibake if requests guesses a legacy encoding.
                try:
                    body = json.loads(_resp.content.decode("utf-8"))
                except Exception:
                    body = _resp.json()
                return _resp, body, None
            except Exception as _exc:
                try:
                    return _resp, None, _exc  # type: ignore[name-defined]
                except Exception:
                    return None, None, _exc

        # Tool availability is controlled by --gemini-search.
        # auto/force => attempt to include Google Search tools; off => disable tools.
        tools_supported = getattr(args, "gemini_search", "auto") != "off"
        last_tool_mode: str = "none"  # 'google_search', 'google_search_retrieval', or 'none'

        def _send(model_id: str) -> Tuple[Optional[requests.Response], Optional[Dict[str, Any]], Optional[Exception]]:
            payload_body = build_payload(tools_supported, use_fallback_tool=False)
            return _post(model_id, payload_body)

        def is_tool_error(exc: requests.HTTPError) -> bool:
            """Heuristic: detect schema/tool-name errors for Gemini grounding tools."""
            response_text = ""
            resp_obj = getattr(exc, "response", None)
            if resp_obj is not None:
                response_text = resp_obj.text or ""
            message = f"{response_text} {exc}".lower()
            # Common Gemini errors:
            # - "Unknown name 'tools'" / "Unknown name 'toolSets'"
            # - "Unknown name 'google_search'" / "google_search_retrieval"
            return (
                "unknown name" in message
                and (
                    "tools" in message
                    or "toolsets" in message
                    or "google_search" in message
                    or "google_search_retrieval" in message
                )
            )

        def attempt(model_id: str) -> Tuple[Optional[requests.Response], Optional[Dict[str, Any]], Optional[Exception]]:
            """Attempt a Gemini call with tools enabled, with a fallback tool name.

            1) Try tools=google_search
            2) If tool schema/name rejected, retry tools=google_search_retrieval
            3) If still rejected, disable tools and retry (falls back to BYO snippets)
            """
            nonlocal tools_supported
            nonlocal last_tool_mode
            last_tool_mode = "none"

            # First try: primary tool
            last_tool_mode = "google_search" if tools_supported else "none"
            resp, body, err = _post(model_id, build_payload(tools_supported, use_fallback_tool=False))
            if err is None:
                return resp, body, err

            if tools_supported and isinstance(err, requests.HTTPError) and is_tool_error(err):
                # Second try: fallback tool name
                last_tool_mode = "google_search_retrieval"
                resp, body, err = _post(model_id, build_payload(True, use_fallback_tool=True))
                if err is None:
                    return resp, body, err

                # Third try: disable tools entirely
                tools_supported = False
                last_tool_mode = "none"
                resp, body, err = _post(model_id, build_payload(False))
                return resp, body, err

            return resp, body, err

        resp, body, err = attempt(resolved_model)

        # Print Gemini search usage info
        if err is None and body is not None:
            used_search = (last_tool_mode != "none") and _gemini_response_used_search(body)
            print(
                f"[{record.get('prompt_id', '')}] Gemini search used: {'yes' if used_search else 'no'} (mode={last_tool_mode})"
            )
        else:
            print(
                f"[{record.get('prompt_id', '')}] Gemini search used: no (mode={last_tool_mode})"
            )

        # If the model is not available for this API key / API version, retry once with a known-good model.
        if err is not None and isinstance(err, requests.HTTPError):
            status = getattr(resp, "status_code", None)
            if status == 404 and resolved_model != GEMINI_FALLBACK_MODEL:
                resp, body, err = attempt(GEMINI_FALLBACK_MODEL)

        search_queries: List[str] = []
        safe_urls: List[str] = []
        if err is None and body is not None:
            text = ""
            candidates = body.get("candidates") or []
            if candidates:
                content = (candidates[0].get("content") or {})
                parts = content.get("parts") or []
                if parts:
                    text = parts[0].get("text", "") or ""

            formatted_response = str(text).strip()
            if not formatted_response:
                formatted_response = "[ERROR] Gemini returned empty content"
            else:
                # Render grounding citations into the response text (Gemini-chat-like).
                formatted_response = _gemini_render_citations_in_text(formatted_response, body)
            search_queries, safe_urls = extract_gemini_network_data(body)
        else:
            # Include response body where possible to make 404/401/etc actionable.
            extra = ""
            try:
                if resp is not None:
                    extra = f" | body={resp.text[:500]}"
            except Exception:
                pass
            formatted_response = f"[ERROR] {type(err).__name__}: {err}{extra}"
            search_queries, safe_urls = [], []
        results.append((record, formatted_response, model_name, search_queries, safe_urls))

    return results


def run_model_batch(model_name: str, api_key: str, batch: List[Dict[str, Any]], args: argparse.Namespace) -> List[Tuple[Dict[str, Any], str, str, List[str], List[str]]]:
    if is_gemini_model(model_name):
        return fetch_gemini_batch(model_name, api_key, batch, args)
    return fetch_openai_batch(model_name, api_key, batch, args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.csv", help="Path to prompts.csv")
    ap.add_argument("--audiences", default="audiences.csv", help="Path to audiences.csv (unused for now)")
    ap.add_argument("--out", default="responses.csv", help="Path to output responses.csv")
    ap.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to run (defaults to OPENAI_MODEL or gpt-5.2-chat-latest)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Override model (else env OPENAI_MODEL or default gpt-5.2-chat-latest)",
    )
    ap.add_argument(
        "--context-prompts",
        default=DEFAULT_CONTEXT_PROMPTS,
        help="Optional path to context prompts CSV that should also be run through the model(s)",
    )
    ap.add_argument(
        "--openai-api-key",
        default=None,
        help="Explicit API key to use for OpenAI models (overrides OPENAI_API_KEY env var)",
    )
    ap.add_argument(
        "--gemini-api-key",
        default=None,
        help="Explicit API key to use for Gemini models (overrides GEMINIAPI_KEY env var)",
    )
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, only run first N prompts")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--max-tokens", type=int, default=1200, help="Max tokens for completion")
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    ap.add_argument(
        "--gemini-search",
        choices=["auto", "force", "off"],
        default="auto",
        help="Gemini search behavior: auto=make Google Search tool available, force=require search, off=disable tools (uses BYO search_context if configured)",
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Prompts per Responses API call (min 1)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent batches to execute per model")
    args = ap.parse_args()

    default_api_key = get_env(DEFAULT_API_KEY_ENV)
    explicit_models = args.models or args.model or get_env("OPENAI_MODEL")
    if explicit_models:
        model_names = [m.strip() for m in explicit_models.split(",") if m.strip()]
        if not model_names:
            raise SystemExit("No models specified via --models or OPENAI_MODEL")
    else:
        model_names = list(DEFAULT_MODELS)
    model_api_keys: Dict[str, str] = {}
    for model_name in model_names:
        key = get_model_api_key(
            model_name,
            default_api_key,
            args.openai_api_key,
            args.gemini_api_key,
        )
        if not key:
            env_name = normalize_model_env_name(model_name)
            raise SystemExit(
                f"Missing API key for model {model_name}. "
                f"Set {env_name} or {DEFAULT_API_KEY_ENV}."
            )
        model_api_keys[model_name] = key

    prompts_df = pd.read_csv(args.prompts)
    validate_columns(prompts_df, args.prompts)

    context_prompts_df: Optional[pd.DataFrame] = None
    if args.context_prompts:
        if os.path.exists(args.context_prompts):
            context_prompts_df = pd.read_csv(args.context_prompts)
            validate_columns(context_prompts_df, args.context_prompts)
        else:
            print(f"[info] Context prompts file '{args.context_prompts}' not found; skipping.")

    combined_df = prompts_df
    if context_prompts_df is not None:
        combined_df = pd.concat([prompts_df, context_prompts_df], ignore_index=True, copy=False)
        total_context = len(context_prompts_df)
        print(f"[info] Added {total_context} context prompts from {args.context_prompts}")

    if args.max_rows and args.max_rows > 0:
        combined_df = combined_df.head(args.max_rows).copy()

    out_rows = []

    batch_size = max(1, args.batch_size)
    prompt_records = combined_df.to_dict("records")
    total_prompts = len(prompt_records)
    expected_results = total_prompts * len(model_names)

    def print_progress(processed: int, total: int, prefix: str = "Processed") -> None:
        msg = f"{prefix} {processed}/{total}"
        print(msg, end="\r", flush=True)

    def append_result(
        record: Dict[str, Any],
        response_text: str,
        model_name: str,
        search_queries: List[str],
        safe_urls: List[str],
    ) -> None:
        row = dict(record)
        row["response"] = normalize_text_human_readable(response_text)
        row["model"] = model_name
        row["search_queries"] = json.dumps([
            normalize_text_human_readable(q) for q in search_queries
        ], ensure_ascii=False)
        row["safe_urls"] = json.dumps(safe_urls, ensure_ascii=False)
        out_rows.append(row)
        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

        if (len(out_rows) % 10) == 0 or len(out_rows) == expected_results:
            print_progress(len(out_rows), expected_results)
            if len(out_rows) == expected_results:
                print()

    for model_name in model_names:
        print(f"Running prompts with model {model_name}")
        before = len(out_rows)
        batches = []
        for start in range(0, total_prompts, batch_size):
            batch = prompt_records[start : start + batch_size]
            if batch:
                batches.append(batch)
        worker_count = max(1, args.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    run_model_batch,
                    model_name,
                    model_api_keys[model_name],
                    batch,
                    args,
                )
                for batch in batches
            ]
            for future in concurrent.futures.as_completed(futures):
                for record, response_text, returned_model, search_queries, safe_urls in future.result():
                    append_result(record, response_text, returned_model, search_queries, safe_urls)
        processed = len(out_rows) - before
        print()
        print(f"Finished collecting {processed} responses from {model_name}")

    base_columns = list(prompts_df.columns)
    out_columns = base_columns + ["response", "model", "search_queries", "safe_urls"]
    out_df = pd.DataFrame(out_rows, columns=out_columns)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()

