#!/usr/bin/env python3
"""Generate Phase 1.5 follow-up responses for outstanding analysis rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import openai
import requests


def _load_env(path: Path | str | None = None) -> None:
    target = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not target.is_file():
        return
    with target.open("r", encoding="utf-8") as env_file:
        for raw in env_file:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value


_load_env()


PERSONA_FIELDS: List[tuple[str, str]] = [
    ("title", "Title"),
    ("trends", "Trends"),
    ("summary", "Summary"),
    ("motivations", "Motivations"),
    ("characteristics", "Characteristics"),
    ("minAge", "Min age"),
    ("maxAge", "Max age"),
    ("country", "Country"),
    ("preferred_models", "Preferred models"),
    ("conversation_style", "Conversation style"),
    ("search_behaviors", "Search behaviors"),
]


DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
# Prefer standard GEMINI_API_KEY; also support legacy GEMINIAPI_KEY used in older workflows.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_API_KEY_ENV_FALLBACK = "GEMINIAPI_KEY"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_BASE = os.environ.get("GEMINI_API_BASE", DEFAULT_GEMINI_API_BASE)

GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
GEMINI_TOOLS_PRIMARY = [{"google_search": {}}]
GEMINI_TOOLS_FALLBACK = [{"google_search_retrieval": {}}]

# Latest ChatGPT snapshot alias (kept in sync with Phase 1)
LATEST_CHATGPT_MODEL = "gpt-5.2-chat-latest"


OPENAI_WEB_SEARCH_TOOL = {"type": "web_search"}

# --- Phase 1.5 internal classifier (no output column) ---
P1_STATUS_COMPLETE = "COMPLETE_ANSWER"
P1_STATUS_NEEDS = "NEEDS_CLARIFICATION"
P1_STATUS_FAILED = "FAILED_ANSWER"
P1_STATUS_ALLOWED = {P1_STATUS_COMPLETE, P1_STATUS_NEEDS, P1_STATUS_FAILED}

CLASSIFIER_MAX_TOKENS = int(os.environ.get("P1_STATUS_MAX_TOKENS", "20"))


def _normalize_p1_status(raw: str) -> str:
    s = (raw or "").strip().upper()
    # Allow minor variations
    s = s.replace(" ", "_")
    if s in P1_STATUS_ALLOWED:
        return s
    if "COMPLETE" in s:
        return P1_STATUS_COMPLETE
    if "FAILED" in s or "ERROR" in s or "NON" in s:
        return P1_STATUS_FAILED
    return P1_STATUS_NEEDS


def classify_p1_status_openai(
    prompt_text: str,
    phase1_response: str,
    model: str,
    api_key: str,
    prompt_id: str = "",
) -> str:
    """Classify a Phase 1 assistant response.

    IMPORTANT: This must NOT force web search tools. Keep it cheap and deterministic.
    Returns one of: COMPLETE_ANSWER, NEEDS_CLARIFICATION, FAILED_ANSWER.
    """
    system = (
        "You are a strict but balanced classifier for chat quality. "
        "Given a user's original question and the assistant's answer, output EXACTLY one label: "
        "COMPLETE_ANSWER, NEEDS_CLARIFICATION, or FAILED_ANSWER. "
        "Balanced policy:\n"
        '- COMPLETE_ANSWER: The assistant provides a substantive answer (steps, explanation, recommendations, etc.), EVEN IF it ends with optional follow-up questions ("If you want, tell me X...").\n'
        '- NEEDS_CLARIFICATION: The assistant explicitly asks the user for info BEFORE it can answer (it defers the answer), and provides little/no actionable guidance (mostly questions or "it depends").\n'
        "- FAILED_ANSWER: The assistant is evasive, irrelevant, empty, or only meta/refusal."
    )
    user = (
        "User question:\n"
        f"{(prompt_text or '').strip()}\n\n"
        "Assistant answer:\n"
        f"{(phase1_response or '').strip()}\n\n"
        "Output one label only."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        # Prefer Responses API when present, but do NOT pass tools/tool_choice.
        if hasattr(client, "responses"):
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                "max_output_tokens": CLASSIFIER_MAX_TOKENS,
            }
            # GPT-5 / GPT-5.2 models do not support temperature.
            if not (model or "").lower().startswith("gpt-5"):
                create_kwargs["temperature"] = 0
            resp = client.responses.create(**create_kwargs)
            label = _extract_response_text(resp)
            return _normalize_p1_status(label)
        # Fallback to chat.completions
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            messages=messages,
        )
        label = resp.choices[0].message.content or ""
        return _normalize_p1_status(label)

    # Legacy SDK
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        messages=messages,
    )
    label = resp["choices"][0]["message"]["content"] or ""
    return _normalize_p1_status(label)


def classify_p1_status_gemini(
    prompt_text: str,
    phase1_response: str,
    model: str,
    api_key: str,
    timeout: float,
    prompt_id: str = "",
) -> str:
    """Gemini version of the Phase 1 status classifier.

    Uses generateContent WITHOUT tools to avoid unnecessary search/grounding.
    """
    resolved_model = resolve_gemini_model_name(model)
    headers, params = build_gemini_auth_headers(api_key)

    system = (
        "You are a strict but balanced classifier for chat quality. "
        "Output EXACTLY one label: COMPLETE_ANSWER, NEEDS_CLARIFICATION, or FAILED_ANSWER. "
        "Balanced policy:\n"
        '- COMPLETE_ANSWER: The assistant provides a substantive answer (steps, explanation, recommendations, etc.), EVEN IF it ends with optional follow-up questions ("If you want, tell me X...").\n'
        '- NEEDS_CLARIFICATION: The assistant explicitly asks the user for info BEFORE it can answer (it defers the answer), and provides little/no actionable guidance (mostly questions or "it depends").\n'
        "- FAILED_ANSWER: The assistant is evasive, irrelevant, empty, or only meta/refusal."
    )
    user = (
        "User question:\n"
        f"{(prompt_text or '').strip()}\n\n"
        "Assistant answer:\n"
        f"{(phase1_response or '').strip()}\n\n"
        "Output one label only."
    )

    payload: dict[str, object] = {
        "contents": [
            {"role": "user", "parts": [{"text": user}]},
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": CLASSIFIER_MAX_TOKENS,
        },
        "systemInstruction": {"parts": [{"text": system}]},
    }

    url = f"{GEMINI_API_BASE}/{resolved_model}:generateContent"
    resp = requests.post(
        url, headers=headers, params=params, json=payload, timeout=timeout
    )
    resp.raise_for_status()
    body = resp.json() or {}

    candidates = body.get("candidates") or []
    text = ""
    if candidates:
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if parts:
            text = parts[0].get("text", "") or ""

    label = (text or "").strip()
    return _normalize_p1_status(label)


def classify_p1_status(
    prompt_text: str,
    phase1_response: str,
    model: str,
    api_key: str,
    timeout: float,
    prompt_id: str = "",
) -> str:
    if is_gemini_model(model):
        return classify_p1_status_gemini(
            prompt_text=prompt_text,
            phase1_response=phase1_response,
            model=model,
            api_key=api_key,
            timeout=timeout,
            prompt_id=prompt_id,
        )
    return classify_p1_status_openai(
        prompt_text=prompt_text,
        phase1_response=phase1_response,
        model=model,
        api_key=api_key,
        prompt_id=prompt_id,
    )


def _iter_contents(item: Any) -> List[Any]:
    if not item:
        return []
    if isinstance(item, dict):
        return item.get("content") or []
    contents = getattr(item, "content", None)
    if contents is None:
        return []
    return contents


def _extract_text_from_output(item: Any) -> str:
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
        fallback = (
            item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        )
        if fallback:
            texts.append(str(fallback))
    return "\n".join([t for t in texts if t]).strip()


def _extract_response_text(response: Any) -> str:
    text = (getattr(response, "output_text", "") or "").strip()
    if text:
        return text
    outputs = list(getattr(response, "output", []) or [])
    if outputs:
        return _extract_text_from_output(outputs[0])
    return ""


def _response_used_web_search(response: Any) -> bool:
    outputs = list(getattr(response, "output", []) or [])
    for item in outputs:
        if isinstance(item, dict):
            if item.get("type") == "web_search_call":
                return True
        else:
            if getattr(item, "type", None) == "web_search_call":
                return True
    return False


def _gemini_response_used_search(body: Optional[Dict[str, Any]]) -> bool:
    if not body or not isinstance(body, dict):
        return False

    candidates = body.get("candidates") or []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue

        gm = cand.get("groundingMetadata") or cand.get("grounding_metadata")
        if isinstance(gm, dict):
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
            return True

        content = cand.get("content")
        if isinstance(content, dict):
            gm2 = content.get("groundingMetadata") or content.get("grounding_metadata")
            if isinstance(gm2, dict) and gm2:
                return True

    return False


# --- Gemini citation rendering (Gemini chat style) ---

_GROUNDING_REDIRECT_CACHE: Dict[str, str] = {}
_GROUNDING_REDIRECT_CACHE_LOCK = Lock()
_VERTEX_REDIRECT_HOST = "vertexaisearch.cloud.google.com/grounding-api-redirect"


def resolve_grounding_redirect(url: str, timeout: float = 10.0) -> str:
    """Resolve vertex grounding redirect wrappers via HEAD then GET with caching."""
    target = (url or "").strip()
    if not target or _VERTEX_REDIRECT_HOST not in target:
        return target
    with _GROUNDING_REDIRECT_CACHE_LOCK:
        cached = _GROUNDING_REDIRECT_CACHE.get(target)
    if cached:
        return cached

    headers = {"User-Agent": "Mozilla/5.0"}
    final_url = target
    for method in ("head", "get"):
        try:
            resp_method = getattr(requests, method)
            resp = resp_method(
                target, allow_redirects=True, timeout=timeout, headers=headers
            )
            resp.raise_for_status()
            resolved = getattr(resp, "url", "")
            if isinstance(resolved, str) and resolved:
                final_url = resolved.strip()
            break
        except requests.RequestException:
            continue

    with _GROUNDING_REDIRECT_CACHE_LOCK:
        _GROUNDING_REDIRECT_CACHE[target] = final_url
    return final_url


_SOURCE_PLACEHOLDER_TEMPLATE = "<<AI_SRC_{number}>>"
_SOURCE_PLACEHOLDER_PATTERN = re.compile(r"<<AI_SRC_(\d+)>>")
_MAX_SOURCE_ENTRIES = 10


def _gemini_render_citations_in_text(text: str, body: Optional[Dict[str, Any]]) -> str:
    """Insert grounding sources inline where Gemini signals them."""
    base_text = (text or "").strip()
    if not base_text or not body or not isinstance(body, dict):
        return base_text

    candidates = body.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return base_text

    gm = candidates[0].get("groundingMetadata") or candidates[0].get(
        "grounding_metadata"
    )
    if not isinstance(gm, dict):
        return base_text

    chunks = gm.get("groundingChunks") or gm.get("grounding_chunks") or []
    supports = gm.get("groundingSupports") or gm.get("grounding_supports") or []
    if not isinstance(chunks, list) or not chunks or not isinstance(supports, list):
        return base_text

    chunk_info_by_index: Dict[int, Dict[str, str]] = {}
    next_number = 1
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        uri = str(web.get("uri") or "").strip()
        if not uri:
            continue
        title = str(web.get("title") or "").strip()
        resolved = resolve_grounding_redirect(uri)
        info = {
            "number": next_number,
            "title": title,
            "uri": uri,
            "resolved_url": resolved or uri,
        }
        chunk_info_by_index[idx] = info
        next_number += 1

    if not chunk_info_by_index:
        return base_text

    def _adjust_position(pos: int) -> int:
        idx = max(0, min(pos, len(base_text)))
        while idx > 0 and base_text[idx - 1].isspace():
            idx -= 1
        return idx

    inserts: List[Tuple[int, str, int]] = []
    seen_positions: set[Tuple[int, int]] = set()
    used_info: Dict[int, Dict[str, str]] = {}

    for sup in supports:
        if not isinstance(sup, dict):
            continue
        seg = sup.get("segment")
        if not isinstance(seg, dict):
            continue
        raw_end = seg.get("endIndex")
        try:
            end_idx = int(raw_end)
        except (TypeError, ValueError):
            continue
        gci = (
            sup.get("groundingChunkIndices") or sup.get("grounding_chunk_indices") or []
        )
        if not isinstance(gci, list):
            continue
        chunk_idx = None
        for candidate in gci:
            try:
                candidate_idx = int(candidate)
            except (TypeError, ValueError):
                continue
            if candidate_idx in chunk_info_by_index:
                chunk_idx = candidate_idx
                break
        if chunk_idx is None:
            continue

        info = chunk_info_by_index.get(chunk_idx)
        if not info:
            continue
        number = info["number"]
        pos = max(0, min(end_idx, len(base_text)))
        pos = _adjust_position(pos)
        if pos < 0 or pos > len(base_text):
            continue
        prefix = ""
        if pos > 0 and not base_text[pos - 1].isspace():
            prefix = " "
        placeholder = _SOURCE_PLACEHOLDER_TEMPLATE.format(number=number)
        marker = f"{prefix}{placeholder}"
        key = (pos, number)
        if key in seen_positions:
            continue
        seen_positions.add(key)
        inserts.append((pos, marker, number))
        used_info[number] = info

    if not inserts:
        return base_text

    rendered = base_text
    inserts.sort(key=lambda item: item[0], reverse=True)
    for pos, marker, _ in inserts:
        if pos < 0 or pos > len(rendered):
            continue
        rendered = rendered[:pos] + marker + rendered[pos:]

    def _replace_marker(match: re.Match[str]) -> str:
        num = int(match.group(1))
        info = used_info.get(num)
        if not info:
            return match.group(0)

        resolved = info.get("resolved_url") or info.get("uri") or ""
        if not resolved:
            return match.group(0)

        # Extract domain for compact inline attribution
        try:
            domain = resolved.split("//", 1)[-1].split("/", 1)[0]
        except Exception:
            domain = resolved

        return f"(AI_CITE_DOMAIN:{domain})"

    rendered_with_urls = _SOURCE_PLACEHOLDER_PATTERN.sub(_replace_marker, rendered)

    # --- Anti "link dump" hardening ---
    # Goal: keep ONLY the URLs that correspond to grounding supports (our inline citations),
    # and strip any other raw URLs that Gemini might dump into the answer body.
    #
    # Strategy:
    # 1) Turn our citation tokens (AI_CITE:<url>) into protected sentinels.
    # 2) Strip ALL remaining URLs anywhere in the answer body.
    # 3) Restore protected sentinels as plain (domain) citations.

    # Matches any http(s) URL token (greedy up to whitespace).
    _URL_ANYWHERE = re.compile(r"https?://\S+", re.IGNORECASE)

    # 1) Protect our citation domains first.
    keep_map: Dict[str, str] = {}
    keep_idx = 0

    def _protect_ai_cite(match: re.Match[str]) -> str:
        nonlocal keep_idx
        url = (match.group(1) or "").strip()
        if not url:
            return match.group(0)
        token = f"<<KEEP_CITE_{keep_idx}>>"
        keep_map[token] = url
        keep_idx += 1
        return token

    # Convert (AI_CITE_DOMAIN:<domain>) into protected tokens.
    rendered_with_urls = re.sub(
        r"\(AI_CITE_DOMAIN:([^\s)]+)\)",
        _protect_ai_cite,
        rendered_with_urls,
        flags=re.IGNORECASE,
    )

    # 2) Remove ANY remaining URLs (this nukes the parenthetical link-dumps and stray URLs).
    rendered_with_urls = _URL_ANYWHERE.sub("", rendered_with_urls)

    # 3) Restore protected citations as compact domain labels
    for token, domain in keep_map.items():
        rendered_with_urls = rendered_with_urls.replace(token, f"({domain})")

    # 4) Cleanup artifacts after removals.
    rendered_with_urls = re.sub(r"\(\s*\)", "", rendered_with_urls)  # empty parens
    rendered_with_urls = re.sub(r"\s{2,}", " ", rendered_with_urls)
    rendered_with_urls = re.sub(r"\s+([,.;:!?])", r"\1", rendered_with_urls)
    rendered_with_urls = rendered_with_urls.strip()

    # Build bottom Sources list ONLY for grounding chunks that were NOT attributed inline.
    unique_entries: list[Tuple[int, str, str]] = []
    seen_urls: set[str] = set()

    for idx, info in chunk_info_by_index.items():
        number = info.get("number")
        # Skip sources that were already attributed inline
        if number in used_info:
            continue

        resolved = info.get("resolved_url") or info.get("uri") or ""
        if not resolved or resolved in seen_urls:
            continue

        seen_urls.add(resolved)
        title = info.get("title") or resolved or "Source"
        unique_entries.append((number, title, resolved))

    if unique_entries:
        truncated = len(unique_entries) > _MAX_SOURCE_ENTRIES
        display_entries = unique_entries[:_MAX_SOURCE_ENTRIES]
        sources_lines = ["Sources:"]
        for number, title, resolved in display_entries:
            sources_lines.append(f"[{number}] {title} - {resolved}")
        if truncated:
            omitted = len(unique_entries) - _MAX_SOURCE_ENTRIES
            sources_lines.append(f"... ({omitted} more sources omitted)")
        # Note: Sources section intentionally keeps full URLs; only the answer body is de-linked.
        rendered_with_urls = f"{rendered_with_urls}\n\n" + "\n".join(sources_lines)

    return rendered_with_urls


FOLLOWUP_PROMPT_MODEL = os.environ.get("FOLLOWUP_PROMPT_MODEL", LATEST_CHATGPT_MODEL)
FOLLOWUP_PROMPT_TEMPERATURE = float(
    os.environ.get("FOLLOWUP_PROMPT_TEMPERATURE", "0.2")
)
FOLLOWUP_PROMPT_MAX_TOKENS = int(os.environ.get("FOLLOWUP_PROMPT_MAX_TOKENS", "120"))


def read_csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [name for name in (reader.fieldnames or []) if name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1.5 GEO follow-up prompts by flagging follow-up candidates from Phase 1 responses."
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("responses.csv"),
        help="Phase 1 responses CSV to scan for clarification follow-ups (default assumes working dir).",
    )
    parser.add_argument(
        "--audience",
        type=Path,
        default=Path("audience_habbits.csv"),
        help="Persona guidance CSV that drives the follow-up tone (default assumes working dir).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Phase15_Responses.csv"),
        help="Output CSV that will carry the new Phase 1.5 responses (saved in working dir).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LATEST_CHATGPT_MODEL,
        help="OpenAI model to use for Phase 1.5 reasoning (default: latest ChatGPT snapshot).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the OpenAI call.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Max tokens for the OpenAI response.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Override the OPENAI_API_KEY environment variable for OpenAI models.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Override the GEMINI_API_KEY environment variable for Gemini models.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for Gemini requests.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are an expert GEO analyst.",
        help="System prompt for the Phase 1.5 assistant.",
    )
    parser.add_argument(
        "--pmg-client-json",
        type=Path,
        default=Path("pmgclient.json"),
        help="JSON dump listing Alli clients (id + name).",
    )
    return parser.parse_args()


def is_gemini_model(model_name: Optional[str]) -> bool:
    return bool(model_name and "gemini" in model_name.lower())


def resolve_gemini_model_name(model_name: str) -> str:
    name = (model_name or "").strip()
    if not name:
        return name
    lowered = name.lower()
    deprecated_aliases = {
        "gemini-1.5",
        "gemini-1.5-latest",
        "gemini-1.5-default",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
    }
    if lowered in deprecated_aliases:
        return GEMINI_FALLBACK_MODEL
    allowed = {
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    }
    if lowered in allowed:
        return lowered
    return name


def build_gemini_auth_headers(
    api_key: Optional[str],
) -> tuple[Dict[str, str], Dict[str, str]]:
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


def get_model_api_key(model_name: str, args: argparse.Namespace) -> Optional[str]:
    if is_gemini_model(model_name):
        return (
            args.gemini_api_key
            or os.environ.get(GEMINI_API_KEY_ENV)
            or os.environ.get(GEMINI_API_KEY_ENV_FALLBACK)
        )
    return args.openai_api_key or os.environ.get(DEFAULT_OPENAI_API_KEY_ENV)


def build_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    """Convert OpenAI-style chat messages into Gemini `contents`.

    Gemini `generateContent` expects roles like `user` and `model`.
    It does not accept `system` inside `contents` (use `systemInstruction` instead).
    """
    contents: list[dict[str, object]] = []
    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        if role == "system":
            continue

        text = (message.get("content") or "").strip()
        if not text:
            continue

        gemini_role = "model" if role in {"assistant", "model"} else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})
    return contents


def load_pmg_clients(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        for key in ("results", "data"):
            if key in raw and isinstance(raw[key], list):
                return [dict(item) for item in raw[key] if isinstance(item, dict)]
        return [raw]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    return []


def build_pmg_maps(
    clients: Iterable[Mapping[str, str]],
) -> tuple[Dict[str, dict[str, str]], Dict[str, dict[str, str]]]:
    by_id: Dict[str, dict[str, str]] = {}
    by_name: Dict[str, dict[str, str]] = {}
    for client in clients:
        client_id = (client.get("id") or client.get("client_id") or "").strip()
        client_name = (client.get("name") or "").strip()
        if client_id:
            by_id[client_id] = {"id": client_id, "name": client_name}
        if client_name:
            by_name[client_name.lower()] = {
                "id": client_id or client_name,
                "name": client_name,
            }
    return by_id, by_name


def build_phase15_output_fieldnames(response_fieldnames: list[str]) -> list[str]:
    """Return the Phase 1.5 export headers, including follow-up columns."""
    output_fieldnames = list(response_fieldnames)
    for field in ("p1c_prompt", "p1c_response"):
        if field not in output_fieldnames:
            output_fieldnames.append(field)
    return output_fieldnames


def write_header_only_csv(path: Path, fieldnames: list[str]) -> None:
    """Write just the header row for a CSV so downstream scripts still have column names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def live_print(message: str) -> None:
    print(message, flush=True)


def _print_progress(current: int, total: int) -> None:
    print(f"\rProgress: {current}/{total}", end="", flush=True)


# --- Phase 1 follow-up gate (cheap; runs before any extra API calls) ---
FOLLOWUP_GATE_PHRASES = (
    "can you clarify",
    "could you clarify",
    "please clarify",
    "can you provide more",
    "could you provide more",
    "need more information",
    "i need more information",
    "to help you",
    "before i can",
    "before i answer",
    "i need to know",
    "what is your",
    "what's your",
    "which one",
    "which type",
    "which brand",
    "which model",
    "what is your budget",
    "what's your budget",
    "what is your age",
    "how old are you",
    "what country",
    "what location",
    "where are you located",
)

FOLLOWUP_REFUSAL_PHRASES = (
    # Common tool-refusal / real-time language
    "unable to access real-time",
    "can't access real-time",
    "cannot access real-time",
    "don't have access to real-time",
    "do not have access to real-time",
    "unable to access real time",
    "can't access real time",
    "cannot access real time",
    "unable to access local pricing",
    "can't access local pricing",
    "cannot access local pricing",
    "unable to access pricing",
    "can't access pricing",
    "cannot access pricing",
)

# A question mark alone is not enough (answers can include rhetorical questions).
# Gate only when the assistant is asking the USER for info.
_USER_QUESTION_RE = re.compile(
    r"\b("
    r"can you|could you|would you|do you|did you|are you|"
    r"what is your|what's your|which of these|which one|"
    r"where are you|where do you|what country|what location|"
    r"how old are you|what is your age|what's your age)\b",
    re.IGNORECASE,
)


def _has_substantive_answer(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()

    if re.search(r"\n\s*\d+\.", t):
        return True
    if "\n- " in t or "\n* " in t:
        if len(re.findall(r"\n\s*[-*]\s+", t)) >= 3:
            return True
    if re.search(
        r"\b(step|steps|first|second|third|here are|try these|you can)\b", low
    ):
        if len(re.findall(r"\b\w+\b", t)) >= 80:
            return True
    return False


def _question_addresses_user(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    # Evaluate only around question marks to avoid matching unrelated words elsewhere
    q_positions = [m.start() for m in re.finditer(r"\?", t)]
    if not q_positions:
        return False

    for pos in q_positions:
        start = max(0, pos - 120)
        end = min(len(t), pos + 120)
        window = t[start:end]

        # Strong indicators in the local window
        if _USER_QUESTION_RE.search(window):
            return True

        # Weaker local indicator: question mentions 'you/your'
        if re.search(r"\b(you|your|yours)\b", window, re.IGNORECASE):
            return True

    return False


def detects_follow_up_question(phase1_response: str) -> bool:
    """Return True if Phase 1 likely needs a user follow-up.

    This is intentionally a CHEAP gate to reduce extra API calls.
    It is allowed to have some false positives; the model-based classifier will reduce them.
    """
    text = (phase1_response or "").strip()
    lowered = text.lower()
    if not text:
        return True  # empty response is effectively a failure

    user_request_detected = False
    if "?" in lowered and _question_addresses_user(text):
        user_request_detected = True
    else:
        for p in FOLLOWUP_GATE_PHRASES:
            if p in lowered:
                user_request_detected = True
                break

    if user_request_detected and not _has_substantive_answer(text):
        return True

    # Refusal / real-time/local pricing language that we want Phase 1.5 to correct
    for p in FOLLOWUP_REFUSAL_PHRASES:
        if p in lowered:
            return True

    # "I'm unable to" + "However" patterns often indicate refusal + suggestion to search elsewhere
    if (
        "i'm unable" in lowered
        or "i am unable" in lowered
        or "i can't" in lowered
        or "i cannot" in lowered
    ):
        if "however" in lowered and (
            "check" in lowered or "visit" in lowered or "look up" in lowered
        ):
            return True

    return False


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]


def normalize_persona_row(row: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if not key:
            continue
        normalized[key.strip()] = (value or "").strip()
    return normalized


def build_persona_map(rows: Iterable[Mapping[str, str]]) -> Dict[str, dict[str, str]]:
    persona_map: dict[str, dict[str, str]] = {}
    for raw_row in rows:
        persona = normalize_persona_row(raw_row)
        persona_id = persona.get("persona_id", "")
        if not persona_id:
            continue
        persona_map[persona_id] = persona
    return persona_map


def find_persona(
    persona_map: Mapping[str, dict[str, str]], prompt_id: str
) -> Optional[dict[str, str]]:
    """Locate persona row for a given prompt_id.

    prompt_id may be reused across models and may include extra segments.
    We try to extract a 24-hex persona_id anywhere in the string first,
    then fall back to the historical suffix matching.
    """
    if not prompt_id:
        return None

    pid_text = (prompt_id or "").strip()
    if not pid_text:
        return None

    # 1) Strong match: any 24-hex token in the prompt_id
    # (Mongo-style IDs like 690a456469242b972dc8b2c8)
    for token in re.findall(r"[0-9a-fA-F]{24}", pid_text):
        token_lower = token.lower()
        if token_lower in persona_map:
            return persona_map[token_lower]
        # Some CSVs may preserve original casing; check both.
        if token in persona_map:
            return persona_map[token]

    # 2) Historical fallback: last underscore segment
    suffix = pid_text.split("_")[-1]
    if suffix:
        if suffix in persona_map:
            return persona_map[suffix]
        lower_suffix = suffix.lower()
        if lower_suffix in persona_map:
            return persona_map[lower_suffix]
        for pid, persona in persona_map.items():
            if pid.startswith(suffix) or pid.lower().startswith(lower_suffix):
                return persona

    return None


def persona_context_text(persona: Optional[Mapping[str, str]]) -> str:
    if not persona:
        return ""
    lines: list[str] = []
    for key, label in PERSONA_FIELDS:
        value = persona.get(key, "")
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def persona_context_dict(persona: Optional[Mapping[str, str]]) -> dict[str, str]:
    context: dict[str, str] = {}
    if not persona:
        return context
    for key, _label in PERSONA_FIELDS:
        context[key] = persona.get(key, "")
    return context


# Helper to build a composite key for Phase 1 responses.
def make_response_key(prompt_id: str, model: Optional[str]) -> tuple[str, str]:
    """Uniquely identify a Phase 1 response row.

    prompt_id can be reused across multiple models, so the key must include model.
    """
    pid = (prompt_id or "").strip()
    m = (model or "").strip() or "unspecified"
    return (pid, m)


def call_openai(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    prompt_id: str = "",
) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Phase 1.5.")

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        if hasattr(client, "responses"):
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                "tools": [OPENAI_WEB_SEARCH_TOOL],
                # Auto mode: do NOT force tool usage; let the model decide when to search.
                "include": ["web_search_call.action.sources"],
                "max_output_tokens": max_tokens,
            }
            # GPT-5 / GPT-5.2 models do not support temperature; only include it for other models.
            if temperature is not None and not (model or "").lower().startswith(
                "gpt-5"
            ):
                create_kwargs["temperature"] = temperature

            response = client.responses.create(**create_kwargs)
            used_search = _response_used_web_search(response)
            print(
                f"[{prompt_id}] OpenAI web_search used: {'yes' if used_search else 'no'}"
            )
            return _extract_response_text(response)
        chat_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None and not (model or "").lower().startswith("gpt-5"):
            chat_kwargs["temperature"] = temperature
        response = client.chat.completions.create(**chat_kwargs)
        content = response.choices[0].message.content or ""
    else:
        openai.api_key = api_key
        legacy_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None and not (model or "").lower().startswith("gpt-5"):
            legacy_kwargs["temperature"] = temperature
        response = openai.ChatCompletion.create(**legacy_kwargs)
        content = response["choices"][0]["message"]["content"] or ""
    return content.strip()


# --- Plain OpenAI call with NO tools/search for user follow-up generation ---
def call_openai_plain(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    prompt_id: str = "",
) -> str:
    """OpenAI call that does NOT enable tools/search.

    Use this for generating the *user* follow-up message (p1c_prompt), where we never want citations,
    browsing behavior, or assistant-style sourcing.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        if hasattr(client, "responses"):
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                "max_output_tokens": max_tokens,
            }
            if temperature is not None and not (model or "").lower().startswith(
                "gpt-5"
            ):
                create_kwargs["temperature"] = temperature
            response = client.responses.create(**create_kwargs)
            return _extract_response_text(response)
        chat_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None and not (model or "").lower().startswith("gpt-5"):
            chat_kwargs["temperature"] = temperature
        response = client.chat.completions.create(**chat_kwargs)
        content = response.choices[0].message.content or ""
    else:
        openai.api_key = api_key
        legacy_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None and not (model or "").lower().startswith("gpt-5"):
            legacy_kwargs["temperature"] = temperature
        response = openai.ChatCompletion.create(**legacy_kwargs)
        content = response["choices"][0]["message"]["content"] or ""

    return (content or "").strip()


def call_gemini(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    timeout: float,
    prompt_id: str = "",
) -> str:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to call Gemini models.")

    resolved_model = resolve_gemini_model_name(model)
    selected_model = resolved_model
    headers, params = build_gemini_auth_headers(api_key)
    system_text = "\n".join(
        (m.get("content") or "").strip()
        for m in messages
        if (m.get("role") or "").strip().lower() == "system"
        and (m.get("content") or "").strip()
    ).strip()

    # Hard guidance to reduce raw URL/link dumping in the answer body.
    anti_link_dump = (
        "Do not output raw URLs in the answer body. "
        "Only include links as inline citations and/or in the final Sources list when grounding is enabled."
    )
    if system_text:
        system_text = system_text + "\n" + anti_link_dump
    else:
        system_text = anti_link_dump

    base_payload: dict[str, object] = {
        "contents": build_gemini_contents(messages),
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_text:
        base_payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    def build_payload(
        include_tools: bool, use_fallback_tool: bool = False
    ) -> dict[str, object]:
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
    ) -> Tuple[
        Optional[requests.Response], Optional[Dict[str, Any]], Optional[Exception]
    ]:
        try:
            url = f"{GEMINI_API_BASE}/{model_id}:generateContent"
            resp = requests.post(
                url,
                headers=headers,
                params=params,
                json=payload_body,
                timeout=timeout,
            )
            resp.raise_for_status()
            parsed_body = None
            try:
                decoded = resp.content.decode("utf-8")
                parsed_body = json.loads(decoded)
            except Exception:
                try:
                    parsed_body = resp.json()
                except Exception:
                    parsed_body = None
            return resp, parsed_body, None
        except Exception as exc:
            try:
                return resp, None, exc  # type: ignore[name-defined]
            except UnboundLocalError:
                return None, None, exc

    def is_tool_error(exc: requests.HTTPError) -> bool:
        response_text = ""
        resp_obj = getattr(exc, "response", None)
        if resp_obj is not None:
            response_text = resp_obj.text or ""
        message = f"{response_text} {exc}".lower()
        # Gemini tool schema errors show up in a few common forms depending on API/version.
        tool_terms = (
            "tools",
            "toolsets",
            "google_search",
            "google_search_retrieval",
            "unknown field",
            "invalid json payload",
            "cannot find field",
        )
        return (
            "unknown name" in message
            or "unknown field" in message
            or "invalid json payload" in message
        ) and any(term in message for term in tool_terms)

    tools_supported = True
    last_tool_mode: str = "none"

    def attempt(
        model_id: str,
    ) -> Tuple[
        Optional[requests.Response], Optional[Dict[str, Any]], Optional[Exception]
    ]:
        nonlocal tools_supported
        nonlocal last_tool_mode
        last_tool_mode = "none"
        last_tool_mode = "google_search" if tools_supported else "none"
        resp, body, err = _post(model_id, build_payload(tools_supported))
        if err is None:
            return resp, body, err

        if (
            tools_supported
            and isinstance(err, requests.HTTPError)
            and is_tool_error(err)
        ):
            last_tool_mode = "google_search_retrieval"
            resp, body, err = _post(
                model_id, build_payload(True, use_fallback_tool=True)
            )
            if err is None:
                return resp, body, err

            tools_supported = False
            last_tool_mode = "none"
            resp, body, err = _post(model_id, build_payload(False))
        return resp, body, err

    resp, body, err = attempt(resolved_model)
    if err is not None and isinstance(err, requests.HTTPError):
        status = getattr(resp, "status_code", None)
        if status == 404 and resolved_model != GEMINI_FALLBACK_MODEL:
            selected_model = GEMINI_FALLBACK_MODEL
            resp, body, err = attempt(GEMINI_FALLBACK_MODEL)

    final_body = body or {}
    if err is None and body is not None:
        grounded = _gemini_response_used_search(final_body)
        used_search = (last_tool_mode != "none") and grounded
        if last_tool_mode != "none" and not grounded:
            print(
                f"[{prompt_id}] Gemini search used: no (mode={last_tool_mode}; tools_sent=yes; grounding_missing=yes)"
            )
        else:
            print(
                f"[{prompt_id}] Gemini search used: {'yes' if used_search else 'no'} (mode={last_tool_mode})"
            )
    else:
        print(f"[{prompt_id}] Gemini search used: no (mode={last_tool_mode})")

    if err is not None:
        raise err

    candidates = final_body.get("candidates") or []
    text = ""
    if candidates:
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if parts:
            text = parts[0].get("text", "") or ""

    formatted = (text or "").strip()
    if not formatted:
        raise RuntimeError("Gemini returned an empty response.")

    # Render grounding citations into the response text (Gemini-chat-like).
    formatted = _gemini_render_citations_in_text(formatted, final_body)
    return formatted


# --- Phase 1.5 follow-up helper functions ---
def _parse_int(value: str) -> Optional[int]:
    try:
        s = str(value).strip()
        if not s:
            return None
        # Allow values like "55" or "55.0"
        return int(float(s))
    except Exception:
        return None


# --- Age clamping helper ---

_AGE_REFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{1,3}\s*(?:years?|yrs?)\s*(?:[-–—]\s*)?old\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}\s*y\/o\b", re.IGNORECASE),
    re.compile(r"\bage\s*(?:is|:)?\s*\d{1,3}\b", re.IGNORECASE),
    re.compile(r"\bmy age is\s*\d{1,3}\b", re.IGNORECASE),
    re.compile(
        r"\bi(?:'m| am| was| was just| just turned| turned| will be| will turn| have been| 've been| 'll be)\s*\d{1,3}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:persona|person|user|consumer|customer)\s+(?:is|are)\s*\d{1,3}\b",
        re.IGNORECASE,
    ),
]


def _clamp_user_age_followup(text: str, persona_context: Mapping[str, str]) -> str:
    """Clamp any age mention in the text into the persona's minAge/maxAge range."""
    raw = (text or "").strip()
    if not raw:
        return raw

    min_age = _parse_int(persona_context.get("minAge", ""))
    max_age = _parse_int(persona_context.get("maxAge", ""))
    if min_age is None and max_age is None:
        return raw

    if min_age is not None and max_age is not None and max_age < min_age:
        min_age, max_age = max_age, min_age

    def clamp_value(value: int) -> int:
        if min_age is not None:
            value = max(min_age, value)
        if max_age is not None:
            value = min(max_age, value)
        return value

    match_found = False

    def _clamp_match(match: re.Match[str]) -> str:
        nonlocal match_found
        match_found = True
        snippet = match.group(0)
        digit_match = re.search(r"\d{1,3}", snippet)
        if not digit_match:
            return snippet
        age_token = digit_match.group(0)
        clamped = clamp_value(int(age_token))
        if str(clamped) == age_token:
            return snippet
        start, end = digit_match.span()
        return snippet[:start] + str(clamped) + snippet[end:]

    processed = raw
    for pattern in _AGE_REFERENCE_PATTERNS:
        processed = pattern.sub(_clamp_match, processed)

    if match_found:
        return processed

    if raw.strip().isdigit():
        return str(clamp_value(int(raw.strip())))

    return raw


def derive_user_followup_from_phase1(
    phase1_response: str, persona_context: Mapping[str, str]
) -> Optional[str]:
    """Return a natural user follow-up message when Phase 1 asked a direct question.

    This intentionally returns a *user-style answer* (e.g., "55"), not an instruction block.
    The model will receive conversation context via the `messages` list.
    """
    text = (phase1_response or "").strip().lower()
    if not text:
        return None

    # Heuristic: age question.
    # Covers: "what is your age", "how old are you", "your age?"
    if re.search(r"\bhow old\b", text) or re.search(r"\bage\b", text):
        min_age = _parse_int(persona_context.get("minAge", ""))
        max_age = _parse_int(persona_context.get("maxAge", ""))

        if min_age is not None and max_age is not None:
            if max_age < min_age:
                min_age, max_age = max_age, min_age
            age = int(round((min_age + max_age) / 2))
            # Clamp defensively
            age = max(min_age, min(max_age, age))
            return str(age)

        # If only one bound exists, pick that bound.
        if min_age is not None:
            return str(min_age)
        if max_age is not None:
            return str(max_age)

        # If persona doesn't specify, don't guess.
        return None

    return None


# --- Additional Phase 1.5 follow-up helpers ---


# Validator: ensure follow-up is a user message, not an assistant answer
def _looks_like_assistant_answer(text: str) -> bool:
    """Heuristic guard: p1c_prompt must be a USER follow-up, not an assistant answer."""
    t = (text or "").strip()
    if not t:
        return True
    lowered = t.lower()

    # Clear assistant-y openers
    if (
        lowered.startswith("here are")
        or lowered.startswith("sure")
        or lowered.startswith("of course")
    ):
        return True

    # Links/citations are a strong signal it's an assistant response
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return True

    # Markdown list formatting
    if "\n- " in t or "\n* " in t or "\n1." in t or "\n2." in t:
        return True

    # Bolded product list style
    if "**" in t and ("serum" in lowered or "best" in lowered or "top" in lowered):
        return True

    # Too long for a user follow-up
    if len(t) > 220:
        return True

    return False


# --- Begin: Stricter p1c_prompt validation helpers ---
def _is_short_direct_answer(text: str) -> bool:
    """Allow short user answers to explicit assistant questions (e.g., age)."""
    t = (text or "").strip()
    if not t:
        return False
    # Numeric-only (e.g., "40")
    if t.isdigit() and len(t) <= 3:
        return True
    # Common age phrasing (kept short)
    if (
        len(t) <= 40
        and re.search(r"\b\d{1,3}\b", t)
        and re.search(r"\b(age|years? old|y/o|yo|i'm|i am)\b", t.lower())
    ):
        return True
    return False


def _looks_like_information_giving(text: str) -> bool:
    """Heuristic: user follow-up should seek info, not provide an answer/breakdown."""
    t = (text or "").strip()
    if not t:
        return True
    lowered = t.lower()

    # Multi-paragraph / explanatory tone
    if "\n\n" in t:
        return True

    # Common assistant-y explanatory openers
    if lowered.startswith("it appears") or lowered.startswith("it looks like"):
        return True

    # Headings / breakdown structure
    if "here's a breakdown" in lowered or "breakdown:" in lowered:
        return True

    return False


def _is_valid_user_followup(text: str) -> bool:
    """Valid p1c_prompt must be a user information-seeking message.

    Rules:
    - Must NOT look like an assistant answer (links/lists/overly long).
    - Must be a question/request (contain '?') OR be a short direct answer (age).
    """
    t = (text or "").strip()
    if not t:
        return False

    if _looks_like_assistant_answer(t):
        return False

    if _looks_like_information_giving(t):
        return False

    # Prefer explicit question marks for info-seeking follow-ups
    if "?" in t:
        return True

    # Allow short direct answers (e.g., age) without a '?'
    if _is_short_direct_answer(t):
        return True

    return False


def sanitize_phase15_text(text: str) -> str:
    """Remove non-ASCII characters and collapse whitespace so prompts/responses stay plain."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.replace("–", "-").replace("—", "-").replace("…", "...")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.replace("\r", " ").replace("\n", " ")
    ascii_only = re.sub(r"\s+", " ", ascii_only)
    return ascii_only.strip()


def _parse_search_behaviors(value: str) -> list[str]:
    """Parse search_behaviors from audience_habbits.csv.

    It may look like a Python-ish list string with single quotes.
    Returns a best-effort list of strings.
    """
    raw = (value or "").strip()
    if not raw:
        return []
    # Try JSON first.
    try:
        if raw.startswith("[") and raw.endswith("]"):
            return [str(x).strip() for x in json.loads(raw) if str(x).strip()]
    except Exception:
        pass
    # Try to coerce single-quoted list into JSON.
    try:
        if raw.startswith("[") and raw.endswith("]"):
            coerced = raw.replace("'", '"')
            return [str(x).strip() for x in json.loads(coerced) if str(x).strip()]
    except Exception:
        pass
    # Fallback: comma-split
    parts = [p.strip().strip("[]'\"") for p in raw.split(",")]
    return [p for p in parts if p]


def derive_generic_user_followup(
    prompt_text: str,
    phase1_response: str,
    persona_context: Mapping[str, str],
    openai_api_key: Optional[str],
    prompt_id: str = "",
) -> str:
    """Generate a short, natural *user* follow-up message.

    This uses a separate OpenAI call so each follow-up is unique and properly grounded
    in the Phase 1 assistant message + persona constraints.

    If the OpenAI key is missing, falls back to a lightweight heuristic.
    """

    # --- fallback heuristic (no OpenAI key) ---
    def _fallback() -> str:
        # If the assistant claims it can't access real-time/local pricing, ask it to search the web.
        p1 = (phase1_response or "").strip().lower()
        if (
            "real-time" in p1
            and (
                "unable to access" in p1
                or "can't access" in p1
                or "cannot access" in p1
                or "don't have access" in p1
                or "do not have access" in p1
            )
        ) or (
            "local pricing" in p1
            and ("unable" in p1 or "cannot" in p1 or "can't" in p1)
        ):
            return "Can you search the web for current prices and tell me the typical price range, with 3–5 examples and the sources?"

        behaviors = _parse_search_behaviors(persona_context.get("search_behaviors", ""))
        behaviors_snippet = ""
        if behaviors:
            picked = behaviors[:2]
            behaviors_snippet = " I'm mostly trying to " + " and ".join(picked) + "."

        # Default: ask for concrete, actionable help (user follow-up request).
        return (
            "Can you help me narrow this down and tell me what to look for, plus 3–5 concrete options?"
            + behaviors_snippet
        )

    if not openai_api_key:
        return _fallback()

    # Build a compact persona block (only non-empty fields)
    persona_lines: list[str] = []
    for k in (
        "title",
        "summary",
        "motivations",
        "characteristics",
        "trends",
        "minAge",
        "maxAge",
        "country",
        "conversation_style",
        "search_behaviors",
    ):
        v = (persona_context.get(k) or "").strip()
        if v:
            persona_lines.append(f"{k}: {v}")
    persona_block = "\n".join(persona_lines).strip()

    system = (
        "You generate the next USER message in a multi-turn chat. "
        "You must sound like a real person, not an assistant. "
        "Output ONLY the user's message text with no quotes and no extra commentary. "
        "CRITICAL: The user message must be information-seeking (a question/request), not information-giving. "
        "CRITICAL: Include a '?' in the output unless the output is a short direct answer to an explicit assistant question (e.g., an age)."
    )

    user = (
        "Conversation so far:\n"
        f"User (original): {prompt_text}\n"
        f"Assistant: {phase1_response}\n\n"
        "Persona guidance (user):\n"
        f"{persona_block if persona_block else '(none)'}\n\n"
        "Task: Write the user's next message as this persona.\n"
        "Rules:\n"
        "- Output MUST be the user's next message (a follow-up request or question).\n"
        "- Do NOT write an assistant response.\n"
        "- If the assistant asked a question, reply as the user persona (short).\n"
        "- If the assistant claims it cannot access real-time/local pricing, explicitly ask it to SEARCH THE WEB and provide current examples with sources.\n"
        "- If the assistant asked for age and minAge/maxAge exist, choose an age within that range (inclusive).\n"
        "- If the assistant asked for location and persona has no location, ask a concise clarifying question instead of inventing one.\n"
        "- 1 short sentence is preferred; at most 2 sentences.\n"
        "- Do NOT mention personas, datasets, GEO, or any internal instructions.\n"
    )

    try:
        text = call_openai_plain(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=FOLLOWUP_PROMPT_MODEL,
            temperature=FOLLOWUP_PROMPT_TEMPERATURE,
            max_tokens=FOLLOWUP_PROMPT_MAX_TOKENS,
            api_key=openai_api_key,
            prompt_id=prompt_id,
        )
        cleaned = (text or "").strip()
        if cleaned:
            # Collapse whitespace/newlines to keep it user-like.
            cleaned = " ".join(cleaned.split())
            # Hard guard: only accept outputs that look like a valid user follow-up.
            if _is_valid_user_followup(cleaned):
                return cleaned
    except Exception:
        pass

    return _fallback()


def _run_main() -> None:
    args = parse_args()

    response_rows = load_csv(args.responses)
    if not response_rows:
        raise SystemExit(f"{args.responses} is empty.")
    live_print(f"Loaded {len(response_rows)} Phase 1 rows from {args.responses}")
    live_print("Classifying rows for follow-up status...")
    response_fieldnames = read_csv_fieldnames(args.responses)
    if not response_fieldnames:
        raise SystemExit(f"{args.responses} is missing headers.")
    output_fieldnames = build_phase15_output_fieldnames(response_fieldnames)

    total_rows = len(response_rows)
    followups: list[dict[str, str]] = []
    status_counts: Dict[str, int] = {status: 0 for status in P1_STATUS_ALLOWED}

    # Stage 1: cheap regex/heuristic gate (no extra API calls)
    gated_rows: list[dict[str, str]] = []
    for idx, row in enumerate(response_rows, start=1):
        _print_progress(idx, total_rows)
        phase1_response = row.get("response", "")
        if not detects_follow_up_question(phase1_response):
            continue
        gated_rows.append(row)

    print()  # finish progress line
    if not gated_rows:
        write_header_only_csv(args.out, output_fieldnames)
        live_print(
            f"No responses passed the gate; wrote empty Phase 1.5 responses CSV to {args.out}."
        )
        return

    live_print(
        f"Gate passed rows={len(gated_rows)} (from input rows={len(response_rows)}). Running classifier on gated rows..."
    )

    # Stage 2: model-based classifier (only on gated rows)
    total_gated = len(gated_rows)
    for idx, row in enumerate(gated_rows, start=1):
        _print_progress(idx, total_gated)
        prompt_id = (row.get("prompt_id") or "").strip()
        prompt_text = row.get("prompt", "")
        phase1_response = row.get("response", "")
        phase1_model = (row.get("model") or "").strip() or args.model

        model_api_key = get_model_api_key(phase1_model, args)
        if not model_api_key:
            raise SystemExit(
                f"Missing API key for model '{phase1_model}'. "
                f"Set {GEMINI_API_KEY_ENV if is_gemini_model(phase1_model) else DEFAULT_OPENAI_API_KEY_ENV}."
            )

        try:
            status = classify_p1_status(
                prompt_text=prompt_text,
                phase1_response=phase1_response,
                model=phase1_model,
                api_key=model_api_key,
                timeout=args.timeout,
                prompt_id=prompt_id,
            )
        except Exception as exc:
            status = P1_STATUS_NEEDS
            print(
                f"[{prompt_id}] Classifier error -> default NEEDS_CLARIFICATION: {exc}"
            )

        status_counts[status] = status_counts.get(status, 0) + 1

        # Only keep rows that truly need follow-up.
        if status in {P1_STATUS_NEEDS, P1_STATUS_FAILED}:
            annotated = dict(row)
            annotated["_p1_status"] = status
            followups.append(annotated)

    print()  # finish progress line
    if not followups:
        write_header_only_csv(args.out, output_fieldnames)
        live_print(
            f"No follow-ups detected; wrote empty Phase 1.5 responses CSV to {args.out}."
        )
        return

    live_print(
        f"Input rows={len(response_rows)}; followups kept after classification={len(followups)}."
    )
    live_print("Phase 1 status counts:")
    for status, count in status_counts.items():
        if count:
            live_print(f"  {status}: {count}")

    responses_map: Dict[tuple[str, str], dict[str, str]] = {}
    for row in response_rows:
        pid = (row.get("prompt_id") or "").strip()
        mdl = (row.get("model") or "").strip() or "unspecified"
        if pid:
            responses_map[make_response_key(pid, mdl)] = row
    persona_map = build_persona_map(load_csv(args.audience) if args.audience.is_file() else [])
    pmg_clients = load_pmg_clients(args.pmg_client_json)
    pmg_by_id, pmg_by_name = build_pmg_maps(pmg_clients)

    total_followups = len(followups)
    model_counts: Dict[str, int] = {}
    for row in followups:
        model_key = (row.get("model") or "").strip() or "unspecified"
        model_counts[model_key] = model_counts.get(model_key, 0) + 1
    live_print(f"Phase 1.5 follow-up count: {total_followups}")
    for model_name, count in model_counts.items():
        live_print(f"  {model_name}: {count}")

    output_rows: list[dict[str, str]] = []
    followup_map: Dict[tuple[str, str], dict[str, str]] = {}

    for idx, row in enumerate(followups, start=1):
        prompt_id = row.get("prompt_id", "")
        prompt_text = row.get("prompt", "")
        phase1_response = row.get("response", "")
        short_q = (prompt_text or "").strip()[:80]
        live_print(
            f"[{idx}/{total_followups}] Preparing follow-up for prompt_id={row.get('prompt_id', '')} "
            f"brand={row.get('brand', '')} | ques={short_q!r}"
        )

        # Determine which model originally answered Phase 1 for this row.
        # prompt_id can map to multiple Phase 1 rows, so use (prompt_id, model).
        phase1_model = (row.get("model") or "").strip() or args.model
        original_response = responses_map.get(
            make_response_key(prompt_id, phase1_model), {}
        )

        model_api_key = get_model_api_key(phase1_model, args)
        if not model_api_key:
            raise SystemExit(
                f"Missing API key for model '{phase1_model}'. "
                f"Set {GEMINI_API_KEY_ENV if is_gemini_model(phase1_model) else DEFAULT_OPENAI_API_KEY_ENV}."
            )
        status = row.get("_p1_status", P1_STATUS_NEEDS)
        live_print(f"[{prompt_id}] p1_status={status}")
        brand = row.get("brand", "") or original_response.get("brand", "")
        brand_id = row.get("brand_id", "") or original_response.get("brand_id", "")
        category = row.get("category", "") or original_response.get("category", "")
        phase = "1.5"
        type_ = row.get("type", "") or original_response.get("type", "")
        sources = row.get("sources", "")
        pmg_entry = None
        if brand_id and brand_id in pmg_by_id:
            pmg_entry = pmg_by_id[brand_id]
        elif brand:
            pmg_entry = pmg_by_name.get(brand.lower())
        if pmg_entry:
            brand = pmg_entry.get("name", brand)
            brand_id = pmg_entry.get("id", brand_id)

        persona = find_persona(persona_map, prompt_id)
        if persona is None:
            hex_tokens = re.findall(r"[0-9a-fA-F]{24}", (prompt_id or ""))
            if hex_tokens:
                live_print(
                    f"[{prompt_id}] WARNING: persona_id token(s) found in prompt_id but not present in persona_map: {hex_tokens}"
                )
            else:
                live_print(
                    f"[{prompt_id}] WARNING: No persona found; age/location constraints may not apply."
                )

        persona_context = persona_context_dict(persona)

        # Build an in-character continuation prompt (no JSON/state framing).
        persona_lines: list[str] = []
        if persona_context.get("title"):
            persona_lines.append(f"Title: {persona_context.get('title')}")
        if persona_context.get("summary"):
            persona_lines.append(f"Summary: {persona_context.get('summary')}")
        if persona_context.get("motivations"):
            persona_lines.append(f"Motivations: {persona_context.get('motivations')}")
        if persona_context.get("characteristics"):
            persona_lines.append(
                f"Characteristics: {persona_context.get('characteristics')}"
            )
        if persona_context.get("trends"):
            persona_lines.append(f"Trends: {persona_context.get('trends')}")

        age_bits: list[str] = []
        if persona_context.get("minAge"):
            age_bits.append(f"minAge={persona_context.get('minAge')}")
        if persona_context.get("maxAge"):
            age_bits.append(f"maxAge={persona_context.get('maxAge')}")
        if age_bits:
            persona_lines.append("Age range: " + ", ".join(age_bits))

        if persona_context.get("country"):
            persona_lines.append(f"Country: {persona_context.get('country')}")
        if persona_context.get("conversation_style"):
            persona_lines.append(
                f"Conversation style: {persona_context.get('conversation_style')}"
            )
        if persona_context.get("search_behaviors"):
            persona_lines.append(
                f"Search behaviors: {persona_context.get('search_behaviors')}"
            )
        if persona_context.get("preferred_models"):
            persona_lines.append(
                f"Preferred models: {persona_context.get('preferred_models')}"
            )

        persona_block = "\n".join(persona_lines).strip()

        # Phase 1.5 user follow-up: keep it natural and rely on the conversation turns for context.
        direct_user_followup = derive_user_followup_from_phase1(
            phase1_response, persona_context
        )
        if direct_user_followup:
            follow_up_instructions = direct_user_followup
        else:
            follow_up_instructions = derive_generic_user_followup(
                prompt_text=prompt_text,
                phase1_response=phase1_response,
                persona_context=persona_context,
                openai_api_key=(
                    args.openai_api_key or os.environ.get(DEFAULT_OPENAI_API_KEY_ENV)
                ),
                prompt_id=prompt_id,
            )

        # Enforce persona age bounds if the follow-up includes an age.
        follow_up_instructions = _clamp_user_age_followup(
            (follow_up_instructions or "").strip(), persona_context
        )

        # Hard guard: p1c_prompt must be a USER information-seeking message (not an assistant answer).
        follow_up_instructions = (follow_up_instructions or "").strip()
        if not _is_valid_user_followup(follow_up_instructions):
            # Fall back to a safe persona-style question that seeks guidance.
            follow_up_instructions = "Can you ask me what you need to know and then recommend the best option for someone like me?"

        follow_up_prompt = sanitize_phase15_text(follow_up_instructions)
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": phase1_response},
            {"role": "user", "content": follow_up_instructions},
        ]

        try:
            print(
                f"[{idx}/{total_followups}] Sending Phase 1.5 prompt to {phase1_model}..."
            )
            if is_gemini_model(phase1_model):
                follow_up_response = call_gemini(
                    messages=messages,
                    model=phase1_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    api_key=model_api_key,
                    timeout=args.timeout,
                )
            else:
                follow_up_response = call_openai(
                    messages=messages,
                    model=phase1_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    api_key=model_api_key,
                    prompt_id=prompt_id,
                )
            live_print(
                f"[{idx}/{total_followups}] Received Phase 1.5 response (length={len(follow_up_response)})."
            )
        except Exception as exc:  # pragma: no cover
            follow_up_response = f"[error: {exc}]"
            live_print(f"[{idx}/{total_followups}] Error during Phase 1.5 call: {exc}")

        followup_map[make_response_key(prompt_id, phase1_model)] = {
            "p1c_prompt": follow_up_prompt,
            "p1c_response": sanitize_phase15_text(follow_up_response),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in response_rows:
        user_row = {key: row.get(key, "") for key in response_fieldnames}
        row_model = (row.get("model") or "").strip() or args.model
        key = make_response_key(row.get("prompt_id", ""), row_model)
        followup_entry = followup_map.get(key)
        user_row["p1c_prompt"] = followup_entry["p1c_prompt"] if followup_entry else ""
        user_row["p1c_response"] = (
            followup_entry["p1c_response"] if followup_entry else ""
        )
        output_rows.append(user_row)
    with args.out.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    live_print(f"Wrote Phase 1.5 responses to {args.out} ({len(output_rows)} rows).")


def main() -> None:
    _run_main()


if __name__ == "__main__":
    main()
