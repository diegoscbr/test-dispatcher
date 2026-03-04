#!/usr/bin/env python3
"""Generate Phase 2 follow-up responses that continue Phase 1 and Phase 1.5 conversations."""

from __future__ import annotations

from copy import deepcopy
import argparse
import csv
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import concurrent.futures
import random
import threading
import time
import re


import requests
import openai

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
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_BASE = os.environ.get("GEMINI_API_BASE", DEFAULT_GEMINI_API_BASE)

GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
# Latest ChatGPT snapshot alias (kept in sync with Phase 1/Phase 1.5)
LATEST_CHATGPT_MODEL = "gpt-5.2-chat-latest"

DEFAULT_PHASE15_FIELDNAMES = [
    "prompt_id",
    "brand_id",
    "brand",
    "category",
    "phase",
    "model",
    "prompt",
    "response",
    "p1_prompt",
    "p1_response",
    "p1c_prompt",
    "p1c_response",
    "persona_id",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PHASE15_RESPONSES_PATH = SCRIPT_DIR / "Phase15_Responses.csv"
DEFAULT_PHASE1_RESPONSES_PATH = SCRIPT_DIR / "responses.csv"
DEFAULT_AUDIENCE_CSV = SCRIPT_DIR / "audience_habbits.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "phase2_responses.csv"

GEMINI_TOOLS_PRIMARY = [{"google_search": {}}]
GEMINI_TOOLS_FALLBACK = [{"google_search_retrieval": {}}]

FOLLOWUP_PROMPT_MODEL = os.environ.get("FOLLOWUP_PROMPT_MODEL", LATEST_CHATGPT_MODEL)
FOLLOWUP_PROMPT_TEMPERATURE = float(os.environ.get("FOLLOWUP_PROMPT_TEMPERATURE", "0.2"))
FOLLOWUP_PROMPT_MAX_TOKENS = int(os.environ.get("FOLLOWUP_PROMPT_MAX_TOKENS", "120"))

OPENAI_WEB_SEARCH_TOOL = {"type": "web_search"}


def load_env_file(path: Optional[str] = None) -> None:
    """Load key=value pairs from a .env file into os.environ."""
    env_path = Path(path or SCRIPT_DIR / ".env")
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


# --- Provider concurrency/throttle/retry helpers ---

class RateLimiter:
    """Simple client-side rate limiter (requests per minute)."""

    def __init__(self, rpm: float) -> None:
        self.rpm = float(rpm or 0.0)
        self._lock = threading.Lock()
        self._next_allowed = 0.0  # monotonic seconds

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        interval = 60.0 / max(self.rpm, 1e-9)
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._next_allowed - now
                if wait <= 0:
                    self._next_allowed = now + interval
                    return
            time.sleep(min(wait, 0.5))


class ProviderController:
    """Controls concurrency + throttling per provider and supports pullback via backoff."""

    def __init__(self, *, name: str, max_inflight: int, rpm: float) -> None:
        self.name = name
        self.sema = threading.BoundedSemaphore(max(1, int(max_inflight)))
        self.limiter = RateLimiter(rpm)

    def __enter__(self):
        self.sema.acquire()
        self.limiter.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.sema.release()
        return False


def _status_code_from_exc(exc: Exception) -> Optional[int]:
    # OpenAI new SDK errors often have .status_code; requests has response.status_code.
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        return sc
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return int(getattr(resp, "status_code", None) or 0) or None
        except Exception:
            return None
    return None


def _is_retryable_status(status: Optional[int]) -> bool:
    return status in {429, 500, 502, 503, 504}


def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    # Exponential backoff with jitter
    delay = min(cap, base * (2 ** max(0, attempt)))
    delay = delay + random.uniform(0.0, 1.0)
    time.sleep(delay)


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
        fallback = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
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
    """Detect whether the Responses API output contains a web_search_call."""
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
    """Best-effort detection of Google Search grounding in Gemini responses."""
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



_GROUNDING_REDIRECT_CACHE: Dict[str, str] = {}
_GROUNDING_REDIRECT_CACHE_LOCK = Lock()
_VERTEX_REDIRECT_HOST = "vertexaisearch.cloud.google.com/grounding-api-redirect"
_SOURCE_PLACEHOLDER_TEMPLATE = "<<AI_SRC_{number}>>"
_SOURCE_PLACEHOLDER_PATTERN = re.compile(r"<<AI_SRC_(\d+)>>")
_MAX_SOURCE_ENTRIES = 10


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
            resp = resp_method(target, allow_redirects=True, timeout=timeout, headers=headers)
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


def _gemini_render_citations_in_text(text: str, body: Optional[Dict[str, Any]]) -> str:
    """Insert grounding sources inline where Gemini signals them."""
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
    if not isinstance(chunks, list) or not chunks or not isinstance(supports, list):
        return base_text

    chunk_info_by_index: Dict[int, Dict[str, Any]] = {}
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
    used_info: Dict[int, Dict[str, Any]] = {}

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
        gci = sup.get("groundingChunkIndices") or sup.get("grounding_chunk_indices") or []
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
        url = info.get("resolved_url") or info.get("uri") or ""
        if not url:
            return match.group(0)
        return f"({url})"

    rendered_with_urls = _SOURCE_PLACEHOLDER_PATTERN.sub(_replace_marker, rendered)

    unique_entries: list[Tuple[int, str, str]] = []
    seen_urls: set[str] = set()
    for number in sorted(used_info):
        info = used_info[number]
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
        rendered_with_urls = f"{rendered_with_urls}\n\n" + "\n".join(sources_lines)

    return rendered_with_urls



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 follow-up prompts for Phase 1 and Phase 1.5 threads."
    )
    parser.add_argument(
        "--phase15-responses",
        type=Path,
        default=DEFAULT_PHASE15_RESPONSES_PATH,
        help="Phase 1.5 follow-up responses CSV (same folder as the script).",
    )
    parser.add_argument(
        "--phase1-responses",
        type=Path,
        default=DEFAULT_PHASE1_RESPONSES_PATH,
        help="Phase 1 responses CSV fallback when Phase 1.5 output is empty (same folder as the script).",
    )
    parser.add_argument(
        "--audience",
        type=Path,
        default=DEFAULT_AUDIENCE_CSV,
        help="Persona guidance CSV for follow-up tone (same folder as the script).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV for Phase 2 responses (same folder as the script).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LATEST_CHATGPT_MODEL,
        help="Fallback model if a Phase 1 model record is missing.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for the Phase 2 assistant call.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Max tokens for the Phase 2 assistant call.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are an expert GEO analyst who always answers in clear, plain English.",
        help="System prompt used when calling the Phase 2 assistant.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Override the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Override the GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for Gemini requests.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=24,
        help="Max concurrent in-flight requests across all items.",
    )
    parser.add_argument(
        "--openai-rpm",
        type=float,
        default=60.0,
        help="Client-side throttle for OpenAI requests per minute (0 to disable).",
    )
    parser.add_argument(
        "--gemini-rpm",
        type=float,
        default=60.0,
        help="Client-side throttle for Gemini requests per minute (0 to disable).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Max retries on 429/5xx with exponential backoff.",
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


def build_gemini_auth_headers(api_key: Optional[str]) -> tuple[Dict[str, str], Dict[str, str]]:
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
        return args.gemini_api_key or os.environ.get(GEMINI_API_KEY_ENV)
    return args.openai_api_key or os.environ.get(DEFAULT_OPENAI_API_KEY_ENV)


def build_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, object]]:
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


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]


def read_csv_fieldnames(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [name for name in (reader.fieldnames or []) if name]


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


def find_persona(persona_map: Mapping[str, dict[str, str]], prompt_id: str) -> Optional[dict[str, str]]:
    if not prompt_id:
        return None
    suffix = prompt_id.split("_")[-1]
    if not suffix:
        return None
    if suffix in persona_map:
        return persona_map[suffix]
    for pid, persona in persona_map.items():
        if pid.startswith(suffix):
            return persona
    return None


def persona_context_dict(persona: Optional[Mapping[str, str]]) -> dict[str, str]:
    context: dict[str, str] = {}
    if not persona:
        return context
    for key, _label in PERSONA_FIELDS:
        context[key] = persona.get(key, "")
    return context


def persona_context_text(persona: Optional[Mapping[str, str]]) -> str:
    if not persona:
        return ""
    lines: list[str] = []
    for key, label in PERSONA_FIELDS:
        value = persona.get(key, "")
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _parse_search_behaviors(value: str) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    try:
        if raw.startswith("[") and raw.endswith("]"):
            return [str(x).strip() for x in json.loads(raw) if str(x).strip()]
    except Exception:
        pass
    try:
        if raw.startswith("[") and raw.endswith("]"):
            coerced = raw.replace("'", '"')
            return [str(x).strip() for x in json.loads(coerced) if str(x).strip()]
    except Exception:
        pass
    parts = [p.strip().strip("[]'\"") for p in raw.split(",")]
    return [p for p in parts if p]


def build_conversation_text(history: Sequence[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, text in history:
        safe_text = (text or "").strip()
        if not safe_text:
            continue
        lines.append(f"{role}: {safe_text}")
    return "\n".join(lines)


def get_first_value(row: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def has_phase15_row(row: Mapping[str, str]) -> bool:
    return bool(
        get_first_value(
            row,
            "p1c_prompt",
            "p1c_response",
            "follow_up_prompt",
            "phase15_response",
        )
    )


def _fallback_user_followup(history: Sequence[tuple[str, str]], persona_context: Mapping[str, str]) -> str:
    behaviors = _parse_search_behaviors(persona_context.get("search_behaviors", ""))
    behavior_snippet = ""
    if behaviors:
        picked = behaviors[:2]
        behavior_snippet = " I'm mostly trying to " + " and ".join(picked) + "."
    last_assistant = ""
    for role, text in reversed(history):
        if role.lower() == "assistant" and text:
            last_assistant = text.strip()
            break
    if last_assistant.endswith("?") or "clarify" in last_assistant.lower():
        return "Can you clarify that a bit more?" + behavior_snippet
    return "Could you share a few specific examples or next steps?" + behavior_snippet


def _looks_like_question(text: str) -> bool:
    candidate = (text or "").strip()
    if not candidate:
        return False
    sanitized = re.sub(r"https?://\S+", "", candidate)
    if "?" in sanitized:
        return True
    return bool(re.match(r"^(who|what|when|where|why|how)\b", sanitized, re.IGNORECASE))


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", (text or ""))


def categorize_p2_intent(text: str) -> str:
    snapshot = _strip_urls(text).lower()
    if any(keyword in snapshot for keyword in ["clarify", "clarification", "clear up", "confused"]):
        return "clarification"
    if any(keyword in snapshot for keyword in ["more detail", "deeper", "expand", "dig", "deepen"]):
        return "deep_dive"
    if any(keyword in snapshot for keyword in ["alternative", "other option", "compare", "besides", "instead"]):
        return "alternatives"
    if _looks_like_question(snapshot):
        return "question"
    return "statement"


def ensure_alternative_prompt(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "Can you share a few alternative options or approaches?"
    if categorize_p2_intent(cleaned) == "alternatives":
        return cleaned
    return f"Can you share a few alternative options or approaches? {cleaned}"


def derive_user_followup_message(
    history: Sequence[tuple[str, str]],
    persona_context: Mapping[str, str],
    openai_api_key: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    prompt_id: str = "",
) -> str:
    if not openai_api_key:
        return _fallback_user_followup(history, persona_context)
    persona_lines: list[str] = []
    for key in (
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
        value = (persona_context.get(key) or "").strip()
        if value:
            persona_lines.append(f"{key}: {value}")
    persona_block = "\n".join(persona_lines).strip() or "(none)"

    system = (
        "You generate the next USER message in a multi-turn chat. "
        "Sound like a real person with no assistant framing."
    )

    conversation = build_conversation_text(history) or "(none)"
    user = (
        "Conversation so far:\n"
        f"{conversation}\n\n"
        "Persona guidance (user):\n"
        f"{persona_block}\n\n"
        "Task: Write the user's next reply as this persona.\n"
        "Rules:\n"
        "- Keep it natural and concise (1-2 sentences).\n"
        "- Do not mention personas, GEO, or analysis tooling.\n"
        "- Focus on asking a clarifying question or requesting a deeper explanation about the topic.\n"
        "- Avoid reacting to the assistant (no praise/disagreement); output only the user's message text."
    )

    try:
        text = call_openai(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_api_key,
            prompt_id=prompt_id,
        )
        cleaned = (text or "").strip()
        if cleaned:
            normalized = " ".join(cleaned.split())
            if _looks_like_question(normalized):
                return normalized
    except Exception:
        pass
    return _fallback_user_followup(history, persona_context)



def _call_openai_once(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    prompt_id: str = "",
) -> str:
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        if hasattr(client, "responses"):
            create_kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                # Auto mode: make web search available but do NOT force it.
                "tools": [OPENAI_WEB_SEARCH_TOOL],
                "include": ["web_search_call.action.sources"],
                "max_output_tokens": max_tokens,
            }
            # GPT-5 / GPT-5.2 family does not support temperature; only include it for other models.
            if temperature is not None and not (model or "").lower().startswith("gpt-5"):
                create_kwargs["temperature"] = temperature

            response = client.responses.create(**create_kwargs)
            used_search = _response_used_web_search(response)
            print(f"[{prompt_id}] OpenAI web_search used: {'yes' if used_search else 'no'}")
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


def call_openai(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    prompt_id: str = "",
    controller: Optional[ProviderController] = None,
    max_retries: int = 0,
) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Phase 2 calls.")

    last_exc: Optional[Exception] = None
    retries = int(max_retries or 0)

    for attempt in range(retries + 1):
        try:
            ctx = controller if controller is not None else None
            if ctx is None:
                # no throttling / concurrency control
                return _call_openai_once(messages, model, temperature, max_tokens, api_key, prompt_id)
            with ctx:
                return _call_openai_once(messages, model, temperature, max_tokens, api_key, prompt_id)
        except Exception as exc:
            last_exc = exc
            status = _status_code_from_exc(exc)
            msg = str(exc).lower()
            retryable = _is_retryable_status(status) or ("rate" in msg and "limit" in msg) or ("429" in msg)
            if attempt >= retries or not retryable:
                break
            print(f"[{prompt_id}] OpenAI retry {attempt + 1}/{retries} after status={status or 'unknown'}")
            _sleep_backoff(attempt)

    raise last_exc or RuntimeError("OpenAI call failed")



def call_gemini(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    timeout: float,
    prompt_id: str = "",
    controller: Optional[ProviderController] = None,
    max_retries: int = 0,
) -> str:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to call Gemini models.")

    def _call_gemini_once() -> str:
        resolved_model = resolve_gemini_model_name(model)
        headers, params = build_gemini_auth_headers(api_key)
        system_text = "\n".join(
            (m.get("content") or "").strip()
            for m in messages
            if (m.get("role") or "").strip().lower() == "system" and (m.get("content") or "").strip()
        ).strip()
        base_payload: dict[str, object] = {
            "contents": build_gemini_contents(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            base_payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        def build_payload(include_tools: bool, use_fallback_tool: bool = False) -> dict[str, object]:
            payload_copy = deepcopy(base_payload)
            if include_tools:
                payload_copy["tools"] = (
                    [dict(t) for t in GEMINI_TOOLS_FALLBACK]
                    if use_fallback_tool
                    else [dict(t) for t in GEMINI_TOOLS_PRIMARY]
                )
            return payload_copy

        def _post(model_id: str, payload_body: dict[str, object]) -> Tuple[Optional[requests.Response], Optional[Dict[str, Any]], Optional[Exception]]:
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
                return resp, resp.json(), None
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
            return (
                "unknown name" in message
                and (
                    "tools" in message
                    or "toolsets" in message
                    or "google_search" in message
                    or "google_search_retrieval" in message
                )
            )

        tools_supported = True
        last_tool_mode: str = "none"

        def attempt(model_id: str) -> Tuple[Optional[requests.Response], Optional[Dict[str, Any]], Optional[Exception]]:
            nonlocal tools_supported
            nonlocal last_tool_mode
            last_tool_mode = "none"
            last_tool_mode = "google_search" if tools_supported else "none"
            resp, body, err = _post(model_id, build_payload(tools_supported, use_fallback_tool=False))
            if err is None:
                return resp, body, err

            if tools_supported and isinstance(err, requests.HTTPError) and is_tool_error(err):
                last_tool_mode = "google_search_retrieval"
                resp, body, err = _post(model_id, build_payload(True, use_fallback_tool=True))
                if err is None:
                    return resp, body, err

                tools_supported = False
                last_tool_mode = "none"
                resp, body, err = _post(model_id, build_payload(False))
            return resp, body, err

        selected_model_id = resolved_model
        resp, body, err = attempt(resolved_model)
        if err is not None and isinstance(err, requests.HTTPError):
            status = getattr(resp, "status_code", None)
            if status == 404 and resolved_model != GEMINI_FALLBACK_MODEL:
                selected_model_id = GEMINI_FALLBACK_MODEL
                resp, body, err = attempt(GEMINI_FALLBACK_MODEL)

        final_body = body or {}
        if err is None and body is not None:
            used_search = (last_tool_mode != "none") and _gemini_response_used_search(final_body)
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
            content = (candidates[0].get("content") or {})
            parts = content.get("parts") or []
            if parts:
                text = parts[0].get("text", "") or ""

        formatted = text.strip()
        if not formatted:
            raise RuntimeError("Gemini returned an empty response.")
        formatted_response = _gemini_render_citations_in_text(formatted, final_body)
        return formatted_response

    last_exc: Optional[Exception] = None
    retries = int(max_retries or 0)

    for attempt_idx in range(retries + 1):
        try:
            if controller is None:
                return _call_gemini_once()
            with controller:
                return _call_gemini_once()
        except Exception as exc:
            last_exc = exc
            status = _status_code_from_exc(exc)
            msg = str(exc).lower()
            retryable = _is_retryable_status(status) or ("rate" in msg and "limit" in msg) or ("429" in msg)
            if attempt_idx >= retries or not retryable:
                break
            print(f"[{prompt_id}] Gemini retry {attempt_idx + 1}/{retries} after status={status or 'unknown'}")
            _sleep_backoff(attempt_idx)

    raise last_exc or RuntimeError("Gemini call failed")


def _run_main() -> None:
    args = parse_args()
    # Primary input is the Phase 1.5 output (which also carries Phase 1 rows).
    # However, Phase 1.5 may legitimately output 0 rows. In that case, fall back to the
    # Phase 1 responses CSV so Phase 2 can still run.
    phase15_rows = load_csv(args.phase15_responses)
    phase15_fieldnames = read_csv_fieldnames(args.phase15_responses)

    using_fallback_phase1 = False

    # If Phase 1.5 file is empty or missing headers, try Phase 1 fallback.
    if not phase15_fieldnames:
        # If the file exists but is empty, do not hard-fail; use Phase 1 fallback.
        if args.phase15_responses.exists() and args.phase15_responses.stat().st_size == 0:
            print(
                f"{args.phase15_responses} is empty (no headers). Will attempt Phase 1 fallback: {args.phase1_responses}"  # noqa: E501
            )
            using_fallback_phase1 = True
        else:
            # Missing headers is only fatal if we also cannot fall back.
            print(
                f"{args.phase15_responses} is missing headers. Will attempt Phase 1 fallback: {args.phase1_responses}"  # noqa: E501
            )
            using_fallback_phase1 = True

    # If Phase 1.5 has headers but no rows, also fall back to Phase 1.
    if phase15_fieldnames and not phase15_rows:
        if args.phase15_responses.exists():
            print(
                f"{args.phase15_responses} contains no rows. Will attempt Phase 1 fallback: {args.phase1_responses}"  # noqa: E501
            )
        else:
            print(
                f"{args.phase15_responses} not found. Will attempt Phase 1 fallback: {args.phase1_responses}"  # noqa: E501
            )
        using_fallback_phase1 = True

    if using_fallback_phase1:
        phase1_rows = load_csv(args.phase1_responses)
        phase1_fieldnames = read_csv_fieldnames(args.phase1_responses)
        if not phase1_fieldnames:
            if args.phase1_responses.exists() and args.phase1_responses.stat().st_size == 0:
                # Completely empty Phase 1 file; continue with an empty worklist using the default schema.
                phase15_fieldnames = DEFAULT_PHASE15_FIELDNAMES
                phase15_rows = []
                print(
                    f"{args.phase1_responses} is empty (no headers). Using default Phase 1 schema and continuing with an empty worklist."  # noqa: E501
                )
            else:
                # If Phase 1 fallback exists but is malformed, surface a clear error.
                raise SystemExit(
                    f"Neither {args.phase15_responses} nor {args.phase1_responses} has readable headers."
                )
        else:
            phase15_fieldnames = phase1_fieldnames
            phase15_rows = phase1_rows
            print(
                f"Using Phase 1 fallback input: {args.phase1_responses} ({len(phase15_rows)} rows)"
            )

    # If still empty after fallback attempts, proceed with an empty worklist.
    if not phase15_rows:
        print("No Phase 1/1.5 rows found; continuing with an empty worklist.")
    persona_map = build_persona_map(load_csv(args.audience))

    phase1_candidates: list[dict[str, str]] = []
    phase15_followups: list[dict[str, str]] = []
    for row in phase15_rows:
        if has_phase15_row(row):
            phase15_followups.append(row)
        else:
            phase1_candidates.append(row)

    total = len(phase1_candidates) + len(phase15_followups)
    print(
        f"Phase 2 worklist: {total} records "
        f"({len(phase1_candidates)} from Phase 1, {len(phase15_followups)} from Phase 1.5)"
    )

    openai_followup_key = args.openai_api_key or os.environ.get(DEFAULT_OPENAI_API_KEY_ENV)

    openai_controller = ProviderController(
        name="openai",
        max_inflight=max(1, int(args.max_workers)),
        rpm=float(args.openai_rpm or 0.0),
    )
    gemini_controller = ProviderController(
        name="gemini",
        max_inflight=max(1, int(args.max_workers)),
        rpm=float(args.gemini_rpm or 0.0),
    )

    combined_items = phase1_candidates + phase15_followups
    output_rows: list[Optional[dict[str, str]]] = [None] * len(combined_items)

    def process_one(idx: int, item: dict[str, str]) -> tuple[int, dict[str, str]]:
        is_phase15 = has_phase15_row(item)
        if is_phase15:
            label = f"Phase1.5 prompt_id={item.get('prompt_id', '')}"
        else:
            label = f"Phase1 prompt_id={item.get('prompt_id', '')}"
        print(f"[{idx}/{total}] Processing {label}")

        prompt_id = item.get("prompt_id", "")
        persona = None
        if is_phase15:
            persona_id = item.get("persona_id", "")
            if persona_id:
                persona = persona_map.get(persona_id)
        else:
            persona = find_persona(persona_map, prompt_id)
        persona_context = persona_context_dict(persona)

        phase1_prompt = get_first_value(item, "p1_prompt", "prompt")
        phase1_response = get_first_value(item, "p1_response", "response")

        follow_up_prompt = ""
        phase15_response = ""
        if is_phase15:
            follow_up_prompt = get_first_value(item, "p1c_prompt", "follow_up_prompt")
            phase15_response = get_first_value(item, "p1c_response", "phase15_response")

        base_assistant_response = phase1_response or phase15_response
        prompt_history: list[tuple[str, str]] = []
        if phase1_prompt:
            prompt_history.append(("User", phase1_prompt))
        if base_assistant_response:
            prompt_history.append(("Assistant", base_assistant_response))
        if is_phase15 and follow_up_prompt:
            prompt_history.append(("User", follow_up_prompt))

        follow_up_prompt_text = derive_user_followup_message(
            prompt_history,
            persona_context,
            openai_followup_key,
            FOLLOWUP_PROMPT_MODEL,
            FOLLOWUP_PROMPT_TEMPERATURE,
            FOLLOWUP_PROMPT_MAX_TOKENS,
            prompt_id=prompt_id,
        )
        if (idx - 1) % 2 == 0:
            follow_up_prompt_text = ensure_alternative_prompt(follow_up_prompt_text)

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": phase1_prompt},
            {"role": "assistant", "content": phase1_response},
        ]
        if is_phase15 and follow_up_prompt:
            messages.append({"role": "user", "content": follow_up_prompt})
        if is_phase15 and phase15_response:
            messages.append({"role": "assistant", "content": phase15_response})
        messages.append({"role": "user", "content": follow_up_prompt_text})

        phase1_model = item.get("model") or args.model
        model_api_key = get_model_api_key(phase1_model, args)
        if not model_api_key:
            raise RuntimeError(
                f"Missing API key for model '{phase1_model}'. Set {GEMINI_API_KEY_ENV if is_gemini_model(phase1_model) else DEFAULT_OPENAI_API_KEY_ENV}."
            )

        try:
            if is_gemini_model(phase1_model):
                phase2_response = call_gemini(
                    messages=messages,
                    model=phase1_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    api_key=model_api_key,
                    timeout=args.timeout,
                    prompt_id=prompt_id,
                    controller=gemini_controller,
                    max_retries=args.max_retries,
                )
            else:
                phase2_response = call_openai(
                    messages=messages,
                    model=phase1_model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    api_key=model_api_key,
                    prompt_id=prompt_id,
                    controller=openai_controller,
                    max_retries=args.max_retries,
                )
        except Exception as exc:  # pragma: no cover
            print(f"[{idx}/{total}] Phase 2 call error: {exc}")
            phase2_response = f"[error: {exc}]"

        base_row: dict[str, str] = {name: (item.get(name) or "") for name in phase15_fieldnames}
        base_row["p2_prompt"] = follow_up_prompt_text
        base_row["p2_response"] = phase2_response
        base_row["p2_category"] = categorize_p2_intent(follow_up_prompt_text)
        return idx, base_row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = [
            ex.submit(process_one, i, combined_items[i - 1])
            for i in range(1, len(combined_items) + 1)
        ]
        for fut in concurrent.futures.as_completed(futures):
            idx_done, row_done = fut.result()
            output_rows[idx_done - 1] = row_done

    final_rows: list[dict[str, str]] = [r for r in output_rows if r is not None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_fieldnames = phase15_fieldnames + ["p2_prompt", "p2_response", "p2_category"]
    with args.out.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Wrote Phase 2 responses to {args.out} ({len(final_rows)} rows)")


def main() -> None:
    _run_main()


if __name__ == "__main__":
    main()

