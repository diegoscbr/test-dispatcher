#!/usr/bin/env python3
"""Generate GEO/AI-visibility search prompts per persona-backed brand.

Input:
  - audience_habbits.csv with the columns: persona_id, title, trends, summary, motivations, characteristics, minAge, maxAge, country, preferred_models, conversation_style, search_behaviors.

Output:
  - prompts.csv with columns:
      brand_id, brand, prompt_id, prompt, category, phase, keywords, funnel
  - context_prompts.csv: same schema as prompts.csv, but contains both the base prompt and a contextualized variant per prompt (with persona-derived context prefix).

Notes:
  - Each persona row provides the brand name/text used to infer industry + search habits, so there is no separate brand.csv.
  - Uses OPENAI_API_KEY from the environment.

Usage:
  python prompt.py --persona_csv audience_habbits.csv --out prompts.csv

Optional:
  python prompt.py --persona_csv audience_habbits.csv --out prompts.csv --model gpt-4o-mini --n_per_category 6
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from model_usage_logger import ModelUsageLogger, extract_usage_from_response
except ImportError:

    class ModelUsageLogger:  # type: ignore[override]
        """No-op fallback when usage logging module is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def record(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def flush(self, *_args: Any, **_kwargs: Any) -> int:
            return 0

    def extract_usage_from_response(_response: Any) -> Dict[str, int]:
        return {}


PHASE_1_CATEGORIES: List[str] = [
    "Learn & Understand",  # definitions, how it works, basics
    "Recommendations",  # what should I choose, best for me
    "Compare & Decide",  # compare options/brands, pros/cons
    "Price & Value",  # cost, worth it, deals, budgeting
    "How-To & Setup",  # getting started, setup, usage
    "Fix & Troubleshoot",  # problems, errors, not working
    "Reviews & Social Proof",  # what people say, reddit, creator recs
]

SEARCH_TRIGGERING_GUIDANCE: List[str] = [
    "Lean into time-sensitive or ‘right now’ asks (e.g., “current interest rates,” “latest rumors,” “what’s trending today”) so the search tool is essential.",
    "Favor comparative shopping or recommendation formats that demand up-to-date pros/cons, pricing, or availability information.",
    "Include location-specific cues (“near me,” “in Brooklyn,” “local to [city]”) so geography-sensitive search results are needed.",
    "Request external credibility by asking what reviews, experts, or policies say about a brand, product, or service.",
    "When evaluating a brand, mention reputation, reliability, or policy changes that require external confirmation.",
    "Prompt for lists or rankings (“top 5,” “best,” “companies that…”) that naturally point to sourced search results.",
]
SEARCH_TRIGGERING_AVOIDANCE: List[str] = [
    "Avoid evergreen definitions, procedural how-tos, or abstract opinions that the assistant could answer without grounding.",
    "Skip short follow-ups that merely narrow context; only include them when they still require fresh, externally sourced data.",
]

FUNNEL_STAGES: List[str] = ["Awareness", "Consideration", "Conversion"]

FUNNEL_CONVERSION_KEYWORDS: List[str] = [
    "buy",
    "purchase",
    "pricing",
    "cost",
    "subscribe",
    "order",
    "trial",
    "quote",
    "demo",
    "plan",
    "checkout",
    "activate",
    "renew",
    "contract",
]

_USAGE_LOGGER = ModelUsageLogger(
    Path(__file__), identifier_label="prompt_generation_step"
)

FUNNEL_CONSIDERATION_KEYWORDS: List[str] = [
    "compare",
    "vs",
    "alternatives",
    "features",
    "capabilities",
    "evaluate",
    "review",
    "benefits",
    "pros",
    "cons",
    "options",
    "best",
    "choose",
    "difference",
    "score",
]

FUNNEL_FALLBACK_TEMPLATES: Dict[str, str] = {
    "Awareness": "I'm just learning about {industry}. What should I understand about it before evaluating options?",
    "Consideration": "Compare {industry} alternatives for me so I can choose the best fit.",
    "Conversion": "I'm ready to get {industry}. What are the next steps to buy or trial it today?",
}

FUNNEL_STAGE_CATEGORY: Dict[str, str] = {
    "Awareness": "Learn & Understand",
    "Consideration": "Compare & Decide",
    "Conversion": "Recommendations",
}

DEFAULT_PROMPTS_PER_PERSONA = 10
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_PMGCLIENT_JSON = Path("pmgclient.json")
DEFAULT_CONTEXT_OUT = "context_prompts.csv"


def _coalesce_raw_value(raw: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return ""


@dataclass
class ClientMetadata:
    id: str
    name: str
    slug: Optional[str]
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> Optional["ClientMetadata"]:
        client_id = _coalesce_raw_value(raw, ("brand_id", "id", "client_id"))
        client_name = _coalesce_raw_value(raw, ("name", "brand", "client_name"))
        if not client_id or not client_name:
            return None
        slug = _coalesce_raw_value(raw, ("slug", "client_slug"))
        return cls(id=client_id, name=client_name, slug=slug or None, raw=raw)


def load_pmg_clients(path: Path) -> List[ClientMetadata]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: List[Mapping[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                entries = [
                    candidate for candidate in value if isinstance(candidate, dict)
                ]
                break
        if not entries:
            entries = [payload]
    elif isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
    else:
        raise ValueError(f"Unsupported pmgclient structure: {type(payload).__name__}")

    clients: List[ClientMetadata] = []
    for entry in entries:
        client = ClientMetadata.from_raw(entry)
        if client:
            clients.append(client)
    return clients


def select_client(
    clients: Sequence[ClientMetadata],
    *,
    client_id: Optional[str] = None,
    client_name: Optional[str] = None,
    client_slug: Optional[str] = None,
) -> ClientMetadata:
    if not clients:
        raise ValueError("pmgclient.json contains no valid client entries")

    def _match(
        value: str, extractor: Callable[[ClientMetadata], Optional[str]]
    ) -> Optional[ClientMetadata]:
        target = value.casefold()
        for candidate in clients:
            attr = extractor(candidate)
            if attr and attr.casefold() == target:
                return candidate
        return None

    if client_id:
        match = _match(client_id, lambda c: c.id)
        if match:
            return match
        raise ValueError(f"No client in pmgclient.json has id '{client_id}'")

    if client_slug:
        match = _match(client_slug, lambda c: c.slug or "")
        if match:
            return match
        raise ValueError(f"No client in pmgclient.json has slug '{client_slug}'")

    if client_name:
        match = _match(client_name, lambda c: c.name)
        if match:
            return match
        raise ValueError(f"No client in pmgclient.json has name '{client_name}'")

    if len(clients) == 1:
        return clients[0]

    options = ", ".join(f"{client.name} ({client.id})" for client in clients)
    raise ValueError(
        "Multiple clients found in pmgclient.json. "
        "Specify --client-id, --client-name, or --client-slug to choose one. "
        f"Available clients: {options}"
    )


# -----------------------------
# OpenAI client compatibility
# -----------------------------


def _get_openai_client():
    """Return a callable (model, messages, **kwargs) -> text.

    Supports:
      - openai>=1.0.0 (OpenAI client)
      - openai<1.0.0 (legacy ChatCompletion)

    Requires OPENAI_API_KEY env var.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is required")

    # Prefer new SDK if available
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

        def _call(
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.2,
            max_tokens: int = 800,
            category: Optional[str] = None,
        ) -> str:
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            duration = time.perf_counter() - start
            _USAGE_LOGGER.record(
                model,
                duration,
                extract_usage_from_response(resp),
                category=category or "default",
            )
            return (resp.choices[0].message.content or "").strip()

        return _call

    except Exception:
        pass

    # Fallback to legacy SDK
    try:
        import openai  # type: ignore

        openai.api_key = api_key

        def _call(
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.2,
            max_tokens: int = 800,
            category: Optional[str] = None,
        ) -> str:
            start = time.perf_counter()
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            duration = time.perf_counter() - start
            _USAGE_LOGGER.record(
                model,
                duration,
                extract_usage_from_response(resp),
                category=category or "default",
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()

        return _call

    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAI client. Install `openai` package (v1+ recommended)."
        ) from e


# -----------------------------
# Core prompt generation logic
# -----------------------------


@dataclass
class PersonaHabit:
    persona_id: str
    title: str
    preferred_models: List[str]
    conversation_style: str
    search_behaviors: List[str]
    brand: Optional[str] = None
    model_notes: Optional[str] = None
    trends: Optional[str] = None
    summary: Optional[str] = None
    motivations: Optional[str] = None
    characteristics: Optional[str] = None
    min_age: Optional[str] = None
    max_age: Optional[str] = None
    intent: Optional[str] = None
    profile_image_url: Optional[str] = None
    total_addressable_audience_size: Optional[str] = None
    published: Optional[str] = None
    archived: Optional[str] = None
    view: Optional[str] = None
    created_by: Optional[str] = None
    country: Optional[str] = None
    created_at: Optional[str] = None
    has_cards: Optional[str] = None
    metadata: Optional[str] = None
    v2_persona_id: Optional[str] = None
    persona_type: Optional[str] = None


@dataclass
class BrandContext:
    brand: str
    industry: str
    audiences: List[str]
    search_habits: List[str]
    persona_id: Optional[str] = None
    persona_title: Optional[str] = None
    persona_search_habits: List[str] = field(default_factory=list)
    persona_conversation_style: Optional[str] = None
    persona_summary: Optional[str] = None
    persona_trends: Optional[str] = None
    persona_motivations: Optional[str] = None
    persona_characteristics: Optional[str] = None
    persona_model_notes: Optional[str] = None
    persona_intent: Optional[str] = None
    persona_age_range: Optional[str] = None
    related_keywords: List[str] = field(default_factory=list)


@dataclass
class KeywordRow:
    keyword_name: str
    platform: Optional[str] = None
    date: Optional[str] = None
    cost: Optional[str] = None
    impressions: Optional[str] = None
    clicks: Optional[str] = None
    conversions: Optional[str] = None


def _stringify_persona_value(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (int, float, bool)):
        return str(raw).strip()
    if isinstance(raw, (list, tuple, set)):
        parts = [_stringify_persona_value(item) for item in raw]
        return ", ".join(part for part in parts if part)
    if isinstance(raw, Mapping):
        return json.dumps(raw, ensure_ascii=True, sort_keys=True)
    return str(raw).strip()


def _coalesce_persona_value(raw: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key not in raw:
            continue
        normalized = _stringify_persona_value(raw.get(key))
        if normalized:
            return normalized
    return ""


def _parse_search_behaviors(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]

    value = _stringify_persona_value(raw)
    if not value:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(value)
            if isinstance(data, str):
                return [data.strip()]
            if isinstance(data, (list, tuple, set)):
                return [str(item).strip() for item in data if str(item).strip()]
        except Exception:
            continue

    cleaned = value.strip("[]")
    return [
        item.strip(" \t\"'").strip()
        for item in cleaned.split(",")
        if item.strip(" \t\"'")
    ]


def _parse_preferred_models(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]

    value = _stringify_persona_value(raw)
    if not value:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, str):
                return [parsed.strip()] if parsed.strip() else []
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            continue

    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _extract_persona_records_from_json(
    payload: Any, source: Path
) -> List[Mapping[str, Any]]:
    records: List[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = []
        for key in ("personas", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if not records:
            values = list(payload.values())
            if values and all(isinstance(value, Mapping) for value in values):
                records = values
            else:
                records = [payload]
    else:
        raise ValueError(
            f"{source} must be a JSON object, a JSON array, or an object containing personas."
        )

    normalized = [record for record in records if isinstance(record, Mapping)]
    if not normalized:
        raise ValueError(f"{source} contains no persona records.")
    return normalized


def _resolve_persona_input_path(path: str) -> Path:
    requested = Path(path).expanduser()
    suffix = requested.suffix.lower()

    candidates: List[Path] = [requested]
    if suffix == ".csv":
        candidates.append(requested.with_suffix(".json"))
    elif suffix == ".json":
        candidates.append(requested.with_suffix(".csv"))
    else:
        candidates.extend(
            [requested.with_suffix(".csv"), requested.with_suffix(".json")]
        )

    if requested.name == "audience_habbits.csv":
        candidates.extend(
            [
                requested.with_name("persona_details.json"),
                requested.with_name("persona_details_by_id.json"),
                requested.with_name("personas.json"),
                requested.with_name("persona.json"),
                SCRIPT_ROOT / "persona_details.json",
                SCRIPT_ROOT / "persona_details_by_id.json",
            ]
        )

    checked: List[str] = []
    seen: Set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    raise FileNotFoundError("Persona input not found. Checked: " + ", ".join(checked))


def load_persona_habits(path: str) -> List[PersonaHabit]:
    input_path = _resolve_persona_input_path(path)

    raw_rows: List[Mapping[str, Any]] = []
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        raw_rows = _extract_persona_records_from_json(payload, input_path)
    else:
        with input_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{input_path} has no header row")
            for row in reader:
                normalized = {
                    (key or "").strip(): (value or "").strip()
                    for key, value in row.items()
                    if key and key.strip()
                }
                if normalized:
                    raw_rows.append(normalized)

    if not raw_rows:
        raise ValueError(f"{input_path} contains no persona records.")

    personas: List[PersonaHabit] = []
    seen_ids: Set[str] = set()
    for row_index, row in enumerate(raw_rows, start=1):
        persona_id = (
            _coalesce_persona_value(
                row, ("persona_id", "id", "_id", "v2PersonaId", "v2_persona_id")
            )
            or f"persona_{row_index:04d}"
        )
        if persona_id in seen_ids:
            continue
        seen_ids.add(persona_id)

        title = _coalesce_persona_value(row, ("title", "prompt", "name")) or (
            f"Persona {row_index}"
        )
        brand_value = _coalesce_persona_value(
            row, ("brand", "client_name", "client", "name")
        )
        if not brand_value:
            brand_value = title

        search_behaviors = _parse_search_behaviors(row.get("search_behaviors"))
        if not search_behaviors:
            search_behaviors = _parse_search_behaviors(row.get("search_habits"))
        if not search_behaviors:
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping):
                topics = metadata.get("googleTopics") or metadata.get("google_topics")
                if isinstance(topics, Mapping):
                    search_behaviors = _parse_search_behaviors(topics.get("queries"))

        personas.append(
            PersonaHabit(
                persona_id=persona_id,
                title=title,
                preferred_models=_parse_preferred_models(row.get("preferred_models")),
                conversation_style=_coalesce_persona_value(
                    row, ("conversation_style", "tone", "style")
                ),
                search_behaviors=search_behaviors,
                model_notes=_coalesce_persona_value(row, ("model_notes",)),
                trends=_coalesce_persona_value(row, ("trends",)),
                summary=_coalesce_persona_value(row, ("summary",)),
                motivations=_coalesce_persona_value(row, ("motivations",)),
                characteristics=_coalesce_persona_value(row, ("characteristics",)),
                min_age=_coalesce_persona_value(row, ("minAge", "min_age")),
                max_age=_coalesce_persona_value(row, ("maxAge", "max_age")),
                intent=_coalesce_persona_value(row, ("intent",)),
                profile_image_url=_coalesce_persona_value(
                    row, ("profileImageUrl", "profile_image_url")
                ),
                total_addressable_audience_size=_coalesce_persona_value(
                    row,
                    (
                        "totalAddressableAudienceSize",
                        "total_addressable_audience_size",
                    ),
                ),
                published=_coalesce_persona_value(row, ("published",)),
                archived=_coalesce_persona_value(row, ("archived",)),
                view=_coalesce_persona_value(row, ("view",)),
                created_by=_coalesce_persona_value(row, ("createdBy", "created_by")),
                country=_coalesce_persona_value(row, ("country",)),
                created_at=_coalesce_persona_value(row, ("createdAt", "created_at")),
                has_cards=_coalesce_persona_value(row, ("hasCards", "has_cards")),
                metadata=_coalesce_persona_value(row, ("metadata",)),
                v2_persona_id=_coalesce_persona_value(
                    row, ("v2PersonaId", "v2_persona_id")
                ),
                brand=brand_value,
                persona_type=_coalesce_persona_value(row, ("persona_type", "type")),
            )
        )

    return personas


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _text_tokens(text: str, min_length: int = 3) -> Set[str]:
    tokens: Set[str] = set()
    if not text:
        return tokens
    for match in _TOKEN_RE.findall(text):
        if len(match) < min_length:
            continue
        tokens.add(match.lower())
    return tokens


def load_keywords_csv(path: Path) -> List[KeywordRow]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    rows: List[KeywordRow] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        for raw in reader:
            normalized = {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in raw.items()
                if key
            }
            keyword_name = normalized.get("keyword_name") or ""
            if not keyword_name:
                continue
            rows.append(
                KeywordRow(
                    keyword_name=keyword_name,
                    platform=normalized.get("platform") or None,
                    date=normalized.get("date") or None,
                    cost=normalized.get("cost") or None,
                    impressions=normalized.get("impressions") or None,
                    clicks=normalized.get("clicks") or None,
                    conversions=normalized.get("conversions") or None,
                )
            )
    return rows


def _normalize_header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def _select_two_name_columns(headers: Sequence[str]) -> List[str]:
    normalized_headers = [
        (header, _normalize_header_key(header)) for header in headers if header
    ]
    columns: List[str] = []
    if not normalized_headers:
        return columns

    def add_matching(hint: str) -> bool:
        nonlocal columns
        for header, normalized in normalized_headers:
            if len(columns) == 2:
                return True
            if normalized == hint and header not in columns:
                columns.append(header)
                if len(columns) == 2:
                    return True
        return False

    for hint in ("corebrand", "brandname", "brand", "name"):
        if add_matching(hint):
            break

    if len(columns) < 2:
        for hint in ("competitor", "competitors"):
            if add_matching(hint):
                break

    if len(columns) < 2:
        for header, normalized in normalized_headers:
            if len(columns) == 2:
                break
            if not normalized:
                continue
            if (
                any(tag in normalized for tag in ("brand", "competitor", "name"))
                and header not in columns
            ):
                columns.append(header)
    return columns[:2]


def load_competitor_brand_names(path: Path, target_name: str) -> Set[str]:
    names: Set[str] = {target_name.strip()} if target_name else set()
    if not path.exists():
        return names
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return names
            name_columns = _select_two_name_columns(reader.fieldnames)
            if not name_columns:
                return names
            target_lower = target_name.strip().lower() if target_name else ""
            for row in reader:
                row_names = []
                for column in name_columns:
                    value = (row.get(column) or "").strip()
                    if value:
                        row_names.append(value)
                if not row_names:
                    continue
                if target_lower and any(r.lower() == target_lower for r in row_names):
                    names.update(row_names)
    except Exception:
        return names
    return names


def filter_keywords_without_brand_names(
    keywords: Sequence[KeywordRow], brand_names: Sequence[str]
) -> List[KeywordRow]:
    normalized_terms = [
        term.strip().lower() for term in brand_names if term and term.strip()
    ]
    if not normalized_terms:
        return list(keywords)
    filtered: List[KeywordRow] = []
    for row in keywords:
        keyword_lower = row.keyword_name.lower()
        if any(term in keyword_lower for term in normalized_terms):
            continue
        filtered.append(row)
    return filtered


def select_keywords_for_context(
    ctx: BrandContext,
    brand_name: str,
    keywords: Sequence[KeywordRow],
    max_keywords: int = 24,
    sample_top_k: int = 80,
) -> List[str]:
    """Select a diverse set of context-relevant keywords.

    Key behavior changes:
      - Default max_keywords increased (was 6)
      - Samples from a top-K candidate pool (adds randomness between runs)
      - Dedupe is case-insensitive
    """

    if not keywords:
        return []

    brand_tokens = _text_tokens(brand_name, min_length=2) if brand_name else set()
    context_sources: List[str] = [
        ctx.industry,
        *(ctx.audiences or []),
        *(ctx.search_habits or []),
        *(ctx.persona_search_habits or []),
        ctx.persona_summary or "",
        ctx.persona_trends or "",
        ctx.persona_motivations or "",
        ctx.persona_characteristics or "",
    ]
    context_tokens: Set[str] = set()
    for source in context_sources:
        context_tokens |= _text_tokens(source)

    candidates: List[Tuple[int, int, int, str]] = []
    for idx, row in enumerate(keywords):
        kw = (row.keyword_name or "").strip()
        if not kw:
            continue

        # Exclude branded keywords (any overlap with brand tokens)
        kw_tokens = _text_tokens(kw, min_length=2)
        if brand_tokens and not brand_tokens.isdisjoint(kw_tokens):
            continue

        # Context overlap score
        score = len(kw_tokens & context_tokens)

        # Light platform boost if the platform appears in search habits
        platform_value = (row.platform or "").strip().lower()
        platform_match = 0
        if platform_value:
            for habit in ctx.search_habits or []:
                if platform_value in habit.lower():
                    platform_match = 1
                    break

        total_score = score + platform_match
        if total_score <= 0:
            continue

        # Note: keep -idx as a stable tiebreaker before sampling
        candidates.append((total_score, score, -idx, kw))

    if not candidates:
        return []

    # Rank by relevance, then sample from the top-K pool for diversity.
    candidates.sort(key=lambda tpl: (-tpl[0], -tpl[1], tpl[2]))
    top = candidates[: min(sample_top_k, len(candidates))]
    random.shuffle(top)

    selected: List[str] = []
    seen: Set[str] = set()
    for _, _, _, keyword in top:
        key = keyword.lower()
        if key in seen:
            continue
        selected.append(keyword)
        seen.add(key)
        if len(selected) >= max_keywords:
            break

    return selected


def detect_keywords_in_prompt(prompt: str, keywords: Sequence[str]) -> List[str]:
    """Return matched keywords found in `prompt`, preferring more specific phrases.

    Behavior:
      - Case-insensitive
      - Uses word-boundary-ish matching (\b ... \b)
      - Matches longest phrases first
      - Removes matches that are substrings of other matched keywords

    This prevents generic keywords (e.g., "music") from drowning out more specific phrases
    (e.g., "music streaming app").
    """

    if not prompt or not keywords:
        return []

    normalized = prompt.lower()

    # Unique keywords (case-insensitive), keep canonical display form
    canonical: Dict[str, str] = {}
    for k in keywords:
        k = (k or "").strip()
        if not k:
            continue
        canonical.setdefault(k.lower(), k)

    # Longest-first matching
    candidates = sorted(canonical.items(), key=lambda kv: -len(kv[0]))

    matched: List[str] = []
    matched_lc: List[str] = []
    for k_lc, k_disp in candidates:
        pattern = r"\b" + re.escape(k_lc) + r"\b"
        if re.search(pattern, normalized):
            matched.append(k_disp)
            matched_lc.append(k_lc)

    # Remove substrings: if "music streaming app" matched, drop "music" and "music streaming"
    filtered: List[str] = []
    for i, k in enumerate(matched):
        k_lc = matched_lc[i]
        if any((k_lc != other) and (k_lc in other) for other in matched_lc):
            continue
        filtered.append(k)

    return filtered


def merge_keyword_sequences(*sequences: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen: Set[str] = set()
    for seq in sequences:
        for keyword in seq:
            candidate = (keyword or "").strip()
            if not candidate:
                continue
            key_lower = candidate.lower()
            if key_lower in seen:
                continue
            seen.add(key_lower)
            merged.append(candidate)
    return merged


def categorize_funnel_stage(text: str) -> str:
    normalized = (text or "").lower()
    if not normalized:
        return "Awareness"
    for keyword in FUNNEL_CONVERSION_KEYWORDS:
        if keyword in normalized:
            return "Conversion"
    for keyword in FUNNEL_CONSIDERATION_KEYWORDS:
        if keyword in normalized:
            return "Consideration"
    return "Awareness"


def _format_funnel_prompt(stage: str, industry: Optional[str]) -> str:
    template = FUNNEL_FALLBACK_TEMPLATES.get(stage)
    if not template:
        template = "Help me with {industry}."
    industry_text = industry or "this industry"
    return template.format(industry=industry_text)


def summarize_persona_summary(
    call_llm, model: str, summary: str, retries: int = 2
) -> str:
    """Condense persona summary text into a short context snippet via the LLM."""

    summary_text = (summary or "").strip()
    if not summary_text:
        return ""

    user_prompt = (
        "Summarize the persona summary below into 1-2 concise sentences written from the persona's "
        "first-person perspective (e.g., 'I value...' or 'I'm looking for...') that capture their perspective "
        "and priorities when asking an AI assistant for help. Provide only plain text, no JSON or markup.\n\n"
        f"PERSONA SUMMARY: {summary_text}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    def _extract_text(candidate: str) -> str:
        candidate = (candidate or "").strip()
        if not candidate:
            return ""
        try:
            data = _robust_json_loads(candidate)
            if isinstance(data, dict):
                value = (
                    data.get("summary")
                    or data.get("Summary")
                    or data.get("summary_text")
                )
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except Exception:
            pass
        return candidate

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            text = call_llm(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=200,
                category="context",
            )
            candidate = _extract_text(text)
            if candidate:
                return candidate
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * attempt)

    fallback = summary_text.split("\n")[0].split(".")[0].strip()
    if fallback:
        safe = fallback
        if not re.match(r"(?i)^i\b", safe):
            safe = "I " + (
                safe[0].lower() + safe[1:] if len(safe) > 1 else safe.lower()
            )
        return safe
    if last_err:
        print(
            f"[warn] Persona summary summarization failed: {last_err}", file=sys.stderr
        )
    return summary_text


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def build_persona_guidance(ctx: BrandContext) -> List[str]:
    guidance: List[str] = []
    if ctx.persona_title:
        guidance.append(
            f"Keep the prompts aligned with persona '{ctx.persona_title}' yet broadly applicable."
        )
    if ctx.persona_conversation_style:
        guidance.append(
            f"Use a {ctx.persona_conversation_style} tone when phrasing requests."
        )
    if ctx.persona_search_habits:
        habits = ", ".join(ctx.persona_search_habits[:3])
        guidance.append(
            f"Draw inspiration from habits such as {habits}, but avoid overly specific references."
        )
    if ctx.persona_trends:
        guidance.append(
            f"Focus on trending topics the persona engages with: {ctx.persona_trends}."
        )
    if ctx.persona_summary:
        guidance.append(
            f"Remember that the persona is described as: {ctx.persona_summary}."
        )
    if ctx.persona_characteristics:
        guidance.append(
            f"Incorporate the persona's characteristics: {ctx.persona_characteristics}."
        )
    if ctx.persona_motivations:
        guidance.append(
            f"Motivate the prompts with persona goals like: {ctx.persona_motivations}."
        )
    if ctx.persona_age_range:
        guidance.append(f"Target age range: {ctx.persona_age_range}.")
    if ctx.persona_intent:
        guidance.append(
            f"Intent stage: {ctx.persona_intent} (Discovery, Research/Evaluation, Decision Support, Purchase Assistance, Post-Purchase Support, or Ongoing Utility)."
        )
    if ctx.persona_model_notes:
        guidance.append(f"Model notes: {ctx.persona_model_notes}.")
    return guidance


def describe_persona(persona: PersonaHabit) -> str:
    parts = [persona.title]
    if persona.conversation_style:
        parts.append(f"tone: {persona.conversation_style}")
    if persona.search_behaviors:
        habits = ", ".join(persona.search_behaviors[:3])
        parts.append(f"habits: {habits}")
    if persona.trends:
        parts.append(f"trends: {persona.trends}")
    if persona.summary:
        parts.append(f"summary: {persona.summary}")
    if persona.motivations:
        parts.append(f"motivations: {persona.motivations}")
    if persona.characteristics:
        parts.append(f"characteristics: {persona.characteristics}")
    return "; ".join(parts)


SYSTEM_PROMPT = (
    "You are a consumer search behavior analyst. "
    "You write prompts exactly like real people would ask an AI assistant (ChatGPT, Perplexity, Gemini, Claude). "
    "Prompts should be phrased as natural-language questions/requests, often personalized (constraints, context). "
    "Output must be valid JSON only."
)


def _robust_json_loads(text: str) -> dict:
    """Best-effort parse JSON from a model response.

    Handles common issues:
      - code fences
      - curly quotes
      - leading/trailing non-JSON text
    """

    raw = (text or "").strip()

    # Strip code fences (```json ... ```)
    if raw.startswith("```"):
        # Take content between the first and last fence
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = "```".join(parts[1:-1]).strip()
        else:
            raw = raw.strip("`").strip()
        # Drop a language tag line if present
        lines = raw.splitlines()
        if lines and lines[0].strip().lower() in {"json", "javascript", "js"}:
            raw = "\n".join(lines[1:]).strip()

    # Normalize curly quotes to straight quotes
    raw = (
        raw.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # First attempt: direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Second attempt: extract the largest JSON object block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1].strip()
        candidate = (
            candidate.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        return json.loads(candidate)

    # Give up
    return json.loads(raw)


# Helper: ask LLM to repair invalid JSON
def _repair_json_via_model(
    call_llm,
    model: str,
    bad_text: str,
    schema_hint: dict,
    retries: int = 2,
    category: Optional[str] = None,
) -> dict:
    """Ask the model to repair an invalid JSON blob into valid JSON matching schema_hint."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You returned invalid JSON. Repair it into VALID JSON only. "
                "Do not add commentary. Do not wrap in code fences. "
                "Ensure all strings are properly escaped and the output matches this schema.\n\n"
                f"SCHEMA_HINT: {json.dumps(schema_hint)}\n\n"
                f"INVALID_JSON: {bad_text}"
            ),
        },
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            txt = call_llm(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1400,
                category=category or "context",
            )
            return _robust_json_loads(txt)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"Failed to repair JSON. Last error: {last_err}")


def infer_brand_context(
    call_llm,
    model: str,
    brand: str,
    persona: Optional[PersonaHabit] = None,
    retries: int = 3,
) -> BrandContext:
    """Infer likely industry + audience and search behaviors from a brand name."""

    user_prompt = {
        "task": "infer_brand_context",
        "brand": brand,
        "requirements": {
            "industry": "Return a concise industry label (e.g., 'athletic apparel', 'CRM software', 'skincare').",
            "audiences": "Return 3-6 audience segments.",
            "search_habits": (
                "Return 6-10 bullet-like phrases describing typical search habits and platforms "
                "(e.g., 'uses TikTok for tutorials', 'reads Reddit for honest reviews', 'compares pricing pages')."
            ),
        },
        "output_schema": {
            "industry": "string",
            "audiences": ["string"],
            "search_habits": ["string"],
        },
    }

    if persona:
        user_prompt["persona"] = {
            "id": persona.persona_id,
            "title": persona.title,
            "conversation_style": persona.conversation_style,
            "search_habits": persona.search_behaviors,
            "trends": persona.trends,
            "summary": persona.summary,
            "motivations": persona.motivations,
            "characteristics": persona.characteristics,
            "model_notes": persona.model_notes,
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Infer the most likely context for this brand name. "
                "If uncertain, make the best plausible guess. "
                "Return JSON matching the schema exactly.\n\n" + json.dumps(user_prompt)
            ),
        },
    ]

    if persona:
        messages[1]["content"] += f"\n\nPersona context: {describe_persona(persona)}"

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        print(
            f"[progress] Inferring brand context for '{brand}' (attempt {attempt}/{retries})"
        )
        try:
            txt = call_llm(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
                category="context",
            )
            data = _robust_json_loads(txt)
            industry = str(data.get("industry", "")).strip() or "unknown"
            audiences = [
                str(x).strip() for x in (data.get("audiences") or []) if str(x).strip()
            ]
            search_habits = [
                str(x).strip()
                for x in (data.get("search_habits") or [])
                if str(x).strip()
            ]

            persona_search_habits = persona.search_behaviors if persona else []
            combined_search_habits: List[str] = []
            for habit in (*persona_search_habits, *search_habits):
                cleaned = (habit or "").strip()
                if cleaned and cleaned not in combined_search_habits:
                    combined_search_habits.append(cleaned)

            if not audiences:
                audiences = ["general consumers"]
            if not combined_search_habits:
                combined_search_habits = [
                    "uses Google for quick answers",
                    "checks reviews before purchase",
                ]

            age_range = None
            if persona and persona.min_age and persona.max_age:
                age_range = f"{persona.min_age}-{persona.max_age}"
            elif persona and persona.min_age:
                age_range = f"{persona.min_age}+"
            elif persona and persona.max_age:
                age_range = f"up to {persona.max_age}"
            return BrandContext(
                brand=brand,
                industry=industry,
                audiences=audiences[:8],
                search_habits=combined_search_habits[:12],
                persona_id=persona.persona_id if persona else None,
                persona_title=persona.title if persona else None,
                persona_search_habits=persona_search_habits,
                persona_conversation_style=persona.conversation_style
                if persona
                else None,
                persona_summary=persona.summary if persona else None,
                persona_trends=persona.trends if persona else None,
                persona_characteristics=persona.characteristics if persona else None,
                persona_motivations=persona.motivations if persona else None,
                persona_model_notes=persona.model_notes if persona else None,
                persona_intent=persona.intent if persona else None,
                persona_age_range=age_range,
            )
        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(
        f"Failed to infer brand context for '{brand}'. Last error: {last_err}"
    )


def infer_industry_questions(
    call_llm,
    model: str,
    ctx: BrandContext,
    n_questions: int = 20,
    retries: int = 3,
) -> List[str]:
    """Infer common consumer questions asked to AI assistants in this industry.

    This produces a canonical set of 'industry questions' that Phase 1 prompts will be based on.
    """

    guidelines = [
        "Return questions/requests as a user would ask an AI assistant.",
        "Use first-person language where natural (e.g., 'What's the best… for me', 'Help me…').",
        "Focus on the industry's real decision points: choosing, comparing, using, troubleshooting, and value.",
        "Do not write marketing copy.",
        "No duplicates; keep each item under ~200 characters when possible.",
    ]
    guidelines.extend(build_persona_guidance(ctx))
    guidelines.extend(SEARCH_TRIGGERING_GUIDANCE)
    guidelines.extend(SEARCH_TRIGGERING_AVOIDANCE)

    user_payload = {
        "task": "infer_industry_questions",
        "industry": ctx.industry,
        "brand": ctx.brand,
        "audiences": ctx.audiences,
        "search_habits": ctx.search_habits,
        "n_questions": n_questions,
        "guidelines": guidelines,
        "output_schema": {"questions": ["string"]},
        "persona_title": ctx.persona_title,
        "persona_search_habits": ctx.persona_search_habits,
        "persona_conversation_style": ctx.persona_conversation_style,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "List the most common questions consumers ask AI assistants in this industry. "
                "Return JSON matching the schema exactly.\n\n"
                + json.dumps(user_payload)
            ),
        },
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        print(
            f"[progress] Inferring industry questions for '{ctx.brand}' (attempt {attempt}/{retries})"
        )
        try:
            txt = call_llm(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=900,
                category="context",
            )
            try:
                data = _robust_json_loads(txt)
            except Exception:
                data = _repair_json_via_model(
                    call_llm=call_llm,
                    model=model,
                    bad_text=txt,
                    schema_hint={"questions": ["string"]},
                    category="context",
                )
            qs = data.get("questions") or []
            qs = [str(q).strip() for q in qs if str(q).strip()]

            # De-dupe preserving order
            seen = set()
            out: List[str] = []
            for q in qs:
                k = q.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(q)

            if not out:
                return [
                    f"What should I know before choosing a {ctx.industry}?",
                    f"How do I pick the best {ctx.industry} option for my needs?",
                ]

            return out[: max(5, n_questions)]

        except Exception as e:
            last_err = e
            time.sleep(0.6 * attempt)

    raise RuntimeError(
        f"Failed to infer industry questions for '{ctx.brand}' ({ctx.industry}). Last error: {last_err}"
    )


# -------------------------------
# Helper: Generate non-brand backfill prompts
# -------------------------------
def generate_nonbrand_backfill(
    call_llm,
    model: str,
    ctx: BrandContext,
    industry_questions: List[str],
    needed_by_category: Dict[str, int],
    retries: int = 2,
    force_keywords: bool = False,
) -> Dict[str, List[str]]:
    """Generate additional prompts that MUST NOT mention the brand name, used to backfill after filtering."""

    # Only request categories we actually need to fill
    cats = [c for c in PHASE_1_CATEGORIES if needed_by_category.get(c, 0) > 0]
    if not cats:
        return {}

    output_schema = {"prompts": {c: ["string"] for c in cats}}

    guidelines = [
        "Generate additional Phase 1 prompts to fill missing slots per category.",
        "CRITICAL: Do NOT mention the brand name anywhere in any prompt.",
        "If you accidentally include the brand name, you must replace it with generic wording instead of leaving it in.",
        "Write prompts like real people asking an AI assistant (questions/requests).",
        "Prefer industry/general wording (e.g., 'a hotel chain', 'a CRM', 'running shoes') where needed.",
        "No duplicates across categories; keep prompts concise and natural.",
        'Do not include double-quote characters (") in any prompt.',
    ]
    guidelines.extend(build_persona_guidance(ctx))
    guidelines.extend(SEARCH_TRIGGERING_GUIDANCE)
    guidelines.extend(SEARCH_TRIGGERING_AVOIDANCE)

    if ctx.related_keywords:
        keywords_list = ", ".join(ctx.related_keywords)
        guidelines.append(
            "When appropriate, fold one of these keywords into the prompt while keeping the tone non-branded: "
            f"{keywords_list}"
        )
        guidelines.append(
            "Only use keywords that align with the request; skip them if they clash with the prompt or mention the brand."
        )
        if force_keywords:
            guidelines.append(
                "Force one of these keywords to be the centerpiece of each prompt; frame an explicit scenario around the keyword while still sounding like a natural AI assistant ask."
            )

    user_payload = {
        "task": "generate_nonbrand_backfill_prompts",
        "phase": 1,
        "brand": ctx.brand,
        "industry": ctx.industry,
        "audiences": ctx.audiences,
        "search_habits": ctx.search_habits,
        "industry_questions": industry_questions,
        "categories": cats,
        "n_per_category": {c: int(needed_by_category[c]) for c in cats},
        "guidelines": guidelines,
        "output_schema": output_schema,
        "persona_title": ctx.persona_title,
        "persona_search_habits": ctx.persona_search_habits,
        "persona_conversation_style": ctx.persona_conversation_style,
    }
    if ctx.related_keywords:
        user_payload["keywords"] = ctx.related_keywords

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate ONLY non-brand prompts. Return JSON matching the schema exactly.\n\n"
                + json.dumps(user_payload)
            ),
        },
    ]

    last_err: Optional[Exception] = None
    needed_total = sum(needed_by_category.get(c, 0) for c in cats)
    for attempt in range(1, retries + 1):
        print(
            f"[progress] Non-brand backfill needs {needed_total} prompts for '{ctx.brand}' "
            f"(attempt {attempt}/{retries})"
        )
        try:
            txt = call_llm(
                model=model,
                messages=messages,
                temperature=0.6,
                max_tokens=900,
                category="prompts",
            )
            try:
                data = _robust_json_loads(txt)
            except Exception:
                data = _repair_json_via_model(
                    call_llm=call_llm,
                    model=model,
                    bad_text=txt,
                    schema_hint=output_schema,
                    category="prompts",
                )

            prompts_obj = data.get("prompts") or {}
            out: Dict[str, List[str]] = {}
            for c in cats:
                vals = prompts_obj.get(c) or []
                vals = [str(v).strip() for v in vals if str(v).strip()]
                out[c] = vals
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(
        f"Failed to generate non-brand backfill prompts for '{ctx.brand}'. Last error: {last_err}"
    )


def generate_phase1_prompts(
    call_llm,
    model: str,
    ctx: BrandContext,
    industry_questions: List[str],
    n_per_category: int,
    retries: int = 3,
    force_keywords: bool = False,
) -> Dict[str, List[str]]:
    """Generate Phase 1 prompts grouped by category."""

    guidelines = [
        "Every prompt must sound like something a person would ask an AI assistant (not a search engine keyword string).",
        "Prefer question/request formats: 'What…', 'Which…', 'Can you…', 'Help me…', 'Recommend…'.",
        "Include personal constraints when helpful (budget, skill level, location, use-case).",
        "CRITICAL: NO prompts may mention the brand name or refer to the brand directly (0% brand mentions).",
        "Prompts must be category/industry-first and fully generic (brand-agnostic).",
        "A portion of prompts should be follow-up style questions that reference previous info implicitly (e.g., 'Given that, what should I do next?'), but still standalone text.",
        "No duplicates across all categories.",
        'Do not include double-quote characters (") in any prompt.',
    ]
    guidelines.extend(build_persona_guidance(ctx))
    if ctx.persona_intent:
        guidelines.append(
            f"Frame prompts to reflect the persona's intent stage: {ctx.persona_intent}."
        )
    if ctx.persona_age_range:
        guidelines.append(
            f"Target age-aware language appropriate for ages {ctx.persona_age_range}."
        )
    guidelines.extend(SEARCH_TRIGGERING_GUIDANCE)
    guidelines.extend(SEARCH_TRIGGERING_AVOIDANCE)

    if ctx.related_keywords:
        keywords_list = ", ".join(ctx.related_keywords)
        guidelines.append(
            "If a keyword fits naturally without mentioning the brand, weave one or two of these into the prompt: "
            f"{keywords_list}"
        )
        guidelines.append(
            "Use keywords sparingly and only when they match the prompt angle; omit them if they feel forced."
        )
        if force_keywords:
            guidelines.append(
                "While in keyword-forcing mode, make at least one keyword the anchor of each prompt and describe a concrete scenario that leans on that keyword."
            )

    user_payload = {
        "task": "generate_prompts_from_industry_questions",
        "phase": 1,
        "brand": ctx.brand,
        "industry": ctx.industry,
        "audiences": ctx.audiences,
        "search_habits": ctx.search_habits,
        "industry_questions": industry_questions,
        "categories": PHASE_1_CATEGORIES,
        "n_per_category": n_per_category,
        "guidelines": guidelines,
        "output_schema": {
            "prompts": {
                "Learn & Understand": ["string"],
                "Recommendations": ["string"],
                "Compare & Decide": ["string"],
                "Price & Value": ["string"],
                "How-To & Setup": ["string"],
                "Fix & Troubleshoot": ["string"],
                "Reviews & Social Proof": ["string"],
            }
        },
        "persona_title": ctx.persona_title,
        "persona_search_habits": ctx.persona_search_habits,
        "persona_conversation_style": ctx.persona_conversation_style,
    }
    if ctx.related_keywords:
        user_payload["keywords"] = ctx.related_keywords

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate Phase 1 search prompts. IMPORTANT: None of the prompts may mention the brand name or refer to the brand. Return JSON matching the schema exactly.\n\n"
                + json.dumps(user_payload)
            ),
        },
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        print(
            f"[progress] Generating Phase 1 prompts for '{ctx.brand}' (attempt {attempt}/{retries})"
        )
        try:
            txt = call_llm(
                model=model,
                messages=messages,
                temperature=0.6,
                max_tokens=1200,
                category="prompts",
            )
            try:
                data = _robust_json_loads(txt)
            except Exception:
                data = _repair_json_via_model(
                    call_llm=call_llm,
                    model=model,
                    bad_text=txt,
                    schema_hint={
                        "prompts": {
                            "Learn": ["string"],
                            "Choose": ["string"],
                            "Compare": ["string"],
                            "Price": ["string"],
                            "Use": ["string"],
                            "Fix": ["string"],
                            "Trust": ["string"],
                        }
                    },
                    category="prompts",
                )
            prompts_obj = data.get("prompts") or {}

            out: Dict[str, List[str]] = {}
            for cat in PHASE_1_CATEGORIES:
                vals = prompts_obj.get(cat) or []
                vals = [str(v).strip() for v in vals if str(v).strip()]
                # De-dupe within category preserving order
                seen = set()
                deduped = []
                for v in vals:
                    key = v.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(v)
                out[cat] = deduped[:n_per_category]

            # Prefer non-brand prompts: filter brand-mention prompts and backfill
            brand_lc = (ctx.brand or "").lower().strip()
            if brand_lc:
                # Filter brand mentions per category
                filtered: Dict[str, List[str]] = {}
                needed: Dict[str, int] = {}
                for cat in PHASE_1_CATEGORIES:
                    vals = out.get(cat, []) or []
                    nonbrand = [v for v in vals if brand_lc not in (v or "").lower()]
                    filtered[cat] = nonbrand
                    needed[cat] = max(0, n_per_category - len(nonbrand))

                # Backfill missing slots with explicitly non-brand prompts
                if any(n > 0 for n in needed.values()):
                    try:
                        backfill = generate_nonbrand_backfill(
                            call_llm=call_llm,
                            model=model,
                            ctx=ctx,
                            industry_questions=industry_questions,
                            needed_by_category=needed,
                            force_keywords=force_keywords,
                        )
                        for cat in PHASE_1_CATEGORIES:
                            if needed.get(cat, 0) <= 0:
                                continue
                            addl = backfill.get(cat) or []
                            addl = [str(v).strip() for v in addl if str(v).strip()]
                            # Ensure backfill also doesn't contain brand
                            addl = [
                                v for v in addl if brand_lc not in (v or "").lower()
                            ]
                            filtered[cat].extend(addl)
                    except Exception:
                        # If backfill fails, proceed with whatever we have
                        pass

                # Trim to requested count
                for cat in PHASE_1_CATEGORIES:
                    out[cat] = (filtered.get(cat) or [])[:n_per_category]

            # Ensure every category has at least 1 prompt
            for cat in PHASE_1_CATEGORIES:
                if not out.get(cat):
                    out[cat] = [f"Help me with {ctx.industry}: {cat.lower()}"]

            return out

        except Exception as e:
            last_err = e
            time.sleep(0.75 * attempt)

    raise RuntimeError(
        f"Failed to generate prompts for '{ctx.brand}'. Last error: {last_err}"
    )


def select_prompts_for_persona(
    prompts_by_category: Dict[str, List[str]],
    total: int,
    used_prompts: set[str],
    related_keywords: Sequence[str] = (),
    industry: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Choose up to `total` prompts across categories with a random yet balanced mix and funnel coverage."""

    all_entries: List[Tuple[str, str, str]] = []
    for category in PHASE_1_CATEGORIES:
        for prompt_text in prompts_by_category.get(category, []):
            cleaned = (prompt_text or "").strip()
            if not cleaned or cleaned in used_prompts:
                continue
            stage = categorize_funnel_stage(cleaned)
            all_entries.append((category, cleaned, stage))

    if not all_entries:
        return []

    # If we have related keywords, optimize for keyword diversity/coverage.
    # This prevents "keyword presence" logic from repeatedly selecting prompts that only match
    # the most generic keyword.
    if related_keywords:
        # Precompute keyword matches for each entry
        entry_kws: List[Tuple[Tuple[str, str, str], Set[str]]] = []
        for entry in all_entries:
            kws = set(
                k.lower() for k in detect_keywords_in_prompt(entry[1], related_keywords)
            )
            entry_kws.append((entry, kws))

        selected: List[Tuple[str, str, str]] = []
        covered: Set[str] = set()
        used_entry: Set[Tuple[str, str, str]] = set()

        # 1) Ensure funnel stage coverage first (Awareness/Consideration/Conversion)
        for stage in FUNNEL_STAGES:
            best = None
            best_gain = -1
            for entry, kws in entry_kws:
                if entry in used_entry:
                    continue
                if entry[2] != stage:
                    continue
                gain = len([k for k in kws if k not in covered])
                if gain > best_gain:
                    best_gain = gain
                    best = (entry, kws)
            if best and len(selected) < total:
                entry, kws = best
                selected.append(entry)
                used_entry.add(entry)
                covered |= kws

        # 2) Greedy: add prompts that introduce the most *new* keywords
        while len(selected) < total:
            best = None
            best_gain = -1
            for entry, kws in entry_kws:
                if entry in used_entry:
                    continue
                gain = len([k for k in kws if k not in covered])
                if gain > best_gain:
                    best_gain = gain
                    best = (entry, kws)
            if not best:
                break
            entry, kws = best
            selected.append(entry)
            used_entry.add(entry)
            covered |= kws

        # 3) If still short, fill randomly from remaining entries
        if len(selected) < total:
            remaining = [entry for entry, _ in entry_kws if entry not in used_entry]
            random.shuffle(remaining)
            for entry in remaining:
                if len(selected) >= total:
                    break
                selected.append(entry)
                used_entry.add(entry)

        # 4) Backfill missing stages with fallback templates if needed
        stage_selected = {s for (_, _, s) in selected}
        missing_stages = [
            stage for stage in FUNNEL_STAGES if stage not in stage_selected
        ]
        for stage in missing_stages:
            fallback_prompt = _format_funnel_prompt(stage, industry)
            if not fallback_prompt or fallback_prompt in used_prompts:
                continue
            fallback_category = FUNNEL_STAGE_CATEGORY.get(stage, PHASE_1_CATEGORIES[0])
            fallback_entry = (fallback_category, fallback_prompt, stage)
            selected.append(fallback_entry)
            if len(selected) >= total:
                break

        random.shuffle(selected)
        return selected[:total]

    random.shuffle(all_entries)
    selected_stage_entries: List[Tuple[str, str, str]] = []
    selected_other_entries: List[Tuple[str, str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    stage_selected: Set[str] = set()

    for stage in FUNNEL_STAGES:
        if len(selected_stage_entries) >= total:
            break
        for entry in all_entries:
            if entry[2] != stage or entry in seen:
                continue
            selected_stage_entries.append(entry)
            seen.add(entry)
            stage_selected.add(stage)
            break

    for entry in all_entries:
        if len(selected_stage_entries) + len(selected_other_entries) >= total:
            break
        if entry in seen:
            continue
        selected_other_entries.append(entry)
        seen.add(entry)

    missing_stages = [stage for stage in FUNNEL_STAGES if stage not in stage_selected]
    for stage in missing_stages:
        fallback_prompt = _format_funnel_prompt(stage, industry)
        if not fallback_prompt or fallback_prompt in used_prompts:
            continue
        fallback_category = FUNNEL_STAGE_CATEGORY.get(stage, PHASE_1_CATEGORIES[0])
        fallback_entry = (fallback_category, fallback_prompt, stage)
        selected_stage_entries.append(fallback_entry)
        stage_selected.add(stage)
        seen.add(fallback_entry)

    selected = selected_stage_entries + selected_other_entries

    final_selected: List[Tuple[str, str, str]] = []
    seen_final: Set[Tuple[str, str, str]] = set()
    for entry in selected:
        if len(final_selected) >= total:
            break
        if entry in seen_final:
            continue
        final_selected.append(entry)
        seen_final.add(entry)

    if len(final_selected) < total and len(selected) > len(final_selected):
        for entry in selected:
            if len(final_selected) >= total:
                break
            if entry in seen_final:
                continue
            final_selected.append(entry)
            seen_final.add(entry)

    random.shuffle(final_selected)
    return final_selected


# -----------------------------
# IO
# -----------------------------


def write_prompts_csv(
    path: str,
    rows: List[Dict[str, str]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = [
        "persona_id",
        "brand_id",
        "brand",
        "prompt_id",
        "prompt",
        "category",
        "keywords",
        "funnel",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows_for_brand(
    brand_index: int,
    brand_id: str,
    brand: str,
    prompt_entries: List[Tuple[str, str, str]],
    phase: int = 1,
    persona_id: Optional[str] = None,
    related_keywords: Sequence[str] = (),
    keyword_candidates: Sequence[str] = (),
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    prompt_counter = 1
    brand_lc = (brand or "").lower()

    persona_suffix = (persona_id or "base")[:8]
    keyword_pool = merge_keyword_sequences(related_keywords, keyword_candidates)
    for category, p, funnel_stage in prompt_entries:
        prompt_id = (
            f"b{brand_index:04d}_p{phase}_q{prompt_counter:04d}_{persona_suffix}"
        )
        used_keywords = detect_keywords_in_prompt(p, keyword_pool)
        rows.append(
            {
                "persona_id": persona_id or "",
                "brand_id": str(brand_id),
                "brand": brand,
                "prompt_id": prompt_id,
                "prompt": p,
                "category": category,
                "keywords": ", ".join(used_keywords),
                "funnel": funnel_stage,
            }
        )
        prompt_counter += 1

    return rows


def build_context_rows_for_brand(
    brand_index: int,
    brand_id: str,
    brand: str,
    base_rows: List[Dict[str, str]],
    context_summary: str,
    phase: int = 1,
    persona_id: Optional[str] = None,
    related_keywords: Sequence[str] = (),
    keyword_candidates: Sequence[str] = (),
) -> List[Dict[str, str]]:
    """Emit a context row per base prompt using the base prompt_id plus a context suffix."""

    rows: List[Dict[str, str]] = []
    brand_lc = (brand or "").lower()
    prefix = (context_summary or "").strip()

    keyword_pool = merge_keyword_sequences(related_keywords, keyword_candidates)
    for base in base_rows:
        base_prompt = (base.get("prompt") or "").strip()
        base_prompt_id = (base.get("prompt_id") or "").strip()
        if not base_prompt or not base_prompt_id:
            continue

        contextual = f"{prefix} {base_prompt}".strip() if prefix else base_prompt
        used_keywords = detect_keywords_in_prompt(contextual, keyword_pool)
        ctx_id = f"{base_prompt_id}_ctx"
        rows.append(
            {
                "persona_id": base.get("persona_id") or persona_id or "",
                "brand_id": str(base.get("brand_id") or brand_id),
                "brand": base.get("brand") or brand,
                "prompt_id": ctx_id,
                "prompt": contextual,
                "category": base.get("category") or "",
                "funnel": base.get("funnel") or "",
                "keywords": ", ".join(used_keywords),
            }
        )

    return rows


def print_prompt_generation_progress(
    base_generated: int,
    base_total: int,
    context_generated: int,
    context_total: int,
) -> None:
    """Overwrite a single line to show total prompt progress."""

    total_target = base_total + context_total
    if total_target <= 0:
        return

    total_generated = base_generated + context_generated
    pct = (total_generated / total_target) * 100 if total_target else 0.0
    line = (
        f"\rPrompts {total_generated}/{total_target} "
        f"({pct:.1f}%): base {base_generated}/{base_total} | "
        f"context {context_generated}/{context_total}"
    )

    prev_len = getattr(print_prompt_generation_progress, "line_len", 0)
    padded_line = line.ljust(prev_len) if len(line) < prev_len else line
    sys.stdout.write(padded_line)
    sys.stdout.flush()
    print_prompt_generation_progress.line_len = len(padded_line)
    print_prompt_generation_progress.has_printed = True


def _process_persona(
    brand_idx: int,
    persona: PersonaHabit,
    persona_label: str,
    total_personas: int,
    selected_client: ClientMetadata,
    call_llm: Callable[..., str],
    model: str,
    n_per_category: int,
    prompts_per_persona: int,
    used_prompts: set[str],
    used_prompts_lock: Lock,
    keyword_rows: Sequence[KeywordRow],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Run the full Phase 1 prompt pipeline for a single persona."""

    print(
        f"[info] ({brand_idx}/{total_personas}) Processing persona: {persona_label} for client {selected_client.name}"
    )
    ctx = infer_brand_context(
        call_llm=call_llm, model=model, brand=selected_client.name, persona=persona
    )
    keyword_suggestions = select_keywords_for_context(
        ctx=ctx,
        brand_name=selected_client.name,
        keywords=keyword_rows,
    )
    ctx.related_keywords = keyword_suggestions
    force_keywords = bool(ctx.related_keywords and random.random() < 0.5)
    industry_qs = infer_industry_questions(
        call_llm=call_llm, model=model, ctx=ctx, n_questions=24
    )
    cat_prompts = generate_phase1_prompts(
        call_llm=call_llm,
        model=model,
        ctx=ctx,
        industry_questions=industry_qs,
        n_per_category=n_per_category,
        force_keywords=force_keywords,
    )

    with used_prompts_lock:
        selected_prompts = select_prompts_for_persona(
            cat_prompts,
            prompts_per_persona,
            used_prompts,
            ctx.related_keywords,
            industry=ctx.industry,
        )
        if not selected_prompts:
            return [], []
        for _, prompt_text, _ in selected_prompts:
            used_prompts.add(prompt_text)

    keyword_candidates = [row.keyword_name for row in keyword_rows if row.keyword_name]

    base_rows = build_rows_for_brand(
        brand_index=brand_idx,
        brand_id=selected_client.id,
        brand=selected_client.name,
        prompt_entries=selected_prompts,
        phase=1,
        persona_id=ctx.persona_id,
        related_keywords=ctx.related_keywords,
        keyword_candidates=keyword_candidates,
    )

    context_summary = summarize_persona_summary(
        call_llm=call_llm,
        model=model,
        summary=persona.summary,
    )

    context_rows = build_context_rows_for_brand(
        brand_index=brand_idx,
        brand_id=selected_client.id,
        brand=selected_client.name,
        base_rows=base_rows,
        context_summary=context_summary,
        phase=1,
        persona_id=ctx.persona_id,
        related_keywords=ctx.related_keywords,
        keyword_candidates=keyword_candidates,
    )

    return base_rows, context_rows


def _run_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate prompts.csv from audience_habbits.csv"
    )
    parser.add_argument("--out", default="prompts.csv", help="Output CSV path")
    parser.add_argument(
        "--context-out",
        default=DEFAULT_CONTEXT_OUT,
        help="Output CSV path for context_prompts.csv (base + contextual prompt variants)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--n_per_category",
        type=int,
        default=10,
        help="Number of prompts to generate per category (Phase 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (only affects minor ordering decisions)",
    )
    parser.add_argument(
        "--persona_csv",
        "--persona-input",
        dest="persona_csv",
        default="audience_habbits.csv",
        help=(
            "Path to persona input (CSV or JSON). "
            "If audience_habbits.csv is missing, the loader falls back to nearby persona JSON files."
        ),
    )

    parser.add_argument(
        "--keyword-csv",
        default="keywords.csv",
        help="Optional keyword CSV (date, platform, keyword_name, cost, impressions, clicks, conversions).",
    )
    parser.add_argument(
        "--competitor-csv",
        type=Path,
        default=Path("competitors.csv"),
        help="Optional competitor CSV containing the brand/competitor names used to keep keywords non-branded.",
    )
    parser.add_argument(
        "--pmg-client-json",
        type=Path,
        default=DEFAULT_PMGCLIENT_JSON,
        help="Path to pmgclient JSON containing client metadata.",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Select the client by ID (matching pmgclient.json entries).",
    )
    parser.add_argument(
        "--client-name",
        type=str,
        default=None,
        help="Select the client by name (case-insensitive, matching pmgclient.json).",
    )
    parser.add_argument(
        "--client-slug",
        type=str,
        default=None,
        help="Select the client by slug (case-insensitive, matching pmgclient.json).",
    )
    parser.add_argument(
        "--prompts-per-persona",
        type=int,
        default=DEFAULT_PROMPTS_PER_PERSONA,
        help="Total prompts to keep per persona (sampled randomly from the generated set)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PROMPT_WORKERS", "24")),
        help="Number of concurrent personas to process (default: 24; can also set PROMPT_WORKERS env var)",
    )

    args = parser.parse_args(argv)
    args.n_per_category = max(1, int(args.n_per_category))
    prompts_per_persona = max(1, int(args.prompts_per_persona))
    random.seed(args.seed)

    try:
        clients = load_pmg_clients(args.pmg_client_json)
    except Exception as exc:
        print(
            f"[error] Could not load pmgclient.json from {args.pmg_client_json}: {exc}",
            file=sys.stderr,
        )
        return 2

    client_id_arg = args.client_id.strip() if args.client_id else None
    client_name_arg = args.client_name.strip() if args.client_name else None
    client_slug_arg = args.client_slug.strip() if args.client_slug else None

    try:
        selected_client = select_client(
            clients,
            client_id=client_id_arg,
            client_name=client_name_arg,
            client_slug=client_slug_arg,
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(
        f"[info] Using client '{selected_client.name}' ({selected_client.id}) for Phase 1 prompts"
    )

    competitor_csv_path = args.competitor_csv or Path("competitors.csv")
    brand_blocklist = load_competitor_brand_names(
        competitor_csv_path, selected_client.name
    )
    if competitor_csv_path.exists():
        print(
            f"[info] Derived {len(brand_blocklist)} brand terms from {competitor_csv_path} for keyword filtering"
        )
    else:
        print(
            f"[info] Competitor CSV not found at {competitor_csv_path}; filtering keywords using client name only"
        )

    try:
        persona_habits = load_persona_habits(args.persona_csv)
    except Exception as exc:
        print(
            f"[error] Could not load persona guidance from {args.persona_csv}: {exc}",
            file=sys.stderr,
        )
        return 2

    if not persona_habits:
        print(
            f"[error] No persona entries found in {args.persona_csv}", file=sys.stderr
        )
        return 2

    keyword_rows: List[KeywordRow] = []
    keyword_csv_arg = (args.keyword_csv or "").strip()
    if keyword_csv_arg:
        keyword_csv_path = Path(keyword_csv_arg)
        if keyword_csv_path.exists():
            try:
                keyword_rows = load_keywords_csv(keyword_csv_path)
                filtered_keywords = filter_keywords_without_brand_names(
                    keyword_rows, brand_blocklist
                )
                removed_count = len(keyword_rows) - len(filtered_keywords)
                keyword_rows = filtered_keywords
                if removed_count:
                    print(
                        f"[info] Removed {removed_count} branded keywords for '{selected_client.name}'"
                    )
                print(
                    f"[info] Loaded {len(keyword_rows)} keyword entries from {keyword_csv_path}"
                )
            except Exception as exc:
                print(
                    f"[warn] Failed to load keywords from {keyword_csv_path}: {exc}",
                    file=sys.stderr,
                )
        else:
            print(
                f"[info] Keyword CSV not found at {keyword_csv_path}; continuing without keywords."
            )

    try:
        call_llm = _get_openai_client()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    all_rows: List[Dict[str, str]] = []
    all_context_rows: List[Dict[str, str]] = []
    used_prompts: set[str] = set()
    used_prompts_lock = Lock()
    total_personas = len(persona_habits)
    max_workers = max(1, min(int(args.workers), total_personas))
    print(f"[info] Using up to {max_workers} worker(s) for prompt generation")
    prompts_target = total_personas * prompts_per_persona
    context_target = prompts_target
    base_prompts_generated = 0
    context_prompts_generated = 0

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for brand_idx, persona in enumerate(persona_habits, start=1):
            persona_label = persona.persona_id or f"persona-{brand_idx}"
            future = executor.submit(
                _process_persona,
                brand_idx,
                persona,
                persona_label,
                total_personas,
                selected_client,
                call_llm,
                args.model,
                args.n_per_category,
                prompts_per_persona,
                used_prompts,
                used_prompts_lock,
                keyword_rows,
            )
            futures[future] = persona_label

        for future in as_completed(futures):
            persona_label = futures[future]
            try:
                rows, context_rows = future.result()
            except Exception as e:
                print(
                    f"[warn] Failed for persona '{persona_label}' while targeting '{selected_client.name}': {e}",
                    file=sys.stderr,
                )
                continue

            if not rows and not context_rows:
                print(
                    f"[warn] No prompts produced for persona {persona_label}, skipping."
                )
                continue

            all_rows.extend(rows)
            all_context_rows.extend(context_rows)
            base_prompts_generated += len(rows)
            context_prompts_generated += len(context_rows)
            print_prompt_generation_progress(
                base_prompts_generated,
                prompts_target,
                context_prompts_generated,
                context_target,
            )

    if getattr(print_prompt_generation_progress, "has_printed", False):
        print()

    if not all_rows and not all_context_rows:
        print("[error] No prompts were generated.", file=sys.stderr)
        return 2

    try:
        write_prompts_csv(args.out, all_rows)
    except Exception as e:
        print(f"[error] Failed to write output CSV: {e}", file=sys.stderr)
        return 2

    try:
        write_prompts_csv(args.context_out, all_context_rows)
    except Exception as e:
        print(f"[error] Failed to write context output CSV: {e}", file=sys.stderr)
        return 2

    print(f"[done] Wrote {len(all_rows)} prompts to: {args.out}")
    print(f"[done] Wrote {len(all_context_rows)} prompts to: {args.context_out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    start = time.perf_counter()
    try:
        return _run_main(argv)
    finally:
        _USAGE_LOGGER.flush(time.perf_counter() - start)


if __name__ == "__main__":
    raise SystemExit(main())
