from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

LOG_DIR = Path(__file__).parent
FINAL_OUTPUT = LOG_DIR / "aivis_final.csv"
DATE_TEMPLATE = LOG_DIR / "aivis_{date}.csv"
WEIGHTS_INPUT = LOG_DIR / "weights.csv"
INTEGRATIONS_TOP50 = LOG_DIR / "consumer_ai_integrations_top50.csv"
INTEGRATIONS_MASTER = LOG_DIR / "consumer_ai_integrations.csv"
APPENDED_FIELDS = [
    "model_key",
    "rows",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "input_price_per_1m",
    "output_price_per_1m",
    "blended_price_per_1m",
    "input_cost",
    "output_cost",
    "total_cost",
]

PRICE_FIELD_SELECTION = [
    "timestamp",
    "identifier_label",
    "category",
    "model",
    "script_elapsed_seconds",
    "total_call_duration_seconds",
    "call_count",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "model_key",
    "rows",
    "total_input_tokens",
    "total_output_tokens",
    "input_price_per_1m",
    "output_price_per_1m",
    "blended_price_per_1m",
    "input_cost",
    "output_cost",
    "total_cost",
]


@dataclass
class Pricing:
    slug: str
    input_price: float
    output_price: float
    blended_price: float


def _normalize_model_name(name: str) -> str:
    if not name:
        return ""
    cleaned = name.strip().lower()
    cleaned = cleaned.replace(".", "-").replace("_", "-")
    cleaned = re.sub(r"[^a-z0-9-]", "", cleaned)
    return cleaned


def _parse_float(value: Optional[str]) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _load_pricing_file(path: Path) -> Dict[str, Pricing]:
    if not path.exists():
        return {}
    pricing_map: Dict[str, Pricing] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            slug_raw = row.get("slug") or row.get("model") or ""
            slug = _normalize_model_name(slug_raw)
            if not slug:
                continue
            pricing_map[slug] = Pricing(
                slug=slug,
                input_price=_parse_float(row.get("pricing_price_1m_input_tokens")),
                output_price=_parse_float(row.get("pricing_price_1m_output_tokens")),
                blended_price=_parse_float(row.get("pricing_price_1m_blended_3_to_1")),
            )
    return pricing_map


def _load_pricing() -> Dict[str, Pricing]:
    merged: Dict[str, Pricing] = {}
    for path in (INTEGRATIONS_MASTER, INTEGRATIONS_TOP50):
        merged.update(_load_pricing_file(path))
    return merged


def _match_pricing(model: str, pricing: Mapping[str, Pricing]) -> Optional[Pricing]:
    normalized = _normalize_model_name(model)
    if not normalized:
        return None
    if normalized in pricing:
        return pricing[normalized]
    for key, value in pricing.items():
        if key in normalized or normalized in key:
            return value
    return None


def _safe_int(value: Optional[str]) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _compute_cost(tokens: int, rate_per_million: float) -> float:
    return (tokens / 1_000_000.0) * rate_per_million


def _enrich_log_row(row: Dict[str, object], pricing: Mapping[str, Pricing]) -> Dict[str, object]:
    model = row.get("model") or ""
    normalized_model = _normalize_model_name(model)
    pricing_entry = _match_pricing(model, pricing)
    input_tokens = _safe_int(row.get("input_tokens"))
    output_tokens = _safe_int(row.get("output_tokens"))
    total_tokens = input_tokens + output_tokens
    input_price = pricing_entry.input_price if pricing_entry else 0.0
    output_price = pricing_entry.output_price if pricing_entry else 0.0
    blended_price = pricing_entry.blended_price if pricing_entry else 0.0
    input_rate = input_price or blended_price
    output_rate = output_price or blended_price
    input_cost = _compute_cost(input_tokens, input_rate)
    output_cost = _compute_cost(output_tokens, output_rate)
    row["model_key"] = normalized_model or "unknown"
    row["rows"] = 1
    row["total_input_tokens"] = input_tokens
    row["total_output_tokens"] = output_tokens
    row["total_tokens"] = total_tokens
    row["input_price_per_1m"] = input_price
    row["output_price_per_1m"] = output_price
    row["blended_price_per_1m"] = blended_price
    row["input_cost"] = input_cost
    row["output_cost"] = output_cost
    row["total_cost"] = input_cost + output_cost
    return row


def _collect_log_rows(pricing: Mapping[str, Pricing]) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    fieldnames: List[str] = []
    seen_fields: set[str] = set()
    for log_path in sorted(LOG_DIR.glob("*_log.csv")):
        with log_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for header in reader.fieldnames or []:
                if header not in seen_fields:
                    seen_fields.add(header)
                    fieldnames.append(header)
            for raw_row in reader:
                rows.append(_enrich_log_row(dict(raw_row), pricing))
    return rows, fieldnames


def _build_fieldnames(
    input_fieldnames: List[str], appended_fields: List[str], extra_fields: Optional[List[str]] = None
) -> List[str]:
    fields = list(input_fieldnames)
    for field in appended_fields:
        if field not in fields:
            fields.append(field)
    extra_fields = extra_fields or []
    for field in extra_fields:
        if field not in fields:
            fields.append(field)
    return fields


def _load_weights_rows() -> Tuple[List[Dict[str, object]], List[str]]:
    if not WEIGHTS_INPUT.exists():
        return [], []
    with WEIGHTS_INPUT.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        return [dict(row) for row in reader], fieldnames


def _write_csv(rows: Iterable[Dict[str, object]], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def main() -> None:
    pricing = _load_pricing()
    if pricing:
        print(f"Loaded pricing for {len(pricing)} models.")
    else:
        print("Warning: no pricing data found (consumer_ai_integrations files missing).")
    rows, input_fieldnames = _collect_log_rows(pricing)
    if not rows:
        print("No *_log.csv files found; nothing to write.")
        return
    base_rows, base_fieldnames = _load_weights_rows()
    today = datetime.now().strftime("%m%d%Y")
    alternate_path = Path(str(DATE_TEMPLATE).replace("{date}", today))
    appended_fields = [field for field in APPENDED_FIELDS if field not in input_fieldnames]
    price_fieldnames = [
        field
        for field in PRICE_FIELD_SELECTION
        if field in input_fieldnames or field in appended_fields
    ]
    price_prefixed_fieldnames = [f"price_{field}" for field in price_fieldnames]
    trimmed_base_fieldnames = [f for f in base_fieldnames if f not in price_fieldnames]
    final_fieldnames = trimmed_base_fieldnames + price_prefixed_fieldnames
    merged_rows: List[Dict[str, object]] = []
    total_rows = max(len(base_rows), len(rows))
    for idx in range(total_rows):
        merged: Dict[str, object] = {}
        if idx < len(base_rows):
            base_row = base_rows[idx]
            for field in trimmed_base_fieldnames:
                merged[field] = base_row.get(field, "")
        if idx < len(rows):
            price_row = rows[idx]
            for field in price_fieldnames:
                merged[f"price_{field}"] = price_row.get(field, "")
        else:
            for prefixed in price_prefixed_fieldnames:
                if prefixed not in merged:
                    merged[prefixed] = ""
        merged_rows.append(merged)
    _write_csv(merged_rows, FINAL_OUTPUT, final_fieldnames)
    _write_csv(merged_rows, alternate_path, final_fieldnames)
    total_cost = sum(float(row.get("total_cost") or 0.0) for row in rows)
    print(
        f"Wrote {len(rows)} log entries to {FINAL_OUTPUT.name} and {alternate_path.name} "
        f"(total cost ${total_cost:.4f})"
    )


if __name__ == "__main__":
    main()

