import csv
import os
import sys
from pathlib import Path

import requests

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
ENV_PATH = Path(__file__).resolve().parent / ".env"


def fetch_models(api_key):
    resp = requests.get(API_URL, headers={"x-api-key": api_key})
    resp.raise_for_status()
    return resp.json()["data"]


def flatten_model(model):
    row = {
        "id": model.get("id"),
        "name": model.get("name"),
        "slug": model.get("slug"),
        "release_date": model.get("release_date"),
        "creator_id": (model.get("model_creator") or {}).get("id"),
        "creator_name": (model.get("model_creator") or {}).get("name"),
        "creator_slug": (model.get("model_creator") or {}).get("slug"),
    }
    for key, val in (model.get("evaluations") or {}).items():
        row[f"eval_{key}"] = val
    for key, val in (model.get("pricing") or {}).items():
        row[f"pricing_{key}"] = val
    row["median_output_tokens_per_second"] = model.get(
        "median_output_tokens_per_second"
    )
    row["median_time_to_first_token_seconds"] = model.get(
        "median_time_to_first_token_seconds"
    )
    row["median_time_to_first_answer_token"] = model.get(
        "median_time_to_first_answer_token"
    )
    return row


def main():
    _load_dotenv()
    api_key = os.environ.get("AI_ANALYSIS_KEY")
    if not api_key:
        print("Error: set AI_ANALYSIS_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    models = fetch_models(api_key)
    rows = [flatten_model(m) for m in models]

    # Collect all columns across all models (evaluations vary per model)
    all_keys = list(dict.fromkeys(k for row in rows for k in row))

    with open(INPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    _write_top50()


INPUT_FILE = "consumer_ai_integrations.csv"
OUTPUT_FILE = "consumer_ai_integrations_top50.csv"

DROP_COLUMNS = {
    "eval_artificial_analysis_coding_index",
    "eval_artificial_analysis_math_index",
    "eval_mmlu_pro",
    "eval_gpqa",
    "eval_hle",
    "eval_livecodebench",
    "eval_scicode",
    "eval_math_500",
    "eval_aime",
    "eval_aime_25",
    "eval_ifbench",
    "eval_lcr",
    "eval_terminalbench_hard",
    "eval_tau2",
}

SORT_KEY = "eval_artificial_analysis_intelligence_index"


def _write_top50():
    with open(INPUT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get(SORT_KEY)]
    rows.sort(key=lambda r: float(r[SORT_KEY]), reverse=True)
    rows = rows[:50]

    if not rows:
        return

    fieldnames = [k for k in rows[0].keys() if k not in DROP_COLUMNS]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


if __name__ == "__main__":
    main()
