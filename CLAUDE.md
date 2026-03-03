# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **GEO/AI-visibility test dispatcher** — a pipeline that generates search prompts for brand visibility testing, dispatches them to LLM APIs, and collects responses. It measures how brands appear in AI-generated search results across different personas and funnel stages.

## Architecture

The pipeline has three stages orchestrated by `execution.py`:

1. **Prompt Generation** (`prompts.py`) — Reads persona data from `audience_habbits.csv` (and optionally client metadata from `pmgclient.json`), calls OpenAI to generate search prompts per brand/persona/funnel-stage, and writes `prompts.csv` + `context_prompts.csv`.

2. **Query Execution** (`query.py`) — Reads `prompts.csv`, sends each prompt to one or more LLM models via OpenRouter-compatible APIs, and writes `responses.csv`. Supports concurrent batching, multiple models, Gemini grounding URL resolution, and web-search-enabled models.

3. **Orchestrator** (`execution.py`) — Runs `prompts.main()` then polls for the required CSV files before running `query.main()`. Entry point: `python execution.py`.

### Shell Scripts

- `runner.sh` — Clones this repo into a temp dir and moves `prompts.py` + `query.py` to `$HOME` (used for deployment/dispatcher environments).
- `show_outputs.sh` — Prints contents of `prompts.csv`, `context_prompts.csv`, and `responses.csv` for quick inspection.

## Running

```bash
# Full pipeline (generates prompts then queries models)
python execution.py

# Prompt generation only
python prompts.py --persona_csv audience_habbits.csv --out prompts.csv

# Query execution only
python query.py --prompts prompts.csv --out responses.csv

# Inspect outputs
bash show_outputs.sh
```

## Key CLI Arguments

**prompts.py**: `--persona_csv`, `--out`, `--model` (default: gpt-4o-mini), `--n_per_category`, `--context_out`

**query.py**: `--prompts`, `--out`, `--models` (comma-separated), `--workers`, `--batch-size`, `--max-tokens`, `--temperature`, `--timeout`, `--sleep`, `--max-rows`

## Environment Variables

- `OPENAI_API_KEY` — Required for both prompt generation and query execution
- `SERPAPI_API_KEY` / `GOOGLE_SEARCH_API_KEY` — Optional, for web search grounding
- Supports `.env` file in project root (auto-loaded by `query.py`)

## Data Flow

```
audience_habbits.csv ─→ prompts.py ─→ prompts.csv + context_prompts.csv ─→ query.py ─→ responses.csv
```

## CSV Schemas

**prompts.csv**: `brand_id, brand, prompt_id, prompt, category, phase, keywords, funnel`

**responses.csv**: `persona_id, brand_id, brand, prompt_id, prompt, category, type, response, model`

## Python Version

Uses Python 3.9+ (system Python on macOS). Dependencies: `pandas`, `openai`, `requests`.
