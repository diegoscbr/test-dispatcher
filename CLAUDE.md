# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **GEO/AI-visibility test dispatcher** — a pipeline that generates search prompts for brand visibility testing, dispatches them to LLM APIs, and collects responses. It measures how brands appear in AI-generated search results across different personas and funnel stages.

## Architecture

The pipeline has four stages orchestrated by `execution.py`:

1. **Prompt Generation** (`prompts.py`, ~2400 lines) — Reads persona data from `audience_habbits.csv` (and optionally client metadata from `pmgclient.json`, `keywords.csv`, `competitors.csv`), calls OpenAI to generate search prompts per brand/persona/funnel-stage across 7 categories (Learn & Understand, Recommendations, Compare & Decide, Price & Value, How-To & Setup, Fix & Troubleshoot, Reviews & Social Proof), and writes `prompts.csv` + `context_prompts.csv`.

2. **Query Execution** (`query.py`, ~1350 lines) — Reads `prompts.csv`, sends each prompt to one or more LLM models (default: `gpt-5.2-chat-latest`, `gemini-2.0-flash`), and writes `responses.csv`. Supports concurrent batching, multiple models, Gemini grounding URL resolution, web-search-enabled models (OpenAI Responses API `web_search` tool and Gemini `google_search` tool), and text normalization (mojibake repair, Unicode NFKC normalization, smart punctuation replacement).

3. **Follow-Up Prompts** (`follow_up_prompts.py`, ~2000 lines) — Phase 1.5: conditionally runs if `responses.csv` contains follow-up candidates. Generates follow-up prompts and writes `Phase15_Responses.csv`.

4. **Phase 2 Deep Conversations** (`phase2.py`, ~1320 lines) — Always runs. Reads `Phase15_Responses.csv` (falls back to `responses.csv` if unavailable), generates a next user message per row, dispatches it to the original model, and writes `phase2_responses.csv`.

5. **Orchestrator** (`execution.py`, ~100 lines) — Runs stages 1–4 sequentially, polling for required CSV files between stages. Entry point: `python execution.py`.

### Shell Scripts

- `runner.sh` — Clones this repo into a temp dir and moves `prompts.py` + `query.py` to `$HOME` (used for deployment/dispatcher environments).
- `show_outputs.sh` — Prints contents of `prompts.csv`, `context_prompts.csv`, `responses.csv`, `Phase15_Responses.csv`, and `phase2_responses.csv` for quick inspection.

## Running

```bash
# Full pipeline (generates prompts then queries models)
python execution.py

# Prompt generation only
python prompts.py --persona_csv audience_habbits.csv --out prompts.csv

# Query execution only
python query.py --prompts prompts.csv --out responses.csv

# Multiple models with custom workers
python query.py --prompts prompts.csv --out responses.csv --models "gpt-4o,gemini-2.0-flash" --workers 2

# Inspect outputs
bash show_outputs.sh
```

## Key CLI Arguments

**prompts.py**: `--persona_csv`, `--out`, `--context-out`, `--model` (default: gpt-4o-mini), `--n_per_category` (default: 10), `--prompts-per-persona` (default: 10), `--workers` (default: 24 via `PROMPT_WORKERS` env), `--seed` (default: 42), `--keyword-csv`, `--competitor-csv`, `--pmg-client-json`, `--client-id`, `--client-name`, `--client-slug`

**query.py**: `--prompts`, `--out`, `--models` (comma-separated), `--model`, `--context-prompts`, `--workers` (default: 4), `--batch-size` (default: 1), `--max-tokens` (default: 1200), `--temperature` (default: 0.2), `--timeout` (default: 60), `--sleep` (default: 0.0), `--max-rows`, `--gemini-search` (auto/force/off), `--openai-api-key`, `--gemini-api-key`

**phase2.py**: `--phase15-responses` (default: Phase15_Responses.csv), `--phase1-responses` (default: responses.csv), `--audience` (default: audience_habbits.csv), `--out` (default: phase2_responses.csv), `--model` (default: gpt-5.2-chat-latest), `--temperature` (default: 0.2), `--max-tokens` (default: 1200), `--system-prompt`, `--openai-api-key`, `--gemini-api-key`, `--timeout` (default: 60), `--max-workers` (default: 24), `--openai-rpm` (default: 60), `--gemini-rpm` (default: 60), `--max-retries` (default: 6)

## Environment Variables

- `OPENAI_API_KEY` — Required for both prompt generation and query execution
- `GEMINI_API_KEY` (fallback: `GEMINIAPI_KEY`) — Required for Gemini models
- `SERPAPI_API_KEY` — Optional, for web search grounding via SerpAPI
- `GOOGLE_SEARCH_API_KEY` / `GOOGLE_SEARCH_ENGINE_ID` — Optional, for Google Custom Search grounding
- `GEMINI_API_BASE` — Custom Gemini endpoint (default: `https://generativelanguage.googleapis.com/v1beta/models`)
- `OPENAI_MODEL` — Override default model in query.py
- `PROMPT_WORKERS` — Override default concurrent workers in prompts.py (default: 24)
- Supports `.env` file in project root (auto-loaded by both `prompts.py` and `query.py`)

## Data Flow

```
audience_habbits.csv ──┐
pmgclient.json ────────┤
keywords.csv ──────────┼─→ prompts.py ─→ prompts.csv + context_prompts.csv ─→ query.py ─→ responses.csv
competitors.csv ───────┘                                                                       │
                                                                          follow_up_prompts.py ←┘ (conditional)
                                                                                       │
                                                                              Phase15_Responses.csv
                                                                                       │
                                                                              phase2.py ←┘ (always; falls back to responses.csv)
                                                                                       │
                                                                              phase2_responses.csv
```

## CSV Schemas

**prompts.csv**: `persona_id, brand_id, brand, prompt_id, prompt, category, keywords, funnel`

**context_prompts.csv**: Same schema as prompts.csv (base prompt + contextualized variant per prompt)

**responses.csv**: `persona_id, brand_id, brand, prompt_id, prompt, category, type, response, model`

**Phase15_Responses.csv**: Same schema as responses.csv with follow-up prompt/response columns added.

**phase2_responses.csv**: Inherits input columns from Phase 1.5 (or Phase 1) plus `p2_prompt, p2_response, p2_category`.

**Prompt ID format**: `b{brand_index:04d}_p{phase}_q{prompt_counter:04d}_{persona_suffix}` (persona_suffix = first 8 chars of persona_id or "base")

## Key Implementation Details

- **Dual API support**: OpenAI uses the Responses API with `web_search` tool; Gemini uses the Generative Language API with `google_search` tool. Model routing is determined by `is_gemini_model()` in query.py.
- **Gemini model resolution**: Old aliases (gemini-1.5-*) fall back to `gemini-2.0-flash`; current models (gemini-2.0-flash, gemini-2.5-pro) pass through unchanged.
- **Text normalization**: query.py applies a multi-step pipeline — mojibake repair (latin1↔utf-8), NFKC normalization, smart punctuation replacement, zero-width character removal, whitespace collapsing.
- **Citation rendering**: Both APIs extract sources and render numbered `[n]` citation markers with a Sources section appended to responses.
- **Funnel stage detection**: Uses keyword matching against hardcoded word lists (`FUNNEL_CONVERSION_KEYWORDS`, `FUNNEL_CONSIDERATION_KEYWORDS`) to classify prompts into Awareness/Consideration/Conversion stages.
- **No test suite**: There are currently no automated tests in this repository.

## Python Version

Uses Python 3.9+ (system Python on macOS). Dependencies: `pandas`, `openai`, `requests`.
