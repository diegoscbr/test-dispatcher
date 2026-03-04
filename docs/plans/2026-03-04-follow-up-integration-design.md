# Phase 1.5 Follow-Up Integration Design

**Date:** 2026-03-04

## Goal

Integrate `follow_up_prompts.py` into `execution.py` as a conditional Phase 3 stage, so the full pipeline runs end-to-end without manual intervention.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Trigger mode | Conditional | Only run Phase 1.5 if `responses.csv` contains rows where the model asked clarifying questions instead of answering |
| Integration method | Direct import | Consistent with how `prompts` and `query` are imported in `execution.py` |
| Output format | Separate file | Keep `Phase15_Responses.csv` alongside `responses.csv` (current standalone behavior) |
| Error handling | Hard failure | Non-zero exit if Phase 1.5 fails |
| `model_usage_logger` | Remove | Strip all usage logging; not available in runtime |

## Pipeline After Integration

```
execution.py
  1. prompts.main()          → prompts.csv + context_prompts.csv
  2. query.main()            → responses.csv
  3. follow_up_prompts.main() → Phase15_Responses.csv  (conditional)
```

## Changes

### `follow_up_prompts.py`

- Remove `from model_usage_logger import ...` (line 20)
- Remove `_USAGE_LOGGER = ModelUsageLogger(...)` (line 85)
- Remove all `_USAGE_LOGGER.record(...)` calls (13 occurrences)
- Remove `_USAGE_LOGGER.flush(...)` in `main()` (line 2038)
- Remove `extract_usage_from_response` and `extract_gemini_usage` references

### `execution.py`

- Add `import follow_up_prompts` alongside existing imports
- Add `_has_follow_up_candidates(path)` — reads `responses.csv` with `csv.DictReader`, calls `follow_up_prompts.detects_follow_up_question()` on each row's `response` field, returns `True` on first match
- Add `_run_followup_with_clean_argv()` — same pattern as `_run_query_with_clean_argv()`, cleans `sys.argv` then calls `follow_up_prompts.main()`
- Update `main()` to: after `query.main()` succeeds, wait for `responses.csv`, run gate check, conditionally run Phase 1.5
- Add `"responses.csv"` awareness (wait for it before gate check)

### Updated `main()` pseudocode

```python
def main():
    workdir = Path.cwd()

    # Stage 1
    prompts_code = _run_entrypoint("prompts.main()", prompts.main)
    if prompts_code != 0:
        return prompts_code

    _wait_for_required_files(workdir)  # prompts.csv, context_prompts.csv

    # Stage 2
    query_code = _run_query_with_clean_argv()
    if query_code != 0:
        return query_code

    # Stage 3 (conditional)
    responses_path = workdir / "responses.csv"
    if not responses_path.is_file():
        print("[exec] responses.csv not found; skipping Phase 1.5")
        return 0

    if _has_follow_up_candidates(responses_path):
        print("[exec] Follow-up candidates detected; running Phase 1.5")
        return _run_followup_with_clean_argv()
    else:
        print("[exec] No follow-up candidates; skipping Phase 1.5")
        return 0
```

### Gate check function

```python
def _has_follow_up_candidates(responses_path: Path) -> bool:
    import csv
    with open(responses_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if follow_up_prompts.detects_follow_up_question(row.get("response", "")):
                return True
    return False
```

## Inputs/Outputs Summary

### Inputs to Phase 1.5

| File | Source | Required |
|---|---|---|
| `responses.csv` | Output of `query.py` | Yes |
| `audience_habbits.csv` | Same input used by `prompts.py` | Yes |
| `pmgclient.json` | Same client metadata | Optional |

### Output of Phase 1.5

| File | Schema |
|---|---|
| `Phase15_Responses.csv` | All columns from `responses.csv` + `p1c_prompt` + `p1c_response` |

## Risk

- `_load_env()` runs at import time in `follow_up_prompts.py` (line 50). This is a side effect but matches `query.py` behavior and should not cause issues.
- The gate check in `execution.py` partially duplicates the gate in `_run_main()`. This is acceptable — the outer gate is a fast short-circuit to skip the import/setup overhead when there are no candidates at all.
