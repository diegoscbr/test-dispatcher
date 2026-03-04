# Phase 1.5 Follow-Up Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate `follow_up_prompts.py` into `execution.py` as a conditional Phase 3 stage so the full pipeline runs end-to-end.

**Architecture:** `execution.py` imports `follow_up_prompts` alongside `prompts` and `query`. After `query.main()` completes, it reads `responses.csv` and calls `follow_up_prompts.detects_follow_up_question()` on each row. If any candidates are found, it runs `follow_up_prompts.main()`. The `model_usage_logger` dependency is removed from `follow_up_prompts.py` since it's unavailable at runtime.

**Tech Stack:** Python 3.9+, csv (stdlib), existing modules (prompts, query, follow_up_prompts)

**Design doc:** `docs/plans/2026-03-04-follow-up-integration-design.md`

---

### Task 1: Remove `model_usage_logger` from `follow_up_prompts.py`

**Files:**
- Modify: `follow_up_prompts.py`

**Step 1: Remove the import line**

Delete line 20:
```python
from model_usage_logger import ModelUsageLogger, extract_usage_from_response, extract_gemini_usage
```

**Step 2: Remove the logger initialization**

Delete line 85:
```python
_USAGE_LOGGER = ModelUsageLogger(Path(__file__), identifier_label="phase1.5")
```

**Step 3: Remove all `_USAGE_LOGGER.record(...)` blocks**

There are 11 blocks to remove. Each block is a `_USAGE_LOGGER.record(...)` call spanning 5-6 lines. Also remove the associated `duration = time.perf_counter() - start` line before each one (the `start = time.perf_counter()` line before the API call should also be removed since it's now unused).

Remove these timing + logging blocks at lines:
- 153-161 (Responses API classifier): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 165, 172-178 (chat.completions classifier): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 184, 191-197 (legacy SDK classifier): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 1107-1112 (call_openai Responses API): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 1126-1131 (call_openai chat.completions): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 1145-1150 (call_openai legacy): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 1185-1190 (call_openai_no_tools Responses API): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 1202-1207 (call_openai_no_tools chat.completions): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 1221-1226 (call_openai_no_tools legacy): remove `start =`, `duration =`, and `_USAGE_LOGGER.record(...)` block
- 1393-1401 (call_gemini): remove `usage = extract_gemini_usage(...)`, `duration =`, and `_USAGE_LOGGER.record(...)` block

**Step 4: Simplify `main()` to remove flush**

Replace:
```python
def main() -> None:
    start = time.perf_counter()
    try:
        _run_main()
    finally:
        _USAGE_LOGGER.flush(time.perf_counter() - start)
```

With:
```python
def main() -> None:
    _run_main()
```

**Step 5: Verify the module imports cleanly**

Run: `python -c "import follow_up_prompts; print('OK')"`
Expected: `OK` (no ImportError)

**Step 6: Commit**

```bash
git add follow_up_prompts.py
git commit -m "chore: remove model_usage_logger dependency from follow_up_prompts"
```

---

### Task 2: Add Phase 1.5 integration to `execution.py`

**Files:**
- Modify: `execution.py`

**Step 1: Add import and csv module**

At the top of `execution.py`, add `import csv` and `import follow_up_prompts` to the imports:

```python
#!/usr/bin/env python3
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import follow_up_prompts
import prompts
import query
```

**Step 2: Add the gate check function**

Add after `_wait_for_required_files`:

```python
def _has_follow_up_candidates(responses_path: Path) -> bool:
    """Scan responses.csv for rows where the model asked a follow-up question."""
    with open(responses_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if follow_up_prompts.detects_follow_up_question(row.get("response", "")):
                return True
    return False
```

**Step 3: Add the follow-up runner function**

Add after `_run_query_with_clean_argv`:

```python
def _run_followup_with_clean_argv() -> int:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return _run_entrypoint("follow_up_prompts.main()", follow_up_prompts.main)
    finally:
        sys.argv = original_argv
```

**Step 4: Update `main()` to include Phase 1.5**

Replace the current `main()`:

```python
def main() -> int:
    workdir = Path.cwd()
    prompts_code = _run_entrypoint("prompts.main()", prompts.main)
    if prompts_code != 0:
        print(f"[exec] prompts.main() failed with exit code {prompts_code}")
        return prompts_code

    _wait_for_required_files(workdir)
    query_code = _run_query_with_clean_argv()
    if query_code != 0:
        print(f"[exec] query.main() failed with exit code {query_code}")
        return query_code

    # Phase 1.5: conditional follow-up generation
    responses_path = workdir / "responses.csv"
    if not responses_path.is_file():
        print("[exec] responses.csv not found; skipping Phase 1.5")
        return 0

    if _has_follow_up_candidates(responses_path):
        print("[exec] Follow-up candidates detected; running Phase 1.5")
        followup_code = _run_followup_with_clean_argv()
        if followup_code != 0:
            print(f"[exec] follow_up_prompts.main() failed with exit code {followup_code}")
        return followup_code
    else:
        print("[exec] No follow-up candidates; skipping Phase 1.5")
        return 0
```

**Step 5: Verify the script loads**

Run: `python -c "import execution; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add execution.py
git commit -m "feat: integrate follow_up_prompts as conditional Phase 1.5 in pipeline"
```

---

### Task 3: Update CLAUDE.md to reflect new pipeline

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the architecture section**

Update the `execution.py` description to mention Phase 1.5:

> 3. **Orchestrator** (`execution.py`, ~80 lines) — Runs `prompts.main()` → polls for CSVs → `query.main()` → conditionally runs `follow_up_prompts.main()` if any responses contain follow-up questions. Entry point: `python execution.py`.

**Step 2: Update the data flow diagram**

```
audience_habbits.csv ──┐
pmgclient.json ────────┤
keywords.csv ──────────┼─→ prompts.py ─→ prompts.csv + context_prompts.csv ─→ query.py ─→ responses.csv
competitors.csv ───────┘                                                                       │
                                                                                               ▼
                                                                              follow_up_prompts.py (conditional)
                                                                                               │
                                                                                               ▼
                                                                                    Phase15_Responses.csv
```

**Step 3: Add Phase15_Responses.csv to CSV schemas section**

Add:
> **Phase15_Responses.csv**: All columns from `responses.csv` + `p1c_prompt`, `p1c_response`

**Step 4: Remove the "not called by execution.py" note from follow_up_prompts.py description**

Update the Additional Modules section to say it is now integrated.

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect Phase 1.5 integration"
```
