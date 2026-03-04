#!/usr/bin/env python3
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import follow_up_prompts
import phase2
import prompts
import query
POLL_INTERVAL_SECONDS = 2
REQUIRED_FILES = ("prompts.csv", "context_prompts.csv")


def _normalize_exit_code(code: object) -> int:
    if isinstance(code, int):
        return code
    return 0 if code is None else 1


def _run_entrypoint(label: str, entrypoint: Callable[[], object]) -> int:
    print(f"[exec] Running {label}")
    try:
        result = entrypoint()
    except SystemExit as exc:
        return _normalize_exit_code(exc.code)
    return _normalize_exit_code(result)


def _wait_for_required_files(base_dir: Path) -> None:
    print(f"[exec] Watching for {', '.join(REQUIRED_FILES)} in {base_dir}")
    while True:
        missing = [name for name in REQUIRED_FILES if not (base_dir / name).is_file()]
        if not missing:
            print("[exec] Required files found.")
            return
        print(f"[exec] Waiting for files: {', '.join(missing)}")
        time.sleep(POLL_INTERVAL_SECONDS)

def _has_follow_up_candidates(response_path: Path) -> bool:
    with open(response_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if follow_up_prompts.detects_follow_up_question(row.get("response", "")):
                return True
    return False

def _run_query_with_clean_argv() -> int:
    original_argv = sys.argv[:]
    try:
        # query.main() uses argparse against sys.argv; clear prompts-only args.
        sys.argv = [original_argv[0]]
        return _run_entrypoint("query.main()", query.main)
    finally:
        sys.argv = original_argv

def _run_followup_with_clean_argv() -> int:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return _run_entrypoint("follow_up_prompts.main()", follow_up_prompts.main)
    finally:
        sys.argv = original_argv

def _run_phase2_with_clean_argv() -> int:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return _run_entrypoint("phase2.main()", phase2.main)
    finally:
        sys.argv = original_argv

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

    # Phase 2 (always runs)
    phase2_code = _run_phase2_with_clean_argv()
    if phase2_code != 0:
        print(f"[exec] phase2.main() failed with exit code {phase2_code}")
    return phase2_code

if __name__ == "__main__":
    raise SystemExit(main())
