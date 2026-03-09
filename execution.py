#!/usr/bin/env python3
import csv
import importlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

POLL_INTERVAL_SECONDS = 2
REQUIRED_FILES = ("prompts.csv", "context_prompts.csv")

DEFAULT_REPO_URL = "https://github.com/diegoscbr/test-dispatcher.git"
PIPELINE_MODULES = [
    "prompts",
    "query",
    "follow_up_prompts",
    "phase2",
]


def _bootstrap_from_repo(repo_url: str, dest: Path) -> None:
    """Clone the repo and move pipeline scripts to *dest*."""
    files_to_move = [f"{m}.py" for m in PIPELINE_MODULES]
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        repo_dir = tmp_dir / "repo"
        print(f"[exec] Cloning {repo_url} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        for name in files_to_move:
            src = repo_dir / name
            if not src.is_file():
                raise FileNotFoundError(f"File not found in repo: {name}")
            target = dest / name
            print(f"[exec] {name} -> {target}")
            shutil.move(str(src), str(target))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("[exec] Bootstrap complete.")


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


def _run_with_clean_argv(label: str, entrypoint: Callable[[], object]) -> int:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        return _run_entrypoint(label, entrypoint)
    finally:
        sys.argv = original_argv


def main() -> int:
    workdir = Path.cwd()

    # --- Bootstrap: clone repo and move scripts if not already present ---
    missing_modules = [m for m in PIPELINE_MODULES if not (workdir / f"{m}.py").is_file()]
    if missing_modules:
        repo_url = DEFAULT_REPO_URL
        for a in sys.argv[1:]:
            if a.startswith(("https://", "http://", "git@")):
                repo_url = a
                break
        _bootstrap_from_repo(repo_url, workdir)

    # --- Lazy-import pipeline modules (available only after bootstrap) ---
    prompts = importlib.import_module("prompts")
    query = importlib.import_module("query")
    follow_up_prompts = importlib.import_module("follow_up_prompts")
    phase2 = importlib.import_module("phase2")

    # Phase 1: prompt generation
    prompts_code = _run_entrypoint("prompts.main()", prompts.main)
    if prompts_code != 0:
        print(f"[exec] prompts.main() failed with exit code {prompts_code}")
        return prompts_code

    _wait_for_required_files(workdir)

    # Phase 1: query execution
    query_code = _run_with_clean_argv("query.main()", query.main)
    if query_code != 0:
        print(f"[exec] query.main() failed with exit code {query_code}")
        return query_code

    # Phase 1.5: follow-up prompts (conditional)
    responses_path = workdir / "responses.csv"
    if not responses_path.is_file():
        print("[exec] responses.csv not found; skipping Phase 1.5")
        return 0

    def _has_follow_up_candidates() -> bool:
        with open(responses_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if follow_up_prompts.detects_follow_up_question(row.get("response", "")):
                    return True
        return False

    if _has_follow_up_candidates():
        print("[exec] Follow-up candidates detected; running Phase 1.5")
        followup_code = _run_with_clean_argv("follow_up_prompts.main()", follow_up_prompts.main)
        if followup_code != 0:
            print(f"[exec] follow_up_prompts.main() failed with exit code {followup_code}")
            return followup_code
    else:
        print("[exec] No follow-up candidates; skipping Phase 1.5")

    # Phase 2 (always runs)
    phase2_code = _run_with_clean_argv("phase2.main()", phase2.main)
    if phase2_code != 0:
        print(f"[exec] phase2.main() failed with exit code {phase2_code}")
    return phase2_code


if __name__ == "__main__":
    raise SystemExit(main())
