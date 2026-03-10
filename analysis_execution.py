#!/usr/bin/env python3
"""Orchestrate the analysis / scoring / pricing pipeline.

Usage:
  python analysis_execution.py --lob "insurance"
  python analysis_execution.py --lob "retail" https://github.com/diegoscbr/test-dispatcher.git

Execution order:
  1. analysis.main()   -> analysis.csv  (requires phase2_responses_enriched.csv)
  2. weights.main()    -> weights.csv   (requires analysis.csv)
  3. price.main()      -> aivis_final.csv (requires weights.csv + *_log.csv)

The --lob argument is forwarded to analysis.py only; it is stripped
before weights and price run so their argparse is not polluted.
"""

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List

DEFAULT_REPO_URL = "https://github.com/diegoscbr/test-dispatcher.git"

PIPELINE_MODULES = [
    "analysis",
    "weights",
    "price",
]

EXTRA_FILES: list[str] = []

REQUIRED_AFTER_STEP = {
    "analysis": ["analysis.csv"],
    "weights": ["weights.csv"],
}


def _bootstrap_from_repo(repo_url: str, dest: Path) -> None:
    """Clone the repo and move pipeline scripts to *dest*."""
    files_to_move = [f"{m}.py" for m in PIPELINE_MODULES] + EXTRA_FILES
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
                print(f"[exec] Warning: {name} not found in repo, skipping")
                continue
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


def _run_with_argv(label: str, entrypoint: Callable[[], object], extra_args: List[str] = None) -> int:
    """Run *entrypoint* with a controlled sys.argv.

    Only *extra_args* (if any) are forwarded; all other CLI noise from the
    execution wrapper is stripped so downstream argparse sees a clean slate.
    """
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] + (extra_args or [])
        return _run_entrypoint(label, entrypoint)
    finally:
        sys.argv = original_argv


def _verify_outputs(workdir: Path, module_name: str) -> bool:
    expected = REQUIRED_AFTER_STEP.get(module_name, [])
    missing = [f for f in expected if not (workdir / f).is_file()]
    if missing:
        print(f"[exec] Missing expected outputs: {', '.join(missing)}")
        return False
    return True


def _parse_execution_args() -> argparse.Namespace:
    """Parse args that belong to the execution wrapper itself.

    Only --lob is recognised as a named flag.  A repo URL, if needed, is
    detected by scanning sys.argv for anything that looks like a URL — the
    same approach models_execution.py uses.  No positional arg is declared
    so argparse cannot accidentally swallow the --lob value.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis -> weights -> price pipeline.",
        add_help=False,
    )
    parser.add_argument("--lob", default=None, help="LOB regex forwarded to analysis.py --lob")
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    exec_args = _parse_execution_args()
    workdir = Path(__file__).resolve().parent

    # Bootstrap: clone repo and move scripts if not already present
    all_needed = [f"{m}.py" for m in PIPELINE_MODULES] + EXTRA_FILES
    missing_files = [f for f in all_needed if not (workdir / f).is_file()]
    if missing_files:
        repo_url = DEFAULT_REPO_URL
        for a in sys.argv[1:]:
            if a.startswith(("https://", "http://", "git@")):
                repo_url = a
                break
        _bootstrap_from_repo(repo_url, workdir)

    # Ensure workdir is on sys.path so imports resolve
    workdir_str = str(workdir)
    if workdir_str not in sys.path:
        sys.path.insert(0, workdir_str)

    # Build the args list forwarded ONLY to analysis.py
    analysis_args: List[str] = []
    if exec_args.lob:
        analysis_args.extend(["--lob", exec_args.lob])
        print(f"[exec] --lob {exec_args.lob!r} will be forwarded to analysis")

    # Step 1: Analysis (requires phase2_responses_enriched.csv)
    enriched_input = workdir / "phase2_responses_enriched.csv"
    if not enriched_input.is_file():
        print("[exec] phase2_responses_enriched.csv not found; cannot run analysis")
        return 1

    analysis = importlib.import_module("analysis")
    code = _run_with_argv("analysis.main()", analysis.main, extra_args=analysis_args)
    if code != 0:
        print(f"[exec] analysis.main() failed with exit code {code}")
        return code
    if not _verify_outputs(workdir, "analysis"):
        return 1

    # Step 2: Weights (requires analysis.csv) — no extra args
    weights = importlib.import_module("weights")
    code = _run_with_argv("weights.main()", weights.main)
    if code != 0:
        print(f"[exec] weights.main() failed with exit code {code}")
        return code
    if not _verify_outputs(workdir, "weights"):
        return 1

    # Step 3: Price (requires weights.csv + *_log.csv) — no extra args
    log_files = list(workdir.glob("*_log.csv"))
    if not log_files:
        print("[exec] No *_log.csv files found; skipping price step")
        print("[exec] Pipeline complete (steps 1-2 only)")
        return 0

    price = importlib.import_module("price")
    code = _run_with_argv("price.main()", price.main)
    if code != 0:
        print(f"[exec] price.main() failed with exit code {code}")
        return code

    final_output = workdir / "aivis_final.csv"
    if not final_output.is_file():
        print("[exec] aivis_final.csv was not created")
        return 1

    print("[exec] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
