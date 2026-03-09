#!/usr/bin/env python3
"""Orchestrate the models enrichment pipeline.

Execution order:
  1. model_use.main()          → consumer_ai_integrations.csv
  2. key_builder.main()        → key.csv
  3. enrich_responses.main()   → phase2_responses_enriched.csv  (requires phase2_responses.csv)
"""

import importlib
import sys
from pathlib import Path
from typing import Callable

PIPELINE_MODULES = [
    "model_use",
    "key_builder",
    "enrich_responses",
]

REQUIRED_AFTER_STEP = {
    "model_use": ["consumer_ai_integrations.csv"],
    "key_builder": ["key.csv"],
}


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


def _run_with_clean_argv(label: str, entrypoint: Callable[[], object]) -> int:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
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


def main() -> int:
    workdir = Path(__file__).resolve().parent

    # Verify all pipeline modules exist
    missing_modules = [m for m in PIPELINE_MODULES if not (workdir / f"{m}.py").is_file()]
    if missing_modules:
        print(f"[exec] Missing pipeline modules: {', '.join(f'{m}.py' for m in missing_modules)}")
        return 1

    # Ensure workdir is on sys.path so imports resolve
    workdir_str = str(workdir)
    if workdir_str not in sys.path:
        sys.path.insert(0, workdir_str)

    # Step 1: Fetch AA model catalog
    model_use = importlib.import_module("model_use")
    code = _run_with_clean_argv("model_use.main()", model_use.main)
    if code != 0:
        print(f"[exec] model_use.main() failed with exit code {code}")
        return code
    if not _verify_outputs(workdir, "model_use"):
        return 1

    # Step 2: Build provider-slug → AA-slug mapping
    key_builder = importlib.import_module("key_builder")
    code = _run_with_clean_argv("key_builder.main()", key_builder.main)
    if code != 0:
        print(f"[exec] key_builder.main() failed with exit code {code}")
        return code
    if not _verify_outputs(workdir, "key_builder"):
        return 1

    # Step 3: Enrich phase2 responses (conditional on input file)
    phase2_input = workdir / "phase2_responses.csv"
    if not phase2_input.is_file():
        print("[exec] phase2_responses.csv not found; skipping enrichment")
        print("[exec] Pipeline complete (steps 1-2 only)")
        return 0

    enrich_responses = importlib.import_module("enrich_responses")
    code = _run_with_clean_argv("enrich_responses.main()", enrich_responses.main)
    if code != 0:
        print(f"[exec] enrich_responses.main() failed with exit code {code}")
        return code

    enriched = workdir / "phase2_responses_enriched.csv"
    if not enriched.is_file():
        print("[exec] phase2_responses_enriched.csv was not created")
        return 1

    print("[exec] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
