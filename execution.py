#!/usr/bin/env python3
import sys
import time
from pathlib import Path
from typing import Callable

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


def _run_query_with_clean_argv() -> int:
    original_argv = sys.argv[:]
    try:
        # query.main() uses argparse against sys.argv; clear prompts-only args.
        sys.argv = [original_argv[0]]
        return _run_entrypoint("query.main()", query.main)
    finally:
        sys.argv = original_argv


def main() -> int:
    workdir = Path.cwd()
    prompts_code = _run_entrypoint("prompts.main()", prompts.main)
    if prompts_code != 0:
        print(f"[exec] prompts.main() failed with exit code {prompts_code}")
        return prompts_code

    _wait_for_required_files(workdir)
    return _run_query_with_clean_argv()


if __name__ == "__main__":
    raise SystemExit(main())
