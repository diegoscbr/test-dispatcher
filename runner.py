#!/usr/bin/env python3
"""Clone the repo into a temp directory and move pipeline scripts to $HOME."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/diegoscbr/test-dispatcher.git"

FILES_TO_MOVE = [
    "prompts.py",
    "query.py",
    "follow_up_prompts.py",
    "phase2.py",
]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo_url = args[0] if args else DEFAULT_REPO_URL

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        repo_dir = tmp_dir / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        for name in FILES_TO_MOVE:
            if not (repo_dir / name).is_file():
                print(f"File not found in repo: {name}", file=sys.stderr)
                return 1

        home = Path.home()
        for name in FILES_TO_MOVE:
            src = repo_dir / name
            dest = home / name
            print(f"Found {name} at: {src}")
            shutil.move(str(src), str(dest))
            print(f"Moved {name} to: {dest}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
