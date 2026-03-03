#!/usr/bin/env python3
import sys
from typing import Iterable, Optional

import prompts


def run(args: Optional[Iterable[str]] = None) -> int:
    arg_list = list(args or [])
    print(f"[exec] prompts.main({arg_list!r})")
    try:
        result = prompts.main(arg_list)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 0 if exc.code is None else 1
    return result if isinstance(result, int) else 0


def main() -> int:
    return run(args=sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
