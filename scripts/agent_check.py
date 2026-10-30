"""Translate supported file-edit hook payloads into the shared fast checker."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    payload = json.load(sys.stdin)
    arguments = payload.get("tool_input", {})
    paths = []
    if isinstance(arguments, dict):
        if arguments.get("file_path"):
            paths.append(arguments["file_path"])
        patch = arguments.get("input", arguments.get("patch", ""))
    else:
        patch = arguments
    if isinstance(patch, str):
        for line in patch.splitlines():
            if line.startswith(("*** Add File: ", "*** Update File: ")):
                paths.append(line.split(": ", 1)[1])
    names = []
    for name in paths:
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        try:
            relative = path.resolve().relative_to(ROOT)
        except ValueError:
            continue
        if path.is_file():
            names.append(str(relative))
    if not names:
        return 0
    return subprocess.call(
        [sys.executable, str(ROOT / "scripts/check.py"), "--files", *names], cwd=ROOT
    )


if __name__ == "__main__":
    sys.exit(main())
