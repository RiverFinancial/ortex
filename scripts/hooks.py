"""Install a worktree-aware hook without changing core.hooksPath."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace a different existing hook")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    hook = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--git-path", "hooks/pre-commit"], cwd=root, text=True
        ).strip()
    )
    if not hook.is_absolute():
        hook = root / hook
    content = '#!/bin/sh\nroot=$(git rev-parse --show-toplevel) || exit 1\ntest -f "$root/scripts/precommit" || exit 0\nexec sh "$root/scripts/precommit" "$@"\n'
    if hook.exists() and hook.read_text() != content and not args.force:
        raise SystemExit(
            f"Existing hook left intact: {shlex.quote(str(hook))}; inspect it before using --force"
        )
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(content)
    hook.chmod(hook.stat().st_mode | 0o111)
    print(f"Installed {os.fspath(hook)}; linked worktrees share this hook")


if __name__ == "__main__":
    main()
