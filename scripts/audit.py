"""Report Rust advisories and retired Hex dependencies without hiding failures."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "dist/reports"
    destination.mkdir(parents=True, exist_ok=True)
    sections = []
    failed = False
    for name, command in [
        (
            "cargo-audit",
            [
                "cargo",
                "audit",
                "--file",
                "native/ortex/Cargo.lock",
                "--db",
                ".cache/advisory-db",
                "--json",
            ],
        ),
        ("hex-audit", ["mix", "hex.audit"]),
    ]:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "MIX_ENV": "test"},
                capture_output=True,
                text=True,
                timeout=300,
            )
            status, output = result.returncode, result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            status, output = 1, str(error)
        (destination / f"{name}.txt").write_text(output)
        sections.append(
            f"## {name}: {'failed' if status else 'passed'}\n\n```text\n{output.rstrip()}\n```\n"
        )
        failed |= status != 0
    report = "# Dependency audit\n\n" + "\n".join(sections)
    (destination / "audit.md").write_text(report)
    print(report)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
