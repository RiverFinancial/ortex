"""Report Rust advisories and retired Hex dependencies without hiding failures."""

import json
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
        if name == "cargo-audit":
            try:
                data = json.JSONDecoder().raw_decode(output)[0]
                (destination / "cargo-audit.json").write_text(json.dumps(data, indent=2) + "\n")
                issues = data.get("vulnerabilities", {}).get("list", [])
                warnings = sum(len(items) for items in data.get("warnings", {}).values())
                output = f"{len(issues)} vulnerabilities; {warnings} informational warnings.\n"
                if issues:
                    output += "\n".join(
                        f"{item['advisory']['id']}: {item['package']['name']} {item['package']['version']} — {item['advisory']['title']}"
                        for item in issues
                    )
            except (ValueError, KeyError):
                pass
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
