"""Run file-selected checks; staged checks use an isolated Git index snapshot."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def run_check(item, directory):
    name, command = item
    try:
        result = subprocess.run(
            command,
            cwd=directory,
            env={
                **os.environ,
                "MIX_ENV": "test",
                "PATH": os.pathsep.join(
                    [
                        str(ROOT / ".venv/bin"),
                        str(ROOT / ".venv/Scripts"),
                        os.environ.get("PATH", ""),
                    ]
                ),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
        return name, result.returncode, result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return name, 1, str(error)


def checks(files, directory, full):
    groups = {
        extension: [p for p in files if p.endswith(extension)]
        for extension in (".py", ".ex", ".exs", ".rs")
    }
    jobs = []
    if groups[".py"]:
        jobs += [
            ("python lint", ["ruff", "check", "--", *groups[".py"]]),
            ("python format", ["ruff", "format", "--check", "--", *groups[".py"]]),
        ]
    elixir = groups[".ex"] + groups[".exs"]
    if elixir:
        jobs.append(("elixir format", ["mix", "format", "--check-formatted", "--", *elixir]))
    if groups[".rs"] or "native/ortex/Cargo.toml" in files or "rust-toolchain.toml" in files:
        jobs.append(
            (
                "rust format",
                ["cargo", "fmt", "--manifest-path", "native/ortex/Cargo.toml", "--check"],
            )
        )
    shell = [p for p in files if p.endswith(".sh") or p == "scripts/precommit"]
    if shell:
        jobs += [
            ("shell lint", ["shellcheck", "--", *shell]),
            ("shell format", ["shfmt", "-d", "--", *shell]),
        ]
    if any(p.startswith(".github/workflows/") for p in files):
        workflows = sorted(
            str(p.relative_to(directory)) for p in (directory / ".github/workflows").glob("*.yml")
        )
        jobs += [
            ("workflow syntax", ["actionlint", *workflows]),
            ("workflow security", ["zizmor", "--offline", "--no-progress", ".github/workflows"]),
        ]
    if full:
        jobs += [
            (
                "elixir compile and credo",
                ["mix", "do", "compile", "--warnings-as-errors", "+", "credo", "--strict"],
            ),
            (
                "rust clippy",
                [
                    "cargo",
                    "clippy",
                    "--manifest-path",
                    "native/ortex/Cargo.toml",
                    "--locked",
                    "--all-targets",
                    "--features",
                    "ort/load-dynamic",
                    "--",
                    "-D",
                    "warnings",
                ],
            ),
        ]
    return jobs


def check_files(files, directory, full):
    files = [p for p in files if (directory / p).is_file()]
    failed = False
    for name in files:
        path = directory / name
        if (
            path.suffix not in {".py", ".ex", ".exs", ".rs", ".toml", ".json", ".md", ".yml", ".sh"}
            and name != "Makefile"
        ):
            continue
        data = path.read_bytes()
        if data and (
            not data.endswith(b"\n")
            or any(line.rstrip(b" \t") != line for line in data.splitlines())
        ):
            print(f"FAIL whitespace: {name}")
            failed = True
    jobs = checks(files, directory, full)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_check, item, directory) for item in jobs]
        for future in futures:
            name, status, output = future.result()
            print(f"{'FAIL' if status else 'PASS'} {name}")
            if output.strip():
                print(output.rstrip())
            failed |= status != 0
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true")
    group.add_argument("--files", nargs="+")
    parser.add_argument("--full", action="store_true", help="also compile and run Credo/Clippy")
    args = parser.parse_args()
    if args.staged and args.full:
        parser.error("staged checks are fast static checks; use make lint for full checks")
    if args.staged:
        names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    else:
        names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    files = args.files or list(dict.fromkeys(os.fsdecode(p) for p in names.split(b"\0") if p))
    for name in files:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            parser.error(f"expected a repository-relative path: {name}")
    if args.staged:
        with tempfile.TemporaryDirectory(prefix="ortex-index-") as snapshot:
            git("checkout-index", "--all", f"--prefix={snapshot}{os.sep}")
            return check_files(files, Path(snapshot), False)
    return check_files(files, ROOT, args.full)


if __name__ == "__main__":
    sys.exit(main())
