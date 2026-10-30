"""Run each system scenario in a fresh BEAM and require successful shutdown."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def execute(command, timeout, cwd=ROOT, env=None):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return 124, output + f"\nScenario exceeded {timeout}s\n"
    return process.returncode, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="scenario stems; default: all")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    scenarios = {p.stem: p for p in sorted((ROOT / "test/system").glob("*.exs"))}
    selected = args.names or list(scenarios)
    if set(selected) - scenarios.keys():
        parser.error(f"unknown scenarios: {set(selected) - scenarios.keys()}")
    env = {**os.environ, "MIX_ENV": "test"}
    subprocess.run(["mix", "compile", "--warnings-as-errors"], cwd=ROOT, env=env, check=True)
    results = []
    for name in selected:
        status, output = execute(
            ["mix", "run", "--no-compile", "--no-deps-check", "--no-start", str(scenarios[name])],
            args.timeout,
            env=env,
        )
        print(f"{'PASS' if status == 0 else 'FAIL'} {name}\n{output}", flush=True)
        results.append({"scenario": name, "exit_status": status, "output": output})
    directory = ROOT / "dist/reports"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "system.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["exit_status"] != 0 for result in results))


if __name__ == "__main__":
    sys.exit(main())
