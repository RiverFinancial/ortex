"""Emit explicit CI combinations and inspect successful full-validation markers."""

import argparse
import datetime
import json
import os
import re
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "ci/compatibility.json"


def api(path):
    repository = os.environ.get("GITHUB_REPOSITORY", "RiverFinancial/ortex")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("invalid repository")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/{path}", headers=headers
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def should_run(sha):
    now = datetime.datetime.now(datetime.timezone.utc)
    artifacts = api("actions/artifacts?name=compatibility-validated&per_page=100")["artifacts"]
    for artifact in artifacts:
        run = artifact.get("workflow_run", {})
        if artifact["expired"] or run.get("head_branch") != "main" or run.get("head_sha") != sha:
            continue
        created = datetime.datetime.fromisoformat(artifact["created_at"].replace("Z", "+00:00"))
        if now - created > datetime.timedelta(days=7):
            continue
        details = api(f"actions/runs/{run['id']}")
        if details["conclusion"] == "success" and details["event"] in {
            "push",
            "schedule",
            "workflow_dispatch",
        }:
            return False
    return True


def matrix(scope):
    policy = json.loads(POLICY.read_text())
    if scope == "alpine":
        return {"include": policy["alpine"]}
    rows = list(policy["pr"])
    if scope == "full":
        defaults = {k: policy["tools"][k] for k in ["elixir", "otp"]}
        rows += [{**defaults, "nx": "locked", **row} for row in policy["platforms"]]
        for runtime, nx in [("1.24.4", "minimum"), ("1.19.2", "latest")]:
            rows.append({**rows[0], "id": f"linux-{runtime}-{nx}", "runtime": runtime, "nx": nx})
    return {"include": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    matrices = commands.add_parser("matrix")
    matrices.add_argument("--scope", choices=["pr", "full", "alpine"], default="pr")
    nightly = commands.add_parser("changed")
    nightly.add_argument("sha")
    commands.add_parser("policy")
    nx = commands.add_parser("nx")
    nx.add_argument("mode", choices=["locked", "minimum", "latest"])
    args = parser.parse_args()
    if args.command == "matrix":
        print(json.dumps(matrix(args.scope), separators=(",", ":")))
    elif args.command == "changed":
        if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
            parser.error("expected a full commit SHA")
        print(str(should_run(args.sha)).lower())
    elif args.command == "nx":
        if args.mode == "minimum":
            project = ROOT / "mix.exs"
            text, count = re.subn(r'\{:nx, "[^"]+"\}', '{:nx, "== 0.6.2"}', project.read_text())
            if count != 1:
                raise ValueError("Nx dependency declaration changed")
            project.write_text(text)
    else:
        tools = json.loads(POLICY.read_text())["tools"]
        mise = tomllib.loads((ROOT / "mise.toml").read_text())["tools"]
        rust = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]
        assert mise["elixir"] == tools["elixir"] + "-otp-" + tools["otp"].split(".")[0]
        assert mise["erlang"] == tools["otp"]
        assert mise["python"] == tools["python"]
        assert rust["channel"] == tools["rust"]
        print("Toolchain declarations agree with CI policy")


if __name__ == "__main__":
    main()
