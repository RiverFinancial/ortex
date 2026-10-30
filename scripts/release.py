"""Validate release versions and render conventional-commit release notes."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

from package import version

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"feat": "Features", "fix": "Fixes", "perf": "Performance", "docs": "Documentation"}


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def resolve(reference):
    return git("rev-parse", "--verify", "--end-of-options", reference + "^{commit}")


def validate(tag=None):
    expected = version()
    manifest = tomllib.loads((ROOT / "native/ortex/Cargo.toml").read_text())
    lock = tomllib.loads((ROOT / "native/ortex/Cargo.lock").read_text())
    native = next(package["version"] for package in lock["package"] if package["name"] == "ortex")
    if manifest["package"]["version"] != expected or native != expected:
        raise ValueError("Mix, Cargo manifest, and Cargo lock versions must agree")
    if tag:
        if tag != "v" + expected or not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            raise ValueError("release tag must match the stable package version")
        if resolve(tag) != resolve("HEAD"):
            raise ValueError("release checkout is not the tagged commit")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", resolve(tag), "origin/main"],
            cwd=ROOT,
            check=True,
        )
    return expected


def group_commit(subject, body):
    match = re.fullmatch(r"([a-z]+)(?:\(([^)]+)\))?(!)?: (.+)", subject)
    if not match:
        return "Other", subject
    kind, scope, breaking, description = match.groups()
    bucket = (
        "Breaking changes"
        if breaking or "BREAKING CHANGE:" in body or "BREAKING-CHANGE:" in body
        else GROUPS.get(kind, "Maintenance")
    )
    return bucket, f"{scope}: {description}" if scope else description


def verify_artifacts(directory, require_complete=True):
    manifest = json.loads((directory / "artifacts.json").read_text())
    if manifest["version"] != version() or (require_complete and not manifest["complete"]):
        raise ValueError("release artifacts must match the version and contain every target")
    for name, digest in manifest["sha256"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"artifact changed after assembly: {name}")
    if require_complete:
        policy = json.loads((ROOT / "ci/compatibility.json").read_text())
        targets = {row["target"] for key in ["pr", "platforms", "alpine"] for row in policy[key]}
        required = {f"ortex-{version()}-{target}.tar.gz" for target in targets}
        required |= {f"ortex-{version()}.tar", f"docs-{version()}.tar.gz", "source.cdx.json"}
        if not required <= manifest["sha256"].keys():
            raise ValueError("release manifest is missing required artifacts")
    return manifest


def notes(start, end, directory):
    target = resolve(end)
    revision = f"{resolve(start)}..{target}" if start else target
    records = (
        subprocess.check_output(
            ["git", "log", "--reverse", "--format=%H%x00%s%x00%b%x00", revision], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    buckets = {}
    for index in range(0, len(records) - 2, 3):
        sha, subject, body = records[index : index + 3]
        bucket, description = group_commit(subject, body)
        buckets.setdefault(bucket, []).append(
            f"- **{description}** ([{sha.strip()[:7]}](https://github.com/RiverFinancial/ortex/commit/{sha.strip()}))"
        )
    output = [f"# Ortex {version()}\n"]
    for bucket in [
        "Breaking changes",
        "Features",
        "Fixes",
        "Performance",
        "Documentation",
        "Maintenance",
        "Other",
    ]:
        if entries := buckets.get(bucket):
            output += [f"## {bucket}\n", "\n".join(entries) + "\n"]
    manifest = verify_artifacts(directory, require_complete=False)
    if not manifest["complete"]:
        output.append("Artifact coverage: partial local assembly; not a release bundle.\n")
    output += ["## SHA-256\n", "| Artifact | SHA-256 |\n|---|---|"]
    for name, digest in manifest["sha256"].items():
        output.append(f"| `{name}` | `{digest}` |")
    return "\n".join(output) + "\n"


def request_json(url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def publish(tag):
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != "RiverFinancial/ortex"
        or os.environ.get("ORTEX_HEX_PUBLISH") != "true"
    ):
        raise ValueError("Hex publishing requires the explicitly enabled protected fork workflow")
    validate(tag)
    manifest = verify_artifacts(ROOT / "dist")
    with tempfile.TemporaryDirectory(prefix="ortex-publish-") as temporary:
        rebuilt = Path(temporary) / "package.tar"
        subprocess.run(["mix", "hex.build", "--output", str(rebuilt)], cwd=ROOT, check=True)
        if (
            hashlib.sha256(rebuilt.read_bytes()).hexdigest()
            != manifest["sha256"][f"ortex-{version()}.tar"]
        ):
            raise ValueError("rebuilt package differs from the approved artifact")
    audience = request_json("https://hex.pm/api/oidc/audience")["audience"]
    separator = "&" if "?" in os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] else "?"
    oidc = request_json(
        os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
        + separator
        + urllib.parse.urlencode({"audience": audience}),
        headers={"Authorization": "Bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]},
    )["value"]
    print(f"::add-mask::{oidc}", flush=True)
    token = request_json(
        "https://hex.pm/api/oauth/token",
        data=urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": oidc,
                "scope": "package:hexpm/ortex",
            }
        ).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )["access_token"]
    print(f"::add-mask::{token}", flush=True)
    subprocess.run(
        ["mix", "hex.publish", "--yes"],
        cwd=ROOT,
        env={**os.environ, "HEX_API_KEY": "Bearer " + token},
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate")
    validation.add_argument("--tag")
    commands.add_parser("verify")
    description = commands.add_parser("notes")
    description.add_argument("--from", dest="start")
    description.add_argument("--to", default="HEAD")
    description.add_argument("--directory", type=Path, default=ROOT / "dist")
    description.add_argument("--output", type=Path)
    publication = commands.add_parser("publish")
    publication.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        print(validate(args.tag))
    elif args.command == "publish":
        publish(args.tag)
    elif args.command == "verify":
        verify_artifacts(ROOT / "dist")
        print("Release artifact checksums verified")
    else:
        output = notes(args.start, args.to, args.directory)
        if args.output:
            args.output.write_text(output)
        else:
            print(output)


if __name__ == "__main__":
    sys.exit(main() or 0)
