"""Verify every prepared model and case file against its committed manifest."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "test/fixtures"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, metadata in manifest.items():
        for suffix, key in [("onnx", "sha256"), ("json", "cases_sha256")]:
            path = directory / f"{name}.{suffix}"
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata[key]:
                raise SystemExit(f"Fixture digest mismatch: {path}")
    print(f"Verified {len(manifest)} prepared model fixtures")


if __name__ == "__main__":
    main()
