"""Write a source dependency inventory in CycloneDX format; ORT is external."""

import json
import subprocess
from pathlib import Path

from package import version

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run(["mix", "run", "--no-start", "scripts/inventory.exs"], cwd=ROOT, check=True)
    result = subprocess.check_output(
        [
            "cargo",
            "metadata",
            "--locked",
            "--format-version",
            "1",
            "--features",
            "ort/load-dynamic",
        ],
        cwd=ROOT / "native/ortex",
        text=True,
    )
    metadata = json.loads(result)
    components = json.loads((ROOT / "dist/hex-components.json").read_text())
    for package in metadata["packages"]:
        if package["name"] == "ortex":
            continue
        component = {
            "type": "library",
            "name": package["name"],
            "version": package["version"],
            "purl": f"pkg:cargo/{package['name']}@{package['version']}",
        }
        if package.get("license"):
            component["licenses"] = [{"expression": package["license"].replace("/", " OR ")}]
        components.append(component)
    for component in components:
        component["bom-ref"] = component["purl"]
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {"type": "library", "name": "ortex", "version": version()},
            "properties": [
                {
                    "name": "ortex:scope",
                    "value": "Source dependency inventory, including build-time Rust crates. Not a per-binary linkage inventory.",
                },
                {
                    "name": "ortex:external-runtime",
                    "value": "ONNX Runtime is supplied by the consumer and is not bundled.",
                },
            ],
        },
        "components": sorted(components, key=lambda item: item["purl"]),
    }
    (ROOT / "dist/source.cdx.json").write_text(json.dumps(bom, indent=2) + "\n")


if __name__ == "__main__":
    main()
