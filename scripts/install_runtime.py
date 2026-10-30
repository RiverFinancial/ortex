"""Install the separately supplied ONNX Runtime; print its path for Make."""

import argparse
import json
import platform
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "runtime"
RECEIPT = DESTINATION / "installed.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "path"])
    parser.add_argument("--version")
    args = parser.parse_args()
    if args.command == "path":
        library = Path(json.loads(RECEIPT.read_text())["library"])
        if not library.is_file():
            raise SystemExit("runtime receipt is stale; run make setup")
        print(library)
        return
    system = platform.system()
    arm = platform.machine().lower() in {"arm64", "aarch64"}
    target = {
        "Darwin": "osx-arm64" if arm else "osx-x86_64",
        "Linux": "linux-aarch64" if arm else "linux-x64",
        "Windows": "win-arm64" if arm else "win-x64",
    }[system]
    version = args.version or ("1.23.2" if target == "osx-x86_64" else "1.24.4")
    extension = "zip" if system == "Windows" else "tgz"
    name = f"onnxruntime-{target}-{version}"
    DESTINATION.mkdir(exist_ok=True)
    archive = DESTINATION / f"{name}.{extension}"
    urllib.request.urlretrieve(
        f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{archive.name}",
        archive,
    )
    if extension == "zip":
        with zipfile.ZipFile(archive) as source:
            source.extractall(DESTINATION)
    else:
        with tarfile.open(archive) as source:
            source.extractall(DESTINATION, filter="data")
    pattern = {"Darwin": "libonnxruntime.*.dylib", "Linux": "libonnxruntime.so.*", "Windows": "onnxruntime.dll"}[system]
    library = next((DESTINATION / name).rglob(pattern)).resolve()
    RECEIPT.write_text(json.dumps({"version": version, "target": target, "library": str(library)}) + "\n")
    print(library)


if __name__ == "__main__":
    main()
