"""Install an official ONNX Runtime release for CI tests (not distributed in NIFs)."""
import os
import platform
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

version = sys.argv[1]
destination = Path(sys.argv[2] if len(sys.argv) > 2 else "runtime").resolve()
system = platform.system()
arm = platform.machine().lower() in ("arm64", "aarch64")
target = {
    "Darwin": "osx-arm64" if arm else "osx-x86_64",
    "Linux": "linux-aarch64" if arm else "linux-x64",
    "Windows": "win-arm64" if arm else "win-x64",
}[system]
extension = "zip" if system == "Windows" else "tgz"
filename = f"onnxruntime-{target}-{version}.{extension}"
destination.mkdir(parents=True, exist_ok=True)
archive = destination / filename
urllib.request.urlretrieve(
    f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{filename}", archive
)
if extension == "zip":
    with zipfile.ZipFile(archive) as source:
        source.extractall(destination)
else:
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
pattern = {"Darwin": "libonnxruntime.*.dylib", "Linux": "libonnxruntime.so.*", "Windows": "onnxruntime.dll"}[system]
library = next(destination.rglob(pattern))
print(library)
if "GITHUB_ENV" in os.environ:
    with open(os.environ["GITHUB_ENV"], "a") as output:
        output.write(f"ORT_DYLIB_PATH={library}\n")
