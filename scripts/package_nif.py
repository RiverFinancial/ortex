"""Package an already-built NIF using rustler_precompiled's exact filename format."""
import sys
import tarfile
from pathlib import Path

target, version = sys.argv[1:3]
windows = "windows" in target
mac = "apple" in target
prefix = "" if windows else "lib"
source_extension = ".dll" if windows else ".dylib" if mac else ".so"
extension = ".dll" if windows else ".so"
name = f"{prefix}ortex-v{version}-nif-2.16-{target}{extension}"
source = Path(f"native/ortex/target/{target}/release/{prefix}ortex{source_extension}")
destination = Path("dist")
destination.mkdir(exist_ok=True)
with tarfile.open(destination / (name + ".tar.gz"), "w:gz") as archive:
    archive.add(source, arcname=name)
print(destination / (name + ".tar.gz"))
