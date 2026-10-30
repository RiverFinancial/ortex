"""Assemble the Hex archive and test it in a clean consuming project."""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from scenario import execute

ROOT = Path(__file__).resolve().parents[1]


def version():
    return re.search(r'version: "([^"]+)"', (ROOT / "mix.exs").read_text())[1]


def build():
    destination = ROOT / "dist" / f"ortex-{version()}.tar"
    destination.parent.mkdir(exist_ok=True)
    subprocess.run(
        [shutil.which("mix") or "mix", "hex.build", "--output", str(destination)],
        cwd=ROOT,
        check=True,
    )
    return destination


def unpack(archive, destination):
    with tarfile.open(archive) as outer:
        payload = outer.extractfile("contents.tar.gz").read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as contents:
        contents.extractall(destination, filter="data")


def nif_archive(target):
    policy = json.loads((ROOT / "ci/compatibility.json").read_text())
    targets = {row["target"] for key in ["pr", "platforms", "alpine"] for row in policy[key]}
    if target not in targets:
        raise ValueError("target is not in the artifact contract")
    source = ROOT / "priv/native" / ("libortex.dll" if "windows" in target else "libortex.so")
    destination = ROOT / "dist" / f"ortex-{version()}-{target}.tar.gz"
    destination.parent.mkdir(exist_ok=True)
    with (
        destination.open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        entry = tarfile.TarInfo(source.name)
        data = source.read_bytes()
        entry.size, entry.mode = len(data), 0o644
        archive.addfile(entry, io.BytesIO(data))
    return destination


def inspect_binary(data, target):
    if "linux" in target:
        if data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01":
            raise ValueError("expected a little-endian ELF64 library")
        machine = struct.unpack_from("<H", data, 18)[0]
        expected = 183 if target.startswith("aarch64") else 62
    elif "apple" in target:
        if data[:4] != b"\xcf\xfa\xed\xfe":
            raise ValueError("expected a Mach-O 64 library")
        machine = struct.unpack_from("<I", data, 4)[0]
        expected = 0x100000C if target.startswith("aarch64") else 0x1000007
    else:
        if data[:2] != b"MZ":
            raise ValueError("expected a PE library")
        offset = struct.unpack_from("<I", data, 0x3C)[0]
        if data[offset : offset + 4] != b"PE\0\0":
            raise ValueError("invalid PE signature")
        machine = struct.unpack_from("<H", data, offset + 4)[0]
        expected = 0x8664
    if machine != expected:
        raise ValueError(f"binary architecture does not match {target}")


def assemble(require_all):
    directory = ROOT / "dist"
    policy = json.loads((ROOT / "ci/compatibility.json").read_text())
    targets = {row["target"] for key in ["pr", "platforms", "alpine"] for row in policy[key]}
    expected = {f"ortex-{version()}-{target}.tar.gz": target for target in targets}
    archives = {path.name: path for path in directory.glob("ortex-*.tar.gz")}
    if archives.keys() - expected.keys():
        raise ValueError("unexpected native artifact version or target")
    if require_all and archives.keys() != expected.keys():
        raise ValueError(f"missing targets: {sorted(expected.keys() - archives.keys())}")
    if not archives:
        raise ValueError("no native archives")
    for name, path in archives.items():
        target = expected[name]
        member_name = "libortex.dll" if "windows" in target else "libortex.so"
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            if len(members) != 1 or not members[0].isfile() or members[0].name != member_name:
                raise ValueError(f"unexpected native archive contents: {name}")
            inspect_binary(archive.extractfile(members[0]).read(), target)
    build()
    files = sorted(
        [
            *archives.values(),
            directory / f"ortex-{version()}.tar",
            *directory.glob("*.cdx.json"),
            *directory.glob("docs-*.tar.gz"),
        ]
    )
    digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (directory / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in digests.items())
    )
    (directory / "artifacts.json").write_text(
        json.dumps(
            {"version": version(), "complete": len(archives) == len(expected), "sha256": digests},
            indent=2,
        )
        + "\n"
    )
    return digests


def test_package(archive, mode, nif, timeout):
    with tempfile.TemporaryDirectory(prefix="ortex-package-") as temporary:
        # Expand Windows 8.3 names before Mix compares priv-directory symlink targets.
        directory = Path(temporary).resolve()
        package = directory / "ortex"
        unpack(archive, package)
        project = directory / "consumer"
        shutil.copytree(ROOT / "test/package", project)
        env = {
            **os.environ,
            "MIX_ENV": "prod",
            "RUSTUP_TOOLCHAIN": "1.93.1",
            "ORTEX_PACKAGE_PATH": str(package),
            "ORTEX_PACKAGE_MODE": mode,
            "ORTEX_FIXTURE_PATH": str(ROOT / "test/fixtures/xgboost.onnx"),
            "ORTEX_CASES_PATH": str(ROOT / "test/fixtures/xgboost.json"),
        }
        if mode == "prebuilt":
            binary = (
                Path(nif).resolve()
                if nif
                else ROOT / "priv/native" / ("libortex.dll" if os.name == "nt" else "libortex.so")
            )
            native = package / "priv/native"
            native.mkdir(parents=True, exist_ok=True)
            shutil.copy2(binary, native / binary.name)
            stubs = directory / "forbidden-compilers"
            stubs.mkdir()
            for name in ["cargo", "rustc"]:
                stub = stubs / name
                stub.write_text(
                    "#!/bin/sh\necho 'Compiler forbidden in prebuilt package check' >&2\nexit 99\n"
                )
                stub.chmod(0o755)
                (stubs / f"{name}.cmd").write_text(
                    "@echo Compiler forbidden in prebuilt package check\r\n@exit /b 99\r\n"
                )
            env["PATH"] = str(stubs) + os.pathsep + env["PATH"]
        for command in [["mix", "deps.get"], ["mix", "run", "smoke.exs"]]:
            status, output = execute(command, timeout, cwd=project, env=env)
            print(output, flush=True)
            if status:
                return status
        invalid = {**env, "ORT_DYLIB_PATH": str(directory / "missing-runtime")}
        status, output = execute(
            ["mix", "run", "--no-compile", "missing_runtime.exs"], timeout, cwd=project, env=invalid
        )
        print(output, flush=True)
        if status:
            return status
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    nif = commands.add_parser("nif")
    nif.add_argument("--target", required=True)
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--all", action="store_true", help="require all seven native targets")
    test = commands.add_parser("test")
    test.add_argument("--archive", type=Path)
    test.add_argument("--mode", choices=["source", "prebuilt"], default="source")
    test.add_argument("--nif", help="prebuilt library; default: local priv/native library")
    test.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.command == "build":
        print(build())
        return 0
    if args.command == "nif":
        print(nif_archive(args.target))
        return 0
    if args.command == "assemble":
        print(json.dumps(assemble(args.all), indent=2))
        return 0
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    archive = args.archive or build()
    return test_package(archive, args.mode, args.nif, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
