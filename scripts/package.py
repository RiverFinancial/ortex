"""Assemble the Hex archive and test it in a clean consuming project."""

import argparse
import io
import os
import re
import shutil
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
    subprocess.run(["mix", "hex.build", "--output", str(destination)], cwd=ROOT, check=True)
    return destination


def unpack(archive, destination):
    with tarfile.open(archive) as outer:
        payload = outer.extractfile("contents.tar.gz").read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as contents:
        contents.extractall(destination, filter="data")


def test_package(archive, mode, nif, timeout):
    with tempfile.TemporaryDirectory(prefix="ortex-package-") as temporary:
        directory = Path(temporary)
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
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    test = commands.add_parser("test")
    test.add_argument("--archive", type=Path)
    test.add_argument("--mode", choices=["source", "prebuilt"], default="source")
    test.add_argument("--nif", help="prebuilt library; default: local priv/native library")
    test.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.command == "build":
        print(build())
        return 0
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    archive = args.archive or build()
    return test_package(archive, args.mode, args.nif, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
