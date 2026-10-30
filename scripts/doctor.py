"""Report the active environment without installing or changing anything."""

import ctypes
import os
import platform
import shutil
import subprocess
import sys


def main():
    failed = False
    print(f"Host: {platform.system()} {platform.machine()} ({platform.libc_ver()})")
    print(f"Python: {platform.python_version()}")
    for executable, arguments in [("elixir", ["--version"]), ("cargo", ["--version"])]:
        if not shutil.which(executable):
            print(f"MISSING: {executable}; run mise install / install rustup")
            failed = True
            continue
        result = subprocess.run(
            [shutil.which(executable), *arguments], capture_output=True, text=True
        )
        print(result.stdout.strip() or result.stderr.strip())
        failed |= result.returncode != 0
    runtime = os.environ.get("ORT_DYLIB_PATH")
    print(f"ORT_DYLIB_PATH: {runtime or '(unset; run make setup)'}")
    if runtime:
        try:
            ctypes.CDLL(runtime)
        except OSError as error:
            print(f"Runtime cannot be loaded: {error}")
            failed = True
    else:
        failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
