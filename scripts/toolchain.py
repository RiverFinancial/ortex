"""Install the declared Rust toolchain once, before checks run concurrently."""

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]
    command = ["rustup", "toolchain", "install", spec["channel"], "--profile", spec["profile"]]
    for component in spec["components"]:
        command += ["--component", component]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
