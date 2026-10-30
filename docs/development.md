# Development

Use `mise install`, then `mise exec -- make setup` and `mise exec -- make doctor`.
Rustup reads the pinned rust-toolchain.toml. GNU Make 3.81 or newer is sufficient;
Windows developers can use Git Bash with Make available on PATH.

The supported BEAM pairs are Elixir 1.19 / OTP 28 and Elixir 1.20 / OTP 29.
Normal development uses the latter. Rust 1.92 is the minimum; the normal pinned
toolchain is 1.93.1. Python 3.12+ runs the standard-library tooling.

ONNX Runtime is a separate dependency. `make setup` installs the development
runtime under runtime/ and records its path. Make passes that path through
ORT_DYLIB_PATH; an explicitly set environment variable takes precedence. Direct
Mix commands need that variable too. ONNX Runtime 1.23.2 is used on Intel macOS
because 1.24.4 has no official Intel macOS archive. Successful process exit is
part of testing; runtime 1.22.0 has exhibited shutdown failures on macOS.

The foundation retains main's ort rc.8 and Rustler 0.29 native implementation.
Rustler migration belongs to the separate feature work. CPU dynamic loading is
the development default; additional Cargo features must be explicitly selected
in the consuming application's config. Library tests remain under test/ with
normal ExUnit discovery. Additional suites are introduced by later stack PRs.

Documentation has its own Mix environment so building docs does not install
EXLA, Torchx, or example tokenizers. Large examples own their extra dependencies.

## Checks and hooks

Activate `.venv/bin` alongside the mise tools after setup. `make lint` runs
format checks, Credo, Clippy, and workflow validation. `make fmt` applies
formatters. `scripts/precommit` runs fast checks against an isolated snapshot of
the Git index, so partially staged files are checked as committed. It never
changes the index or working tree. Filenames are passed as arguments, not shell
code. Full checks require prepared dependencies; hooks never install them.

`make hooks` installs a launcher through `git rev-parse --git-path`; it refuses
to replace an unrelated hook. Git's hooks directory is shared by linked
worktrees. Agent hooks are adapters to the same checker and require the agent's
normal hook review/trust. No global permissions or model settings are changed.
