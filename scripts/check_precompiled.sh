#!/usr/bin/env bash
# Run from a clean checkout with a matching archive in the supplied directory.
# Cargo/rustc stubs make accidental fallback to source compilation fail loudly.
set -euo pipefail
artifact_dir=$(cd "$1" && pwd)
export RUSTLER_PRECOMPILED_GLOBAL_CACHE_PATH="$artifact_dir"
unset ORTEX_BUILD RUSTLER_PRECOMPILED_FORCE_BUILD_ALL
elixir scripts/checksums.exs "$artifact_dir"
stub_dir=$(mktemp -d)
trap 'rm -rf "$stub_dir"' EXIT
for compiler in cargo rustc; do
  cat > "$stub_dir/$compiler" <<'STUB'
#!/bin/sh
echo 'ERROR: precompiled test attempted Rust compilation' >&2
exit 99
STUB
  chmod +x "$stub_dir/$compiler"
done
export PATH="$stub_dir:$PATH"
export MIX_ENV=test
mix clean
mix test
mix run --no-start scripts/check_global_pool.exs
mix run --no-start scripts/check_environment_defaults.exs
mix run --no-start scripts/check_runtime_logs.exs info
mix run --no-start scripts/check_runtime_logs.exs warning
ORT_DYLIB_PATH=/does-not-exist/libonnxruntime.so mix run --no-start scripts/check_binary_without_runtime.exs
