# Precompiled NIF contract

The NIF targets ABI 2.16 (OTP 24+) and ONNX Runtime API 17 by default. Runtime
1.17.3 and 1.24.4 are tested. Artifacts contain only the CPU NIF, not the runtime.
The consumer supplies a library via `ORT_DYLIB_PATH`. Source builds need no
runtime development headers or OpenSSL when `load-dynamic` is selected.

Targets: `aarch64-apple-darwin`, `x86_64-apple-darwin`,
`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`,
`x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`, and
`x86_64-pc-windows-msvc`. Archives are fetched only from River's fork releases.

RustlerPrecompiled verifies SHA-256 using the checksum map inside the package.
Custom Cargo feature lists force a source build so a CPU artifact cannot silently
stand in for an accelerator-enabled build. `ORTEX_BUILD=true` also forces source
compilation. Include Rustler in the consuming application's dependencies for that
path. Feature lists replace defaults; include a linking mode and an API floor.

To verify locally, build the desired native target from `native/ortex`, run
`python3 scripts/package_nif.py TARGET VERSION` from the root, then run
`bash scripts/check_precompiled.sh dist` with `ORT_DYLIB_PATH` set. The check
uses failing cargo/rustc stubs to reject accidental fallback to compilation.
Generated archives and checksum files are ignored by Git.

macOS runtime 1.22.0 passed inference assertions but aborted on process exit in
validation, matching the symptom in ONNX Runtime issue 24579. Runtime 1.23.2
exits cleanly locally; the release matrix must check process exit as well as
inference results. Runtime 1.24.4 has no official Intel macOS artifact.
