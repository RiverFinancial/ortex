# Build and release

The `Source validation` workflow runs tests against API 17 / runtime 1.17.3,
API 17 / runtime 1.24.4, and API 24 / runtime 1.24.4. It checks formatting,
warnings, shared global pools, and source builds on both Alpine architectures.
The `Precompiled NIFs` workflow builds seven native targets, with no ONNX Runtime
installed at NIF compilation time. Each archive is then tested through
RustlerPrecompiled with `cargo` and `rustc` replaced by failing stubs. Musl
consumers run in a separate Alpine image that has no Rust toolchain installed.

## Artifact contract

- Cargo: ort exactly rc.15, default `load-dynamic` and `api-17`.
- NIF ABI: 2.16, supports OTP 24 and newer.
- x86_64/aarch64 Linux glibc, x86_64/aarch64 Linux musl, arm64/Intel macOS,
  and x86_64 Windows MSVC.
- Linux glibc artifacts are built on Ubuntu 22.04 (x86_64, glibc 2.35) and
  Ubuntu 24.04 (aarch64, glibc 2.39); older glibc distributions require source
  builds. Musl has separate artifacts and never consumes glibc binaries.
- CPU provider only; runtime library supplied by the consumer. Custom Cargo
  features select source builds, preventing accidental reuse of CPU-only assets.
- Musl artifacts are built inside pinned Alpine with its native Rust 1.96.1
  toolchain, avoiding accidental glibc libgcc dependencies from a host linker.
- The archive contains only the NIF. RustlerPrecompiled verifies SHA-256 against
  the checksum map shipped inside the Hex package.

The macOS 1.22.0 runtime aborted at process exit after passing all inference
assertions, on both Intel CI and local arm64. The symptom is reported in
[ONNX Runtime issue 24579](https://github.com/microsoft/onnxruntime/issues/24579).
Local arm64 runtime 1.23.2 exits cleanly; Intel CI uses 1.23.2 because official
runtime 1.24.4 binaries are arm64-only on macOS. Keep process-exit success in the
consumer checks: assertions alone must not conceal a shutdown abort.

## Publishing a version

1. Set matching stable versions in `mix.exs` and `native/ortex/Cargo.toml`, refresh
   the Cargo lock, update the changelog, and merge validated changes.
2. Push the matching `vVERSION` tag. This runs source validation and all seven
   build/consumer jobs. Only after they pass does the release job proceed.
3. The release job generates all seven checksums, builds the Hex archive with
   the checksum map included, attests the archives/package/checksum map, and
   creates a **draft** GitHub release. No Hex publication occurs on tag push.
4. Review the draft and its artifacts. Verify provenance, for example:
   `gh attestation verify FILE --repo RiverFinancial/ortex`.
5. Publishing the GitHub release triggers the Hex publication workflow. It
   verifies the released artifacts' attestations and packages the tagged source
   with the verified checksum file. This needs a package-owner `HEX_API_KEY`
   repository secret and ownership of `ortex` on Hex. Set those up before
   publishing a release. They are not needed for builds or draft PRs.

Do not use `--ignore-unavailable` when preparing a release. Missing targets,
missing checksums, mismatched versions, failed consumers, or absent provenance
must fail the pipeline. Checksums are generated into the package, not guessed or
committed before binaries exist. Do not alter a published release's NIF assets.
Upgrade from 0.1.x in a fresh VM; the native resource representations changed.
Versioned resource names prevent takeover of old-layout resources during reload.

## Local precompiled verification

Build the native target from `native/ortex` so its `.cargo/config.toml` applies,
then run `python3 scripts/package_nif.py TARGET VERSION` from the repository root.
With `ORT_DYLIB_PATH` set, run `bash scripts/check_precompiled.sh dist`. This
generates a local checksum map and tests installation without Rust compilation.
Generated checksums, caches, and archives are ignored by Git.

For Alpine, create `Dockerfile.precompiled-alpine.dockerignore` from `.dockerignore`
with its `dist` line removed, place a matching musl archive in `dist`, and build
`Dockerfile.precompiled-alpine`. Source verification uses `Dockerfile.alpine`.
