# Changelog

## 0.2.0 (unreleased)

- Upgrade ort to pinned rc.15 and Rustler to 0.38; Elixir 1.16 is required.
- Add explicit environment ownership, mutex-protected sessions, and safe binary decoding.
- Expose validated session options and shared environment thread pools.
- Load ONNX bytes directly with the copying constructor and normal session options.
- Inspect structured I/O types, symbolic dimensions, model fields, and custom properties.
- Bind tensor inputs by binary name and return named tensor outputs with map-based runs.
- Select and order runtime outputs and assign a per-run log tag with `run/3`.
- Decode ZipMap classifier probabilities to maps and tensor sequences to lists.
- Round-trip UTF-8 tensor and scalar strings with `Ortex.StringTensor`.
- Borrow aligned native tensor storage during synchronous inference, avoiding an extra input clone.
- Emit public load/run telemetry spans with durations, model references, tags, and value summaries.
- Configure provider option maps and query compiled/runtime provider availability.
- Specialize symbolic model dimensions with validated session overrides.
- Support checksum-verified precompiled CPU NIFs and source-build overrides.
- Default to dynamic runtime loading and API 17, without runtime build headers.

- Build and validate all seven artifacts, checksum maps, and a package in CI.
- Stage attested draft releases only in RiverFinancial/ortex; no Hex publisher.

Upgrade in a fresh VM; native resource layouts changed.
