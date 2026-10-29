# Changelog

## 0.2.0 (unreleased)

- Upgrade ort to pinned rc.15 and Rustler to 0.38; Elixir 1.16 is required.
- Add explicit environment ownership, mutex-protected sessions, and safe binary decoding.
- Expose validated session options and shared environment thread pools.
- Support checksum-verified precompiled CPU NIFs and source-build overrides.
- Default to dynamic runtime loading and API 17, without runtime build headers.

Upgrade in a fresh VM; native resource layouts changed.
