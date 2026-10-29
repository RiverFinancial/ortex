# Changelog

## 0.2.0 (unreleased)

- Pin ort to 2.0.0-rc.15 and upgrade Rustler to 0.38. Elixir 1.16 is now required.
  Upgrade in a fresh VM; native resources have new layouts and versioned names.
- Serialize inference per session without the previous unsafe Sync implementation;
  retain legacy public load/run arities and the native init compatibility wrapper.
- Add validated keyword session options, threading/spinning controls, an explicit
  environment handle, and optional shared global thread pools.
- Reject unknown/uncompiled providers. New keyword calls fail on EP registration
  errors by default; legacy positional calls preserve registration fallback.
- Default to runtime dynamic loading and API 17; expose API 17 through 31 and
  explicit source-build accelerator features. No ONNX Runtime headers at build time.
- Add seven precompiled targets, checksum validation, artifact provenance, and
  tagged release / Hex publishing workflows. ONNX Runtime is supplied separately.
- Decode BEAM binaries into aligned owned storage, reject malformed tensor sizes,
  and return recoverable errors for unsupported output types and invalid bools.
- Add independent XGBoost reference scores, concurrency/option tests, Alpine
  validation, and a repeatable end-to-end benchmark.

Upstream work consulted: serpent213's PR 47 (upgrade), Kip Cole's PR 49
(rustler_precompiled), and the session-option proposals in PRs 50, 51, and 52,
including omondiedward's threading validation approach. These were reviewed as
design inputs; changes were implemented against rc.15 rather than merged wholesale.
