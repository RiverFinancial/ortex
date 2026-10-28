# Inference baseline

The checked-in ONNX model is eight depth-three XGBoost regression trees trained on
deterministic synthetic data. It contains no River data or production model weights.
`python/generate_xgboost_fixture.py` regenerates it and its four reference scores,
checking XGBoost predictions independently against ONNX Runtime 1.24.4. Python is
not required to run the Elixir tests.

Run `MIX_ENV=test mix run bench/inference.exs 2000 8` for end-to-end public-API
latency, throughput with shared-model callers, and five repeated-inference RSS
samples. Results include tensor conversions and task scheduling. RSS includes the
whole VM and allocator retention; a short flat run cannot disprove a leak.
`ORTEX_BENCH_OPTIONS='[[intra_threads: 1, intra_op_spinning: false]]'` selects
keyword options after the upgrade. Use separate VMs to compare configurations.

## Before upgrade, 2026-10-28

- Source: upstream `main` at the fork point, ort rc.8, Nx 0.6.2, Rustler 0.29.1.
- Machine: arm64 macOS, Elixir 1.20.4 / OTP 29, Rust 1.93.1.
- The inherited static build produced a NIF that dyld rejected with a misaligned
  LINKEDIT string-pool error. For this comparison only, the existing source was
  rebuilt with `cargo build --release --locked --features ort/load-dynamic` and
  `ORT_DYLIB_PATH` set to the Python wheel's ONNX Runtime 1.24.4 library.
- Tests: 119 passed (including two doctests), two optional ResNet checks excluded.
- CPU provider, graph optimization disabled, 2,000 timed calls, concurrency eight.
- Serial throughput: 56,632 calls/s; p50: 9 us; p99: 138 us.
- Shared-model concurrent throughput: 33,145 calls/s.
- RSS before: 109,264,896 bytes. Five subsequent 2,000-call samples:
  110,624,768; 110,706,688; 110,706,688; 110,755,840; 110,755,840 bytes.

This characterizes repeated inference on our synthetic workload; upstream issue
44's specific reproducer and a long-duration production-like soak remain separate
investigations. Report measurements as observations, not performance guarantees.
