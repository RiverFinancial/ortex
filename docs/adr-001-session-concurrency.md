# ADR 001: serialize inference within a session

Status: implemented, subject to workload measurements.

ort rc.15 requires mutable access to a session during inference. Its source
explains that some ONNX Runtime execution-provider allocators and statistics
trackers are not thread safe; bypassing that requirement with `unsafe impl Sync`
is not a supported concurrency strategy. This corrects the earlier research's
claim that exclusivity was merely an API choice.

Each NIF session resource holds `Mutex<Session>`. The lock covers inference and
output extraction, because outputs may borrow session-owned memory. Poisoned
locks return an error requiring model reload; we do not silently recover a
potentially inconsistent runtime session. Immutable input/output descriptors are
copied when loading and can be inspected without the inference lock.

Multiple callers can share a model safely, with runs serialized per session.
Different session resources may execute concurrently. Keep the existing `DirtyIo`
scheduling classification during this migration: changing the dirty scheduler
pool at the same time would confound the baseline. Choosing `DirtyCpu` remains a
separate measurement, especially because mutex waiters occupy dirty threads.

If representative load shows unacceptable queuing, introduce an explicit,
bounded session pool per model. Measure memory before pulling `PrepackedWeights`
forward. Do not assume the synthetic fixture proves production throughput.

ort rc.14 also requires an explicit environment. Sessions retain ort's environment
through their internal session ownership. An environment resource is additionally
owned by the BEAM application and retained for VM lifetime: ort permits only one
creation, even after its last reference is dropped. This supports application
restarts without Rust static environment/session resources. Global pool settings
are selected before startup; session-local thread tuning requires an explicit
independent pool when the global pool is enabled.

Sources: `ort` v2.0.0-rc.15 `src/session/mod.rs` and `src/environment.rs`; see
`docs/baseline.md` for the reproducible before/after measurements.
