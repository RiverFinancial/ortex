# Fork integration

Development stays in `RiverFinancial/ortex`. Changes are reviewed as a stack of
small draft PRs; `codex/ortex-integration` combines the current stack for source,
precompiled-consumer, and package validation. No workflow publishes to Hex or
writes to another repository. The broad original PR is superseded by the stack.

## Review order

| PR | Change |
| --- | --- |
| 2 | Synthetic tree-model inference baseline |
| 3 | Binary size, overflow, and alignment safety |
| 4 | ort/Rustler migration, explicit environment, session mutex |
| 5 | Validated keyword session options |
| 6 | Shared environment thread pool |
| 7 | Precompiled NIF loading and source-build overrides |
| 8 | Fork-only artifact and draft-release automation |
| 9 | Copying model-byte loading |
| 10 | Tensor validation before runtime startup |
| 11 | Combined constructor/pool/restart validation |
| 12 | Structured types, dimensions, and model metadata |
| 13 | Named tensor inputs and outputs |
| 14 | Per-run output selection and tags |
| 15 | Map and Sequence output values |
| 16 | String tensors and scalar round-trips |
| 17 | Borrowed aligned native inputs |
| 18 | Longer inference and load/disposal diagnostic |
| 19 | Public load/run telemetry |
| 20 | Provider configuration and availability |
| 21 | Symbolic dimension overrides |
| 22 | Session profiling trace retrieval |
| 23 | Bounded runtime log bridge |
| 24 | Environment defaults and runtime selection |
| 25 | Numeric storage and Serving support contract |
| 26 | Runtime-independent ONNX graph summaries |
| 27 | One-attempt native cancellation controls |
| 28 | Managed deadlines and caller-exit cancellation |
| 29 | Dirty-scheduler measurement and policy |
| 30 | Final lifecycle/managed-run soak and review ledger |
| 31 | Atomic control ownership before worker/deadline creation |

Each PR targets the preceding branch. Merge in order with merge commits to
preserve ancestry. If using squash/rebase merges, rebase and retarget the remaining
stack and rerun its checks before proceeding. The integration branch combines
validation rather than replacing the small PRs. CI checks each intermediate state.
Task commits are unsigned because the local signing agent cancelled requests.

## Integration gates

Set `ORT_DYLIB_PATH` to a tested runtime library before inference. Force a native
rebuild when changing Rust sources or build modes; old Rustler build caches can
otherwise leave a stale NIF in use.

```sh
ORTEX_BUILD=true MIX_ENV=test mix compile --force --warnings-as-errors
ORTEX_BUILD=true MIX_ENV=test mix test
ORTEX_BUILD=true MIX_ENV=test mix run --no-start scripts/check_global_pool.exs
ORTEX_BUILD=true ORT_DYLIB_PATH=/does-not-exist/libonnxruntime.so MIX_ENV=test \
  mix run --no-start scripts/check_binary_without_runtime.exs
```

The assembled stack currently has 218 checks, with two optional ResNet checks
excluded when that external checkpoint is absent. Its fixture contains only
synthetic data. Memory loading checks file/bytes score and descriptor parity,
source-owner process exit, malformed-model recovery, and concurrent loading.
Metadata fixtures cover named/unnamed dimensions, scalar booleans, strings,
sequence/map descriptors, Unicode custom properties, and concurrent reads/runs.
Named-I/O tests check order-independent binding, Unicode output names, constructor
parity, invalid and duplicate names, and concurrent runs on a shared model.
Run-option checks cover native output selection, requested positional order,
invalid selections/options, and independent concurrent call configuration.
Value fixtures cover integer/string ZipMap classes, Unicode labels, tensor and
empty sequences, zero-input graphs, nested value ownership, and non-finite map
float error recovery. Unsupported map types remain recoverable errors.
String checks cover Unicode, empty strings/tensors, matrix and scalar shapes,
memory ownership, malformed inputs at both API layers, and numeric/bool scalars.
Telemetry checks assert start/stop/exception contracts, file/memory load metadata,
named selected outputs, model references, durations, tags, and shape summaries.
Provider checks cover CPU arena options through both constructors, malformed and
unknown options, uncompiled-provider errors, and build/runtime availability.
All accelerator feature mappings are compile-checked together; accelerator runtime
execution still needs its corresponding libraries and hardware.
Borrowed-input tests retain resources across owner exit, concurrent reuse, slicing,
and temporary boolean conversion. Raw BEAM binaries still take a validated aligned
copy; `TensorRef` borrows native storage only until synchronous inference returns.

`bench/input_copy.exs` separates transfer cost from inference with an existing
Ortex resource. On a local 3 MiB identity input (100 iterations), removing the
redundant native clone changed median latency from 121 to 85 microseconds for a
reused resource and 207 to 166 microseconds from BinaryBackend. These are local
diagnostics, not cross-platform latency guarantees or a raw-binary zero-copy claim.

`bench/lifecycle.exs` checks golden scores through ordinary runs, alternating
file/memory load-disposal owners, and optional bounded runs. The final Apple
silicon run used 100,000 ordinary runs, 1,000 load/disposal cycles, and 100,000
managed bounded calls on each runtime:

```sh
ORTEX_BUILD=true MIX_ENV=test mix run bench/lifecycle.exs 100000 1000 100000
```

| Runtime | Warmup RSS | Maximum ordinary-run RSS | Final disposal RSS | Bounded RSS min/max | Final bounded RSS |
| --- | --- | --- | --- | --- | --- |
| 1.17.3 | 105,185,280 | 105,742,336 | 104,677,376 | 102,825,984 / 104,153,088 | 103,038,976 |
| 1.24.4 | 107,577,344 | 108,003,328 | 107,003,904 | 105,299,968 / 106,496,000 | 105,299,968 |

Sampled BEAM process counts stayed at 124 throughout. The harness uses
`Enum.each`, discarding per-call results; collecting comprehensions in an earlier
diagnostic retained an accumulator and distorted heap/native-memory observations.
Temporary lifetime counters independently showed controls being reclaimed and
were removed before the final run. No production native code changed for this
harness correction.

Byte counts include BEAM and allocator retention. These samples show no sustained
growth for the exercised synthetic CPU workload, including managed worker/control
ownership; they do not prove leak freedom for other graphs, providers, or runtime
versions. Repeat with production-representative fixtures before consumer rollout.

The separate global-pool VM exercises **both** file and memory constructors:
inherited pools, rejected ignored settings, explicit independent pools, unchanged
scores, and resource survival through application restart. The precompiled check
runs these same gates with cargo/rustc/protoc replaced by failing stubs.

Separate runtime-log VMs configure info and warning levels before the one-time
environment initialization. They assert per-session log IDs from both constructors,
application restart delivery, and a 256-entry native queue with counted overflow.
Callbacks enqueue strings only; the BEAM environment process drains them into
Logger. Each string field is bounded at 16 KiB with a UTF-8-safe truncation marker.
The environment resource is V2; upgrades require a fresh VM.

The environment-default VM gives config runtime paths precedence over an invalid
`ORT_DYLIB_PATH`. It observes actual BFCArena creation to distinguish inherited
CPU arena settings from explicit session overrides through both constructors,
then verifies retained resource identity and rejected configuration changes across
application restarts. Legacy positional loading keeps its CPU defaults.

CI covers API 17/runtime 1.17.3, API 17/runtime 1.24.4, API 24/runtime 1.24.4,
both Alpine architectures, all seven NIF targets, and complete checksum/package
assembly. PR checks and workflow artifacts are the current evidence; a test count
alone does not prove those platform or constructor combinations passed. The final
validation PR records exact integration run links without requiring a release tag.
Runtime cancellation is cooperative and may finish after the caller deadline.
Accelerator runtime execution remains unvalidated without its libraries/hardware.

The final ownership fix reserves a control before creating its watcher or deadline.
Regression tests cover a zero-deadline duplicate before native entry, wrong-model
reuse, and 16 concurrent public reservations with exactly one successful owner.
After the fix, another 100,000 bounded calls per runtime retained checked scores,
bounded sampled memory, and flat process counts. Run-control resources are V2.

## Deferred scope

The stack covers the selected CPU/API 17–24 modernization, session/environment
controls, I/O values, metadata/graph inspection, observability, cancellation, and
Nx support decisions. Per-run profiling toggles require API 25 and remain outside
the tested runtime matrix; session profiling is available now.

GPU/NN primitives remain a later or measurement-dependent stage: shared weights
and pooling, AOT compilation, I/O binding, device allocation, and adapters. Keep
legacy positional APIs and `Native.show_session/1` available while extending the
public interface so existing fork consumers have a migration path.

Training, custom operators, Model Editor, alternative backends, the NPU tail,
upstream collaboration, and shared-package ownership are outside this fork work.
