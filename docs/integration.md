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

Each PR targets the preceding branch. Merge or retarget in order; the integration
branch is for validation rather than a replacement broad PR. CI on each branch
checks that its intermediate state works independently.

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

The assembled stack currently has 158 checks, with two optional ResNet checks
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

The separate global-pool VM exercises **both** file and memory constructors:
inherited pools, rejected ignored settings, explicit independent pools, unchanged
scores, and resource survival through application restart. The precompiled check
runs these same gates with cargo/rustc replaced by failing stubs.

CI covers API 17/runtime 1.17.3, API 17/runtime 1.24.4, API 24/runtime 1.24.4,
both Alpine architectures, all seven NIF targets, and complete checksum/package
assembly. PR checks and workflow artifacts are the current evidence; a test count
alone does not prove those platform or constructor combinations passed.

## Remaining work

The initial runtime/build foundation is not the whole capability backlog. Continue
with separate focused PRs for:

- Run cancellation, timeout, and caller-exit handling.
- String tensors, scalar convenience APIs, and borrowed tensor inputs.
- EP-specific configuration and availability checks; further environment controls.
- Elixir telemetry, runtime logging bridge, and run-level profiling.
- ONNX graph summaries from protobuf: versions, operators, initializers, and ZipMap.
- An explicit decision for the partial Nx backend and Serving support.
- Longer memory/load/disposal measurements and dirty-scheduler classification.

GPU/NN primitives remain a later or measurement-dependent stage: shared weights
and pooling, AOT compilation, I/O binding, device allocation, and adapters. Keep
legacy positional APIs and `Native.show_session/1` available while extending the
public interface so existing fork consumers have a migration path.

Training, custom operators, Model Editor, alternative backends, the NPU tail,
upstream collaboration, and shared-package ownership are outside this fork work.
