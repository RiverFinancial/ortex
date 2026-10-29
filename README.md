# Ortex

`Ortex` is a wrapper around [ONNX Runtime](https://onnxruntime.ai/) (implemented as
bindings to [`ort`](https://github.com/pykeio/ort)). Ortex leverages
[`Nx.Serving`](https://hexdocs.pm/nx/Nx.Serving.html) to easily deploy ONNX models
that run concurrently and distributed in a cluster. Ortex also provides a storage-only
tensor implementation for ease of use.

ONNX models are a standard machine learning model format that can be exported from most ML
libraries like PyTorch and TensorFlow. Ortex allows for easy loading and fast inference of
ONNX models using different backends available to ONNX Runtime such as CUDA, TensorRT, Core
ML, and ARM Compute Library.

## Examples

TL;DR:

```elixir
iex> model = Ortex.load("./models/resnet50.onnx")
#Ortex.Model<
  inputs: [{"input", "Float32", [nil, 3, 224, 224]}]
  outputs: [{"output", "Float32", [nil, 1000]}]>
iex> {output} = Ortex.run(model, Nx.broadcast(0.0, {1, 3, 224, 224}))
iex> output |> Nx.backend_transfer() |> Nx.argmax
#Nx.Tensor<
  s64
  499
>
```

Inspecting a model shows the expected inputs, outputs, data types, and shapes. Axes with
`nil` represent a dynamic size.

To see more real world examples see the `examples` folder.

Public loading and inference calls emit `:telemetry` spans. See `Ortex.Telemetry`
for event names, measurements, and shape/type metadata.

`Ortex.Graph.from_binary(bytes)` and `Ortex.Graph.from_file(path)` summarize ONNX
versions, operators, initializer sizes, and ZipMap presence without creating a
runtime session or opening external weight files.

### Serving

`Ortex` also implements `Nx.Serving` behaviour. To use it in your application's
supervision tree consult the `Nx.Serving` docs.

```elixir
iex> serving = Nx.Serving.new(Ortex.Serving, model)
iex> batch = Nx.Batch.stack([{Nx.broadcast(0.0, {3, 224, 224})}])
iex> {result} = Nx.Serving.run(serving, batch)
iex> result |> Nx.backend_transfer() |> Nx.argmax(axis: 1)
#Nx.Tensor<
  s64[1]
  [499]
>
```

## Installation in River's fork

Pin a reviewed commit from `RiverFinancial/ortex` in your dependencies:

```elixir
{:ortex, github: "RiverFinancial/ortex", ref: "<reviewed-commit-sha>"}
```

Until the matching fork release assets exist, set `ORTEX_BUILD=true` and add
`{:rustler, "~> 0.38.0"}` to your application dependencies to build from source.
Source builds require Rust >= 1.92 and Elixir >= 1.16.

The precompiled CPU NIF targets Linux x86_64/aarch64 with glibc or musl,
both macOS architectures, and Windows x86_64 MSVC. It contains no ONNX Runtime
library: set `ORT_DYLIB_PATH` before startup to the library installed on your
machine. For Alpine runtime 1.24.4, use `/usr/lib/libonnxruntime.so.1.24.4`.

API 17 is the default floor; it is tested against runtime 1.17.3 and 1.24.4.
Custom Cargo features automatically select a source build. Include both a
linking strategy and an API floor in the replacement feature list:

```elixir
config :ortex, Ortex.Native, features: ["load-dynamic", "api-24", "coreml"]
```

See [the precompiled contract](docs/precompiled.md) for NIF ABI, checksum,
runtime compatibility, and local verification details.

## Session options and concurrency

```elixir
model = Ortex.load("model.onnx",
  providers: [:cpu], optimization_level: 3,
  intra_threads: 1, inter_threads: 1,
  intra_op_spinning: false, inter_op_spinning: false
)
```

Old positional load arities and tuple-based inference outputs remain available.
New keyword options reject duplicates, unknown keys, and invalid values. Unknown
or uncompiled providers raise instead of becoming CPU. Keyword calls fail on
provider registration errors unless `error_on_failure: false` is specified;
legacy calls retain runtime registration fallback behavior.

Calls sharing a model are serialized through a native mutex; separate model
sessions may run concurrently. This follows ort's mutable-session requirement.
See [the concurrency ADR](docs/adr-001-session-concurrency.md) and
[the baseline measurements](docs/baseline.md).

For a shared pool across models, configure the environment before starting Ortex:

```elixir
config :ortex, :environment,
  global_thread_pool: [intra_threads: 1, inter_threads: 1, spinning: false]
```

Global-pool sessions reject local thread-count/spinning options unless
`independent_thread_pool: true` is also supplied. Environment configuration is
fixed for the VM lifetime. See `Ortex.Environment` and `Ortex.load/2` for options.

## Loading model bytes

```elixir
model = Ortex.load_from_memory(onnx_bytes, intra_threads: 1)
```

Memory loading uses the same validated keyword options and requires no temporary
file. The runtime owns the graph after loading, so the source binary may be
released. Models with external tensor-data files still need file loading with
those files available. `Ortex.load/1` continues to interpret binaries as paths.

## Fork release automation

The release matrix builds and tests all seven targets without installing ONNX
Runtime at native compilation time. Tagged builds generate checksums, assemble
a Hex package as a validation artifact, attest the files, and stage a draft
GitHub release in RiverFinancial/ortex. No workflow publishes to Hex or writes
to another repository. See [the release instructions](docs/releases.md).
