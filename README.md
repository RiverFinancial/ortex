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

## Installation

`Ortex` can be installed by adding `ortex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ortex, "~> 0.2.0"}
  ]
end
```

The 0.2.0 release is being prepared in this fork; until it is published, use a
pinned Git commit and set `ORTEX_BUILD=true`. Existing Hex 0.1.10 is unchanged.

Released packages use checksum-verified precompiled CPU NIFs for Linux x86_64 and
aarch64 (glibc and musl), both macOS architectures, and Windows x86_64 MSVC. These
NIFs **do not bundle ONNX Runtime**. Install a matching runtime shared library and
set `ORT_DYLIB_PATH` to its full path before starting the VM. Alpine's runtime-only
package uses a versioned path, e.g. `/usr/lib/libonnxruntime.so.1.24.4`, rather than
the unversioned symlink supplied by development headers. Runtime accelerator
libraries are also the application's responsibility.

The default API floor is 17 (ONNX Runtime 1.17+). We validate against official
runtime 1.17.3 and 1.24.4 releases and Alpine's 1.24.4 package. A newer runtime can
serve an older API; selecting `api-24` instead requires runtime 1.24 or newer.
The runtime version and the ort Rust crate version are distinct.

Source builds need Rust >= 1.92, Elixir >= 1.16, and OTP >= 24. The release NIF ABI
is 2.16, also usable by newer OTP versions. Cargo's default build uses dynamic
loading and needs no ONNX Runtime headers, library, or OpenSSL at compile time.

```elixir
# For source builds, also add {:rustler, "~> 0.38.0"} to your application deps.
config :ortex, Ortex.Native,
  features: ["load-dynamic", "api-24", "coreml"]
```

Custom features automatically select a source build. Feature lists replace the
defaults, so include a linking strategy and an explicit API floor. Cargo API
features are additive: enabling several selects the highest floor. CPU is the
default provider; accelerators are opt-in. `download-binaries`, `tls-rustls`, and
`lax-feature-matching` are available for custom source builds; the distributed
artifacts always use `load-dynamic`.

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

## Development and release

```sh
ORTEX_BUILD=true MIX_ENV=test mix deps.get
ORTEX_BUILD=true MIX_ENV=test mix test
ORTEX_BUILD=true MIX_ENV=test mix run --no-start scripts/check_global_pool.exs
docker build -f Dockerfile.alpine -t ortex-alpine .
```

The optional ResNet tests require a separately downloaded model. The checked-in
synthetic XGBoost fixture and all core tests run without model downloads or Python.
[Release instructions](docs/releases.md) cover tagged builds, attestations,
checksums, and Hex publication. Maintainer access is not required to develop in
this fork; publishing the existing Hex package requires ownership.
