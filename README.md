# Ortex

Ortex wraps [ONNX Runtime](https://onnxruntime.ai/) for Elixir through Rust and
supports numeric tensor transfer and inference with Nx.Serving.

This is the RiverFinancial fork. Pin a reviewed commit from
https://github.com/RiverFinancial/ortex when consuming unreleased work; the
foundation stack does not publish a new Hex release.

## Development

Review `mise.toml`, run `mise trust` and `mise install`, then:

```sh
mise exec -- make setup
mise exec -- make doctor
mise exec -- make help
```

Supported BEAM pairs are Elixir 1.19 / OTP 28 and Elixir 1.20 / OTP 29. Rust 1.92
is the minimum. The source NIF currently retains ort rc.8 / Rustler 0.29; their
migration is separate feature work. Nx versions from 0.6.2 up to 1.0 are allowed
(excluding 1.0), with 0.13.1 locked for normal development.

CPU dynamic loading is the development default. ONNX Runtime is installed
separately by `make setup`. Make reads its recorded path; set ORT_DYLIB_PATH to
use another library. When consuming this source package, configure the mode
explicitly in your application's config:

```elixir
import Config
config :ortex, Ortex.Native, features: ["ort/load-dynamic"]
```

The consumer supplies ORT_DYLIB_PATH. Execution providers require a matching
runtime and explicit Cargo features; they are not inferred from the host OS.

## Checks and examples

```sh
make check
make test.system
make test.package
make test.models
make coverage
make docs
make test.alpine
make bench ARGS=5000
```

Library tests and doctests use normal ExUnit discovery under test/. System,
package, and model suites are documented in [testing](docs/testing.md). XGBoost,
Iris, and MNIST have prepared reference inputs and outputs; for example, with
ORT_DYLIB_PATH set:

```sh
MIX_ENV=test mix run examples/run.exs xgboost
```

Numeric outputs use Ortex.Backend for storage. Transfer to a computation backend
before arithmetic. See [examples](examples/README.md) and
[Nx.Serving](https://hexdocs.pm/nx/Nx.Serving.html) for usage.

[Development](docs/development.md), [CI policy](docs/ci.md), and
[release procedures](docs/releasing.md) describe the supported environment,
validation matrices, artifacts, and publishing gates.
