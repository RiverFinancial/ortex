# Testing

`mix test` discovers library `*_test.exs` files normally under test/, including
doctests. Mix excludes system/, package/, support/, and fixtures/ from discovery.
Model ExUnit tests have `@moduletag :model` and are excluded by default; run
`make test.models` to select them explicitly. No selected suite silently skips
a missing fixture.

`make test.system` invokes scripts/scenario.py. Each test/system/*.exs scenario
runs in its own BEAM after compilation. Nonzero exit (including a native shutdown
abort) fails. Timeout kills the process group on Unix and process tree on Windows.
The runner stores exit status and output in dist/reports/system.json. Additional
process-wide configuration scenarios should accompany the feature implementing
that configuration; the foundation tests main's existing behavior.

`make test.package` assembles a Hex tarball, extracts its actual contents into a
temporary directory, and runs the separate test/package project against it. It
checks source compilation and an explicitly configured prebuilt library with
Cargo/rustc replaced by failing stubs. The runtime path comes from the caller's
ORT_DYLIB_PATH. The package check also rejects a missing runtime in a separate VM; ort rc.8's
legacy behavior is a caught NIF panic, not the migrated descriptive loader error.
This is a package installation check, not the feature stack's
automatic RustlerPrecompiled download path. That feature can reuse this harness.

`make test.models` checks XGBoost, Iris, and MNIST. Prepared artifacts and reference
cases are recorded in test/fixtures/manifest.json. XGBoost and Iris results come
from the original estimators; MNIST uses Python ORT as a binding reference, with
synthetic pixels. None of these checks claim to validate ORT's operators or model
accuracy. Iris uses tensor probabilities on main; ZipMap needs the separate
container-output feature. Install fixtures/requirements.txt in a separate Python
3.12 environment before `make fixtures PYTHON=/path/to/exporter/bin/python`. Generation validates exports with ONNX
and Python ORT; ordinary tests need neither Python exporters nor network access.

`make test.tooling` checks the harnesses themselves, including partial staging,
filenames with shell characters, timeout handling, and archive path traversal.

`make coverage` reports Elixir library lines using built-in OTP coverage. The
NIF interface is excluded; generated code should also be excluded when introduced.
The HTML and Markdown report are under cover/. Rust and fresh-VM scenarios are
outside that metric. Initially coverage is reported without a percentage gate.

`make bench ARGS=5000` runs the ad hoc CPU XGBoost benchmark with reference checks
on every call. Timings include tensor transfer; they are observations for the
recorded environment, not pass/fail performance guarantees.

The inherited ResNet, DistilBERT, and StableLM demos are manual examples. Their
export dependencies and weights are not part of normal checks. The original
tiny multi-input model remains for legacy API and dtype tests.
