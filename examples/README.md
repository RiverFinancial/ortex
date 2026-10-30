# Examples

Small XGBoost, Iris, and MNIST examples run from the repository root via
`MIX_ENV=test mix run examples/run.exs NAME`, with ORT_DYLIB_PATH set. Their
prepared artifacts also power `make test.models`.

The tokenizer examples have an independent Mix project: from this directory, run `mix deps.get`
and `mix run distilbert/distilbert_classification.exs` (or another example).
Set ORT_DYLIB_PATH to the separately installed runtime. Model paths are relative
to this directory. The Python export dependencies are separate from the library.

The inherited DistilBERT and StableLM exports remain manual demonstrations. Their
model revisions and export environments have not yet been made reproducible;
they are not validation gates. The foundation test PR adds small prepared models
with recorded provenance and expected outputs.
