# Examples

Examples have an independent Mix project: from this directory, run `mix deps.get`
and `mix run distilbert/distilbert_classification.exs` (or another example).
Set ORT_DYLIB_PATH to the separately installed runtime. Model paths are relative
to this directory. The Python export dependencies are separate from the library.

The inherited DistilBERT and StableLM exports remain manual demonstrations. Their
model revisions and export environments have not yet been made reproducible;
they are not validation gates. The foundation test PR adds small prepared models
with recorded provenance and expected outputs.
