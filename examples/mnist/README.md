# MNIST

From the repository root, with ORT_DYLIB_PATH set:
`MIX_ENV=test mix run examples/run.exs mnist`.

A small trained convolutional model consumes deterministic synthetic pixels.
`make test.models` compares its numerical outputs against Python ORT; this checks
binding and tensor layout rather than classification accuracy. The exact model
revision, digest, and notices are under test/fixtures/.
