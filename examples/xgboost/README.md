# XGBoost

From the repository root, with ORT_DYLIB_PATH set:
`MIX_ENV=test mix run examples/run.exs xgboost`.

Eight shallow regression trees trained on deterministic synthetic data. Four
prepared rows are checked against the original estimator by `make test.models`.
See test/fixtures/manifest.json and fixtures/generate.py for provenance.
