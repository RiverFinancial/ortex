# Iris

From the repository root, with ORT_DYLIB_PATH set:
`MIX_ENV=test mix run examples/run.exs iris`.

StandardScaler and LogisticRegression produce class labels and a probability
matrix. `make test.models` compares both against scikit-learn. See
test/fixtures/NOTICE.md for attribution and dataset license information.
