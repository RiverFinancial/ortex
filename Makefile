.DEFAULT_GOAL := help
PYTHON ?= python3
export PATH := $(CURDIR)/.venv/bin:$(CURDIR)/.venv/Scripts:$(CURDIR)/.cache/audit/bin:$(PATH)
export MIX_ENV := test
export ORT_DYLIB_PATH ?= $(shell $(PYTHON) scripts/install_runtime.py path 2>/dev/null)

.PHONY: help setup deps doctor test docs hooks fmt lint check
.PHONY: test.system test.package test.models test.tooling coverage bench fixtures package
.PHONY: test.alpine
.PHONY: setup.audit audit

help: ## Show development commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Fetch Mix dependencies and the development runtime
	mix local.hex --force
	mix local.rebar --force
	$(MAKE) deps
	$(PYTHON) scripts/install_runtime.py install
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -r scripts/requirements.txt

deps: ## Fetch the locked test and documentation dependencies
	mix deps.get --check-locked
	MIX_ENV=docs mix deps.get --check-locked

doctor: ## Describe tools, architecture, and runtime selection
	$(PYTHON) scripts/doctor.py

test: ## Run library tests and doctests
	mix test

docs: ## Build documentation with warnings treated as errors
	MIX_ENV=docs mix docs --warnings-as-errors

hooks: ## Install the pre-commit launcher (refuses to overwrite an unrelated hook)
	$(PYTHON) scripts/hooks.py

fmt: ## Apply the language formatters
	mix format
	cd native/ortex && cargo fmt
	ruff format .
	shfmt -w scripts/precommit

lint: ## Check formatting, static analysis, and compiler warnings
	$(PYTHON) scripts/check.py --full

check: lint test ## Run the ordinary local checks

test.system: ## Run each system scenario in a fresh VM
	$(PYTHON) scripts/scenario.py

test.package: ## Test the assembled package in clean source and prebuilt consumers
	$(PYTHON) scripts/package.py test --mode source
	$(PYTHON) scripts/package.py test --mode prebuilt

test.models: ## Check prepared model inputs against their reference outputs
	$(PYTHON) scripts/fixtures.py
	mix test test/models --include model

test.tooling: ## Test the development harnesses
	$(PYTHON) -m unittest discover -s test/tooling -p '*_test.py' -v

coverage: ## Write scoped Elixir HTML and Markdown coverage reports
	mix test --cover
	mix run --no-start --no-compile scripts/coverage.exs

bench: ## Measure the public inference workload (ad hoc)
	mix run bench/inference.exs $(ARGS)

fixtures: ## Regenerate model fixtures (requires the separate exporter environment)
	$(PYTHON) fixtures/generate.py

package: ## Assemble the local Hex archive
	$(PYTHON) scripts/package.py build

test.alpine: ## Run the same suites in the pinned Alpine environment
	docker build -f ci/containers/alpine.Dockerfile -t ortex-foundation-alpine .
	docker run --rm ortex-foundation-alpine

setup.audit: ## Install the pinned advisory checker into the local cache
	cargo install cargo-audit --version 0.22.2 --locked --root .cache/audit

audit: ## Report Rust advisories and retired Hex dependencies
	$(PYTHON) scripts/audit.py
