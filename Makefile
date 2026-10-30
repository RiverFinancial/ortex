.DEFAULT_GOAL := help
PYTHON ?= python3
export MIX_ENV := test
export ORT_DYLIB_PATH ?= $(shell $(PYTHON) scripts/install_runtime.py path 2>/dev/null)

.PHONY: help setup deps doctor test docs

help: ## Show development commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Fetch Mix dependencies and the development runtime
	mix local.hex --force
	mix local.rebar --force
	$(MAKE) deps
	$(PYTHON) scripts/install_runtime.py install

deps: ## Fetch the locked test and documentation dependencies
	mix deps.get --check-locked
	MIX_ENV=docs mix deps.get --check-locked

doctor: ## Describe tools, architecture, and runtime selection
	$(PYTHON) scripts/doctor.py

test: ## Run library tests and doctests
	mix test

docs: ## Build documentation with warnings treated as errors
	MIX_ENV=docs mix docs --warnings-as-errors
