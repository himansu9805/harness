.DEFAULT_GOAL := help

# Run tools through Poetry so they use the project's pinned versions and config.
POETRY := poetry
RUN := $(POETRY) run
SRC := harness tests

.PHONY: help install format lint isort flake8 pylint test check pre-commit clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and register git hooks.
	$(POETRY) install --with dev
	$(RUN) pre-commit install

format: ## Auto-fix import ordering in place.
	$(RUN) isort $(SRC)

isort: ## Check import ordering.
	$(RUN) isort --check-only --diff $(SRC)

flake8: ## Run flake8 style checks.
	$(RUN) flake8 $(SRC)

pylint: ## Run pylint static analysis.
	$(RUN) pylint harness

lint: isort flake8 pylint ## Run all linters (isort check, flake8, pylint).

test: ## Run the test suite.
	$(RUN) pytest

check: lint test ## Run linters and tests (what CI should run).

pre-commit: ## Run all pre-commit hooks against every file.
	$(RUN) pre-commit run --all-files

clean: ## Remove Python caches and build artifacts.
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete
	rm -rf .pytest_cache
