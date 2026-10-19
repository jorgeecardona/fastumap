# fastumap -- human task verbs. The CI-graded tasks (lint/typecheck/test/build/docs)
# delegate to the matching moon task, so the command + its inputs/outputs live in
# ONE place (moon.yml) and `make` runs exactly what CI runs, with moon's caching.
# The rest (sync, format = write, the bench/grade drivers, tox, bump, install-hooks)
# have no moon task and run directly here.
# One-time setup after cloning:  make install-hooks
.DEFAULT_GOAL := help
.PHONY: help sync lint format typecheck test check docs build grade bench bench-report tox fargate constrained minibatch-1m bump install-hooks

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## install Python deps + compile the Rust extension (maturin)
	uv sync

lint: ## lint (ruff) -- the CI :lint task
	moon run :lint

format: ## auto-format (ruff) over src tests bench
	uv run ruff format src tests bench

typecheck: ## type-check (pyright) -- the CI :typecheck task
	moon run :typecheck

test: ## run the test suite with coverage -- the CI :test task
	moon run :test

check: lint typecheck test ## the full CI test stage: lint + typecheck + test

docs: ## build the mkdocs site strictly (as the pages job / Read the Docs do) -- the CI :docs task
	moon run :docs

build: ## build sdist + wheel into dist/ -- the CI :build task
	moon run :build

grade: ## print the grading table (fastumap vs umap-learn vs PCA vs random) over five seeds
	uv run python -m fastumap.grade

bench: ## run the MNIST benchmark
	uv run python bench/mnist.py

# Regenerate every benchmark number the READMEs quote into a dated report under docs/bench/.
# Slow on purpose (real MNIST fits + k-means); this is the one command that makes the README
# figures reproducible instead of hand-pasted.
bench-report: ## regenerate the README benchmark numbers into a dated report under docs/bench/
	uv run python bench/report.py

tox: ## run the suite across Python 3.11/3.12/3.13 (uv provisions each interpreter)
	uv run tox

# Run under a Fargate-like CPU/memory cap (needs Docker): import + fit timing on 0.5 vCPU.
fargate: ## import + fit timing under a Fargate-like 0.5 vCPU cap (needs Docker)
	sh bench/fargate.sh

# Fargate-shaped probe, but the full curve: fit AND cluster wall-clock at several n under 0.5, 1
# and 2 vCPU, into a dated table. Needs Docker and a wheel in dist/ (`make build`). Slow (~15 min).
constrained: ## full fit+cluster wall-clock curve under 0.5/1/2 vCPU caps (needs Docker + a wheel)
	sh bench/constrained.sh

# The roadmap's mini-batch bar: 1M x 128 into ~64 groups under 10 s on 2 vCPU, inertia within ~5%
# of full Lloyd. Needs Docker, a wheel in dist/ (`make build`) and the cached SIFT1M descriptors.
minibatch-1m: ## mini-batch k-means bar: 1M x 128 under 10 s on 2 vCPU (needs Docker + a wheel)
	sh bench/minibatch_1m.sh

bump: ## bump version in pyproject.toml -- TYPE=patch|minor|major (default patch)
	sh scripts/bump-version $(or $(TYPE),patch)

install-hooks: ## point git at the tracked .githooks/ (pre-commit + pre-push)
	git config core.hooksPath .githooks
