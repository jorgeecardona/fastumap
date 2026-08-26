.PHONY: sync lint format typecheck test check build grade bench bench-report tox fargate constrained install-hooks

sync:
	uv sync

lint:
	uv run ruff check src tests bench

format:
	uv run ruff format src tests bench

typecheck:
	uv run pyright

test:
	uv run pytest

# Mirror CI's gate locally.
check: lint typecheck test

build:
	uv build --out-dir dist

# Print the grading table (fastumap vs umap-learn vs PCA vs random) over five seeds.
grade:
	uv run python -m fastumap.grade

bench:
	uv run python bench/mnist.py

# Regenerate every benchmark number the READMEs quote into a dated report under docs/bench/.
# Slow on purpose (real MNIST fits + k-means); this is the one command that makes the README
# figures reproducible instead of hand-pasted.
bench-report:
	uv run python bench/report.py

# Run the suite across Python 3.11/3.12/3.13 (uv provisions each interpreter).
tox:
	uv run tox

# Run under a Fargate-like CPU/memory cap (needs Docker): import + fit timing on 0.5 vCPU.
fargate:
	sh bench/fargate.sh

# Fargate-shaped probe, but the full curve: fit AND cluster wall-clock at several n under 0.5, 1
# and 2 vCPU, into a dated table. Needs Docker and a wheel in dist/ (`make build`). Slow (~15 min).
constrained:
	sh bench/constrained.sh

bump:
	sh scripts/bump-version $(TYPE)

install-hooks:
	git config core.hooksPath .githooks
