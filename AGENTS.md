# AGENTS.md

Guidance for AI coding agents working in this repo. Humans: read README.md and CHANGELOG.md.

## Overview

fastumap is a UMAP dimensionality-reduction and clustering library for environments where a JIT
compiler at import time is unacceptable (serverless functions, CPU-limited containers, autoscaled
workers). It reimplements UMAP on NumPy/SciPy with precompiled Rust kernels shipped in the wheel,
so nothing is left to compile at import — import time is ~12 ms versus ~21 s for umap-learn.

Architecture:

- `src/fastumap/` — Python package. Core modules: `projection.py` (UMAP fit/transform),
  `layout.py` (SGD), `neighbors.py` (kNN), `clustering.py` (k-means, spectral, dbscan, hdbscan),
  `graph.py` (fuzzy simplicial set), `metrics.py`.
- `rust/` — Rust extension, `fastumap._accel`. Implements the SGD layout kernel and the k-means
  SIMD GEMM kernel. Built by maturin (the build backend in `pyproject.toml`) as an abi3 extension
  module and shipped in every prebuilt wheel. The pure-NumPy path remains as a silent fallback
  when the extension is absent (e.g. an unbuilt source checkout).
- `fastumap._accel` is the module name; `fastumap.accelerator_active()` reports whether it is live.
- Lazy imports are intentional (they protect the fast cold start); do not hoist them to the top
  of a module.

## Build

- Toolchain (moon, uv) pinned in `.prototools`; run `proto install` to bring it up.
- `uv sync` installs Python deps **and** compiles the Rust extension via maturin. This is the
  normal dev setup step. A fresh clone without `uv sync` runs on the slower numpy fallback.
- Building from source (including `uv sync`) requires a Rust toolchain.
- `make build` (`uv build --out-dir dist`) builds the sdist and wheel into `dist/`.
- To force-rebuild the Rust extension without a full build: `uv sync --reinstall-package fastumap`.

## Test

- `make test` runs pytest (`uv run pytest`).
- `make check` = `make lint` + `make typecheck` + `make test`. Must pass before every commit.
- Coverage floor is 90% (enforced by pytest-cov via moon's `test` task). `grade.py` is excluded.
- CI: `moon run :ci` (lint + format-check + typecheck + test + version-guard). On `main`,
  `moon run :release` publishes to PyPI via OIDC trusted publishing if the version is new.
- Multi-Python tox sweep: `make tox` (Python 3.11, 3.12, 3.13).

## Style

- `make format` — `uv run ruff format src tests bench` (line length 100, target py311).
- `make lint` — `uv run ruff check src tests bench`.
- `make typecheck` — `uv run pyright` (strict mode on `src` and `tests`).
  - `reportMissingTypeStubs`, `reportUnknownMemberType`, `reportUnknownVariableType`,
    `reportUnknownArgumentType` are relaxed to accommodate numpy 2.x and scipy's missing stubs.
    Everything else in strict is enforced.
- Keep `src/` and `tests/` fully annotated.

## Gotchas

- **Version lives only in `pyproject.toml`** (static `version = "..."`). Use `make bump TYPE=patch`
  (or `minor`/`major`) to bump; never hand-edit or copy the version elsewhere. CI's
  `scripts/ci-version-guard` enforces that any `src/` change carries a bump.
- **Rust extension must be compiled before tests are meaningful.** After any change under `rust/`,
  run `uv sync` (or `uv sync --reinstall-package fastumap`) to rebuild `fastumap._accel`. If you
  skip this, tests still pass on the numpy fallback but do not exercise the native kernel.
- **`make check` does not rebuild the wheel.** `moon run :test` is aware of `rust/**/*.rs` as an
  input and triggers a rebuild through maturin when those files change; `make test` does not.
- **No numba or llvmlite at import.** A test asserts this. Never add a dependency that pulls in
  numba (even transitively) into the runtime deps.
- **Approximate kNN (`fastumap[ann]`) changes reproducibility.** Above ~10k points the exact
  O(n²) kNN dominates; install `fastumap[ann]` (faiss-cpu) for the approximate backend. The dev
  group already includes `faiss-cpu` so tests exercise both paths; production installs may not.
- **No prebuilt wheel for macOS.** It builds from the sdist and requires a Rust toolchain.
  Linux wheels are abi3 (manylinux + musllinux, x86_64 + aarch64). A Windows abi3 wheel is built
  in CI as a best-effort job (`wheels-windows`, `allow_failure: true`).
- **Track follow-ups in `docs/issues/`**, not as scattered TODO comments.
