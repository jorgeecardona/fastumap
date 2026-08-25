# Changelog

All notable changes to fastumap are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow SemVer.

## [Unreleased]

## [0.2.3] - 2026-08-25

### Added
- `assign_clusters(embeddings, centroids, *, metric=...)` — the predict half of a fit/predict
  split. Assigns new rows to the nearest of a fitted `cluster(..., return_centroids=True)` /
  `kmeans` centroid set without re-clustering, reusing the same blocked nearest-centroid step so
  peak memory stays bounded. `metric` must match the fit (`"cosine"` unit-normalises, as fitting
  did). Exported at the top level.

## [0.2.2] - 2026-08-25

### Added
- `cluster(..., return_centroids=True)` returns a `ClusterResult` (labels, centroids, inertia)
  instead of just labels, so you can compare runs or pick `n_clusters` by inertia without a second
  pass. `inertia` is the k-means objective — the sum of squared distances to the assigned centroid,
  computed one block at a time so the (n, d) difference is never fully materialised. `ClusterResult`
  is exported at the top level. The default (`return_centroids=False`) is unchanged — still a plain
  label array. Works for both `method="kmeans"` and `method="spectral"` (centroids in the input
  space).

## [0.2.1] - 2026-08-25

### Fixed
- CI: the aarch64/Graviton wheel is published again. The 0.2.0 pipeline's `--zig` cross-compile
  failed with "Failed to find zig" because maturin runs `python3 -m ziglang` and, without the
  venv first on `PATH`, that resolved to the system python (no ziglang); 0.2.0 shipped x86_64 +
  sdist only. The job now puts `.venv/bin` first on `PATH` before the maturin builds, so the
  aarch64 wheel builds and publishes as the README states.

## [0.2.0] - 2026-08-25

### Changed
- **The native kernels are now built into `fastumap` itself** — there is no separate
  `fastumap-accel` package any more. `fastumap` is a maturin-built package: the pure-Python
  `src/fastumap/` plus a compiled Rust extension (`fastumap._accel`, the SGD + k-means kernels)
  shipped in the same abi3 wheel (`cp311-abi3`; Linux x86_64 and aarch64/Graviton, with an sdist
  for other platforms). The kernels are used automatically — no `pip install fastumap[accel]`,
  no accel-detection dance. `import fastumap` stays in the ~15 ms range because the extension is
  imported lazily. On a platform without a prebuilt wheel, pip builds from the sdist, which needs
  a Rust toolchain. The pure-NumPy path remains only as a runtime fallback for an unbuilt source
  checkout (`accelerator_active()` reports which path a run used).
- The tagline drops "pure NumPy and SciPy": fastumap now compiles native kernels at **build**
  time. The value is a fast import (no import-time JIT) and sensible defaults, not purity.
- `pip install fastumap[accel]` is removed (the accelerator is not optional any more).
  `pip install fastumap[ann]` (approximate kNN via faiss) is unchanged.

## [0.1.10] - 2026-08-23

### Added
- Optional native accelerator wiring. If `fastumap-accel` (a Rust reimplementation of the
  SGD, `rust/`) is importable, `optimize_layout` uses it for the default `chunk_count<=1`
  path; otherwise the pure-numpy path runs unchanged. The base install stays numpy+scipy —
  the accelerator is opt-in (`pip install fastumap-accel`). It does umap-learn's true
  in-place walk, so it is **faster and higher quality**: measured ~1.7–1.9× vs the numpy SGD
  with better overlap (0.244 vs 0.233 at n=1000). Deterministic; a stub-less compiled module
  so it is imported dynamically. `tests/test_accel.py` covers the fast path (skipped when
  absent). The accel wheel is abi3 (`cp311-abi3`), one build for Python 3.11 and later.

## [0.1.9] - 2026-08-23

### Changed
- `UMAPModel.train` is now cached as **float32**, halving the model's memory footprint
  (~40 MB → ~20 MB for a 5000 × 1024 model — the concern raised in GitLab #4, and what makes
  many cached copies affordable on a memory-constrained Lambda/ECS worker). `transform`
  upcasts to float64 for the distance compute, so quality and determinism are unchanged.

## [0.1.8] - 2026-08-23

### Added
- `pca_dim` on `umap_project` / `fit` / `spectral_project` (default `None` = off): pre-reduce
  the input to that many principal components (exact SVD) before the kNN. The kNN cost grows
  with the feature dimension, so for wide inputs (1024-dim embeddings) `pca_dim=100` shrinks
  the dominant cost; embeddings are low-rank, so overlap is preserved (within ~0.01 in tests,
  e.g. 0.559 → 0.547 reducing 64→16 dims). `UMAPModel` stores the PCA basis so `transform`
  projects new points the same way. Deterministic; the off path is byte-identical to before.

### Notes
- This release also carries the `fit`/`transform` server-usage docs (GitLab #3/#4) onto the
  PyPI project page, which the 0.1.7 release predated.

## [0.1.7] - 2026-08-23

### Fixed
- `fastumap.__version__` now derives from the installed distribution metadata
  (`importlib.metadata`) instead of a hardcoded string, so it can no longer drift from the
  published version (GitLab #1 — 0.1.6 reported `0.1.0`). pyproject stays the single source
  of truth (like musil); a test asserts `__version__` == metadata == pyproject.

### Added
- README "Using this in a server" section and a thread-safety guarantee bullet (GitLab #3):
  safe to call from `asyncio.to_thread`, and the fit-once / `transform`-per-request pattern
  for a long-lived server.

## [0.1.6] - 2026-08-23

### Changed
- Sparse input decision (#16): **not supported**. `umap_project`/`fit`/`transform` take a
  dense numpy array; a `scipy.sparse` matrix now raises a clear `TypeError` pointing to
  `.toarray()` instead of a cryptic failure. Rationale documented in README ("Sparse
  input"): the blocked kNN uses dense BLAS matmul and the target workload is dense
  encoder/CLS embeddings. Revisit if a sparse workload actually appears.

## [0.1.5] - 2026-08-23

### Added
- `fit()` → `UMAPModel` and `transform(model, new)` (roadmap #13): place new points into an
  existing layout without refitting — what makes a drift view incremental instead of a full
  recompute per request. New points get their neighbours in the training set (`knn_between`),
  a membership-weighted-average seed, then a fixed-tail optimiser (`optimize_transform`) that
  moves only the new points while the training layout stays frozen. Deterministic, does not
  mutate the base layout, and lands new points near same-cluster training points (>0.8 in
  the test). `umap_project` is now `fit(...).embedding`. `compute_membership` gains
  `exclude_self` (off for transform, where neighbours come from a different set).

## [0.1.4] - 2026-08-23

### Added
- `metric="cosine"` on `umap_project` / `spectral_project` (roadmap #14) — the metric that
  matters for text/CLS embeddings. Implemented as a blocked-brute-force cosine kNN
  (`knn_cosine`): L2-normalise rows once, then the block dot product is cosine similarity,
  distance = 1 − sim. Deterministic. A test shows it beats euclidean on directional
  clusters (same direction, magnitudes spanning 0.1–10). `knn`/`knn_cosine` are the new
  neighbour entry points; an unknown metric raises `ValueError`.

## [0.1.3] - 2026-08-23

### Changed
- `optimize_layout` scatter: replace `np.add.at` with one per-dimension `np.bincount`
  reduction (#19). ~31% faster at n=1000×1024 (7.7s → 5.3s single-thread), ~19% at
  n=5000 (39.5s → 32.1s). overlap@15 unchanged within noise (n=1000 0.2338 → 0.2331,
  n=2000 0.1387 → 0.1381); determinism preserved. Bytes differ from 0.1.2 (bincount sums
  in a different float order), which is expected and does not affect quality.

## [0.1.2] - 2026-08-23

### Fixed
- `compute_membership`: clamp the exponent argument at 0 before `np.exp` so the discarded
  `np.where` branch no longer raises `overflow encountered in exp` at scale. Output is
  value-identical (only the masked-out branch changed); determinism preserved.

### Added
- Peak-RSS ceiling test (roadmap #9): a subprocess measures peak RSS for a 5000 × 1024 fit
  and asserts it stays under 200 MB. Measured ~102 MB added over the import+input baseline
  (175 MB total peak), consistent with the 512-row blocked kNN never materialising n-by-n.

## [0.1.1] - 2026-08-23

### Added
- `chunk_count` knob on `umap_project` / `optimize_layout` (roadmap #6). Default `1` keeps
  the fast per-epoch snapshot update unchanged; higher values process each epoch's edges in
  that many shuffled chunks so later chunks see earlier moves (partial in-epoch feedback),
  trading speed for local overlap. Measured: overlap@15 at n=1000 rises 0.234 → 0.244
  (chunk_count=10) → 0.248 (chunk_count=40 ≈ umap-learn), at ~5×/~10× cost; n=2000 0.139 →
  0.147 (chunk_count=10). Stays deterministic.

### Notes
- #5 (per-edge negative-sample schedule) was measured neutral vs the flat rate and reverted.

## [0.1.0] - 2026-08-22

Initial port of a working single-file numpy+scipy UMAP into a typed, tested library.

### Added
- `umap_project(matrix, dimensions, ...)` — full fuzzy-simplicial-set UMAP in pure
  numpy + scipy: blocked brute-force kNN, `smooth_knn_dist` (vectorised binary search),
  fuzzy-union symmetrisation, spectral initialisation via `scipy.sparse.linalg.eigsh`,
  `find_ab_params` via `scipy.optimize.curve_fit`, and a vectorised attract/repel SGD.
- `spectral_project(matrix, dimensions, ...)` — the spectral initialisation alone.
- Metrics: `neighbor_overlap`, `global_distance_correlation`, `random_layout`.
- 2D and 3D output, both first-class.
- Deterministic given a seed (pinned ARPACK start vector + seeded negative sampler).
- Full type annotations, pyright strict.
- musil-style CI: moon + uv + proto, OIDC Trusted Publishing to PyPI.
