# Changelog

All notable changes to fastumap are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow SemVer.

## [Unreleased]

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
