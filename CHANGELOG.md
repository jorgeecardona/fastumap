# Changelog

All notable changes to fastumap are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow SemVer.

## [Unreleased]

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
