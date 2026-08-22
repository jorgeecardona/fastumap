# Changelog

All notable changes to fastumap are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow SemVer.

## [Unreleased]

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
