# Changelog

All notable changes to fastumap are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow SemVer.

## [Unreleased]

## [0.2.23] - 2026-08-27

### Added
- `umap_project(return_model=True)` returns `(embedding, model)` so a fit can be pinned and reused.
  `UMAPModel.transform(new_points)` places new points into that same layout without refitting (a
  method over the existing module-level `transform`). This is what keeps a scatter stable across a
  scope change: the same points land in the same places instead of a refit reshuffling the axes.
- `assign_clusters` gains `return_distances=True` (also returns the distance to the assigned
  centroid per row) and `max_distance=...` (points beyond it are labelled `-1` noise instead of
  being absorbed into the least-far cluster). The distance is measured in the space the fit ran in
  (unit-normalised rows for cosine).
- `umap_project` now accepts `dens_frac` and `dens_var_shift`, matching `fit` (they were reachable
  through `fit` but not through the `umap_project` wrapper).

### Notes
- Placing new points into pinned k-means centroids (fit once, assign later against the same groups)
  is already supported by `assign_clusters`, which is metric-aware and raises on a dimension
  mismatch. No separate `assign` function was added, to keep one name per concept.

## [0.2.22] - 2026-08-27

### Added
- `choose_k(return_evidence=True)` returns a `ChooseKResult` (the chosen k, the per-candidate
  inertias, the sample size actually used, and `n_iter`/`random_state`) instead of the bare
  integer, so a caller can publish the evidence behind the choice, not just the number. The
  docstring now names the selection criterion (the inertia elbow) explicitly, distinct from the
  distance `metric`. `ChooseKResult` is exported from the top-level package.
- `choose_k(n_jobs=...)` runs the independent per-k fits in parallel. Each fit is pinned to one
  BLAS thread for the sweep, so `n_jobs` of them run concurrently without oversubscribing, and the
  result is bit-identical to `n_jobs=1`. On a 17-candidate sweep this turns a sequential ~17-fit
  cost into roughly `17 / n_jobs`.

## [0.2.21] - 2026-08-27

### Changed
- Internal cleanups only; no public behavior change. Deduplicated three copies of the union-find
  root lookup (dbscan, the mutual-reachability MST, single linkage) into one `_uf_find`, the two
  copies of the repulsion-gradient SGD block (`optimize_layout`, `optimize_transform`) into one
  `_repel_grad`, and the two identical `_unit_rows` (neighbors, clustering) into a shared
  `_types.unit_rows`. Removed the `local_connectivity` parameter from `smooth_knn_dist` and
  `fuzzy_simplicial_set`: it was threaded through but discarded, and only the UMAP default (1.0)
  is implemented.
- `pyproject.toml` now lists the Issues tracker and Documentation site under `project.urls`.

## [0.2.20] - 2026-08-27

### Fixed
- `cluster(method="kmeans")` and `cluster(method="spectral")` called with the documented default
  `n_clusters=None` raised `ValueError` instead of clustering. `None` now means "pick k", resolved
  by `choose_k` internally.
- `transform()` raised "need at least 2 points" on a single new point. Placing one point into a
  fitted layout is well defined (one query per request is a primary serving path); only `fit` and
  `spectral_project` still require n >= 2.
- `cluster(n_neighbors=...)` was silently ignored for `method="dbscan"` and `method="hdbscan"`. It
  is now forwarded. The default is unchanged: `n_neighbors=None` uses each method's own default
  (30 for the density methods, 15 for spectral), so existing results are stable.
- densMAP returned an all-NaN embedding on degenerate input (all-duplicate rows, or any perfectly
  regular graph), because the local-radius z-score divided by a zero standard deviation. A flat
  radius field is now zero.
- `neighbor_overlap(a, b, k=0)` raised `ZeroDivisionError`; it now raises a clear `ValueError`.
- Native k-means was not bit-identical across machines with different core counts. Two parallel
  f32 reductions were order-sensitive: the k-means++ seed selection (an order-dependent low bit
  picked a different seed and diverged the whole run) and the centroid means. Both now use a
  fixed-order f64 accumulation, so labels and centroids are identical whatever the vCPU count. This
  is the guarantee the "bit-identical across machines" claim needs for CPU-constrained containers
  with varying vCPU allotments.

### Added
- BLAS thread capping (`with_blas_limit`) now also wraps every public clustering entry point
  (`cluster`, `kmeans`, `dbscan`, `hdbscan`, `choose_k`, `assign_clusters`) and `spectral_project`,
  not just `fit`/`transform`. Clustering is the workload that runs in CPU-constrained containers,
  so it needs the same cap against BLAS oversubscription.
- `kmeans` is now exported from the top-level `fastumap` package (it was reachable only via
  `fastumap.clustering`), so the README examples that use it run as written.

## [0.2.19] - 2026-08-27

### Changed
- Documentation rewritten in a neutral, declarative register: the README (and therefore the PyPI
  long description), the PyPI summary, and the docs site pages. No facts, numbers, tables or code
  samples changed; the emphasis markup, rhetorical framing and second-person coaching did.
- The opening no longer leads with import time as the headline claim. Import cost is a mechanism,
  not a benefit, and on its own it reads as a micro-benchmark. It now leads with the constraint the
  library exists for: environments where a JIT compiler at import time is not acceptable
  (serverless functions, small or CPU-limited containers, autoscaled workers paying a cold start
  per process). The import table stays, as evidence rather than as the claim.

## [0.2.18] - 2026-08-26

### Added
- **Windows x86_64 wheel.** `pip install fastumap` on Windows no longer needs a Rust toolchain —
  it gets a prebuilt `cp311-abi3-win_amd64` wheel like every other supported platform. Built by a
  new `wheels-windows` CI job on GitLab's hosted Windows runner, which is available on the Free
  plan at cost factor 1; no GitHub Actions mirror was needed. VERIFIED on the PyPI files list for
  0.2.17: `fastumap-0.2.17-cp311-abi3-win_amd64.whl`.

### Changed
- The README's accelerator section listed only Linux x86_64 and aarch64 as prebuilt. It now names
  every platform that actually ships a wheel (Linux x86_64/aarch64, musllinux, Windows x86_64) and
  says plainly that macOS still builds from the sdist.

### Known gaps
- Still no macOS wheel. GitLab's hosted macOS runners are Premium/Ultimate only and this namespace
  is Free, so it is an account-plan decision rather than a CI problem. The two paths — the GitLab
  for Open Source program, or a GitHub Actions mirror for that one wheel — are written up in
  `docs/issues/25-08-2026-macos-windows-wheels-in-ci.md`.

## [0.2.17] - 2026-08-26

### Added
- **Mini-batch k-means** (Sculley 2010) in the native kernel, and `kmeans(..., batch_size=...)` to
  control it. Full Lloyd sweeps all `n` rows every iteration; mini-batch samples `batch_size` rows
  and moves only the centroids they land on, so per-iteration work drops from `O(n*k*d)` to
  `O(batch*k*d)`. It engages automatically above 200,000 rows.

  MEASURED on SIFT1M (1,000,000 x 128, k=64) inside a `docker --cpus=2` container: **5.2 s against
  full Lloyd's 25.3 s, at 1.010x the inertia** — 4.9x faster for a 1% worse objective, clearing the
  roadmap's bar of under 10 s within ~5%. On MNIST 60000x784 it lands at 1.013x Lloyd.

  The switch is on `n` **alone, never on core count**, so the same input and seed give the same
  labels on any machine. `batch_size=0` forces full Lloyd at any size; a positive int forces
  mini-batch with that batch.

  KNOWN WEAKNESS, pinned by a test rather than hidden: on *perfectly separated* clusters mini-batch
  can leave two centroids inside one group and none in another (measured 4.8x worse inertia on
  synthetic blobs 8 sigma apart), because the running-mean update cannot walk a centroid back
  across an empty gap the way Lloyd's recompute can. Real, overlapping data does not trigger it.
- `make minibatch-1m` (`bench/minibatch_1m.sh` + `bench/minibatch_1m.py`) — the probe behind those
  numbers, writing a dated report to `docs/bench/`.

### Changed
- The README's clustering section no longer tells you to avoid clustering on a fractional-vCPU
  box. That advice was written before this kernel existed and is now measurably wrong.

## [0.2.16] - 2026-08-26

### Added
- **`make constrained`** (`bench/constrained.sh` + `bench/constrained.py`) — a reproducible
  constrained-vCPU probe. Runs `umap_project` *and* `cluster` at several n under
  `docker --cpus=0.5`, `1` and `2` (a CFS quota, the same mechanism AWS Fargate's cgroup uses to
  meter CPU) and writes a dated table to `docs/bench/`. It installs a wheel built from the working
  tree rather than `pip install .`, so the container needs no Rust toolchain and what is measured
  is the artifact PyPI ships. Every run compares its own 0.5-vCPU fit times against the observed
  production Fargate numbers, so drift shows up as a row moving away from them.

  MEASURED, and it does **not** match production in absolute terms: at 0.5 vCPU this probe fits
  n=250/500/1000/5000 in 3.5/2.8/5.6/33.2 s against Fargate's observed 7.6/28/83/270 s. A local cap
  reproduces Fargate's mechanism but not its hardware — half a core of a workstation CPU is far
  more compute than half a shared Fargate vCPU, and the container still sees every host core. The
  probe's value is drift on a fixed host, not absolute parity; the report says so.

## [0.2.15] - 2026-08-26

### Added
- **Docs site** — a navigable MkDocs Material site published to GitLab Pages by a `pages` job on
  `main`: Home, Projection, Clustering, In a server, Speed and quality, Large inputs, The native
  accelerator, Benchmark reports, Guarantees and reproducibility, API, Contributing. The API page
  is generated from the source by mkdocstrings. Every prose page pulls its text out of `README.md`
  with pymdownx snippet sections, so the README stays the single source of truth and the site
  cannot drift from it. The `pages` job installs only the `docs` extra (`--no-install-project`),
  so it needs no Rust toolchain and builds in seconds; `--strict` fails it on a broken cross-link
  rather than publishing one.
- `bench/report.py` also writes `docs/bench/latest.md` beside the dated report, so the site's
  benchmarks page always shows the newest run with no page edit.

### Changed
- The `docs` extra now includes `ruff`; mkdocstrings uses it to format generated signatures.

## [0.2.14] - 2026-08-26

### Added
- `make bench-report` (`bench/report.py`) — one command that regenerates every benchmark number
  the READMEs quote into a dated report under `docs/bench/`: projection quality (fastumap vs
  umap-learn vs PCA on MNIST — wall time, overlap@15, global distance correlation) and k-means
  (wall time *and* inertia, native kernel vs numpy). The report records the CPU, core count, RAM,
  platform, Python and every package version, plus the load average — and prints a warning banner
  when the machine was busy enough to distort wall times.

### Changed
- The README's "Quality relative to umap-learn" table is now generated by that command instead of
  pasted by hand, and its numbers are re-measured. The previous table claimed fastumap was much
  slower at n=20000 (110 s vs umap-learn's 31 s); the regenerated run measures 57 s vs 69 s, so
  that paragraph was rewritten. PCA is now shown as the control in the table.

## [0.2.13] - 2026-08-26

### Changed
- The x86_64 manylinux wheel is now cross-built with zig against glibc 2.17, so it is tagged
  `manylinux_2_17_x86_64.manylinux2014_x86_64` instead of the builder image's own
  `manylinux_2_34` (MEASURED locally: `fastumap-0.2.12-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`).
  Hosts with glibc < 2.34 — Ubuntu 20.04, Amazon Linux 2, older CentOS — get the wheel instead of
  falling back to the sdist + a Rust toolchain. The native build stays as a fallback, so a zig
  flake still ships an x86_64 wheel with the old, narrower floor. CI + packaging only.

## [0.2.12] - 2026-08-26

### Added
- **`hdbscan(...)` and `cluster(method="hdbscan")`** — hierarchical density clustering. Same job as
  `dbscan` but with no `eps` to pick, so clusters of *different* densities are found together:
  core distances inflate the kNN edges into mutual-reachability distances, an MST over those gives
  the single-linkage hierarchy, branches under `min_cluster_size` are condensed away as noise, and
  the surviving branches are chosen by stability (excess of mass). Verified against scikit-learn's
  `HDBSCAN` (adjusted Rand > 0.95) on a dense/diffuse pair that *no* single `eps` can separate.
  `min_cluster_size` (default 5) is the knob; `min_samples` defaults to it. Approximate HDBSCAN:
  the MST is built over the `n_neighbors` kNN edges, and a disconnected kNN graph is bridged at
  infinite distance. The root is never a cluster, so single-group data comes back all `-1` (same as
  scikit-learn's `allow_single_cluster=False`). Exported at the top level.

## [0.2.11] - 2026-08-25

### Added
- **densMAP** — `umap_project(x, 2, densmap=True)` (and `fit(densmap=True)`), the density-preserving
  UMAP variant. It keeps dense regions dense and sparse regions sparse instead of flattening local
  density. A faithful port of umap-learn's density gradient: verified to lift the original-vs-
  embedding density correlation from ~0.02 to ~0.90 on a mixed-density dataset, matching umap-learn's
  own ~0.02→0.92. `dens_lambda` (default 2.0) sets the strength; the density term runs in the final
  `dens_frac` (0.3) of epochs after a plain-UMAP warmup. It forces the numpy SGD path (the Rust
  kernel does plain UMAP), so it is slower.

## [0.2.10] - 2026-08-25

### Added
- **`dbscan(...)` and `cluster(method="dbscan")`** — density clustering. Finds clusters of any
  shape and labels outliers `-1`, with no `n_clusters` to pick — the canonical clusterer after a
  UMAP reduction, and it finds non-convex clusters k-means cannot (two-moons at purity 1.0). Core
  points (≥ `min_samples` within `eps`) grow clusters by union-find, borders join, the rest is
  noise. `eps=None` auto-picks the median k-distance. Neighbours come from fastumap's own kNN
  (blocked exact, or the faiss approx backend above ~16k points), so it scales past a naive O(n²)
  DBSCAN. Exported at the top level.

## [0.2.9] - 2026-08-25

### Added
- **Project to any output dimension**, not just 2 or 3. `umap_project(x, dimensions=10)` now works.
  The SGD and spectral init were always dimension-agnostic; only one validation line forbade it.
  Use a mid dimension (~10) when the goal is to *cluster* the result — umap-learn's own advice is
  to reduce to ~10 dims, not 2, before clustering. `dimensions` must be in `[1, n_features]`.
- **`precomputed_knn`** on `umap_project` / `fit`: pass your own `(indices, distances)` (each
  `(n, k)`, column 0 the point itself, e.g. from a faiss index you already built) to skip the
  built-in neighbour search. Feeding the same kNN reproduces the built-in layout bit-for-bit. It
  cannot be combined with `pca_dim`.

## [0.2.8] - 2026-08-25

### Changed
- Documentation pass over everything a human reads: the README, rust/README, and every module and
  function docstring and comment across `src/fastumap/` and the Rust kernel. Dense run-on prose is
  broken into short sentences and scannable bullets, in simpler technical English. Every fact,
  number, and caveat is kept verbatim; no code changed. Also removed a stale "unbuilt scaffold"
  note from `lib.rs`.

## [0.2.7] - 2026-08-25

### Added
- musllinux wheels (Alpine / musl libc), x86_64 and aarch64, cross-built with zig alongside the
  manylinux wheels. `pip install fastumap` now works on `python:*-alpine` images with no compiler,
  instead of falling back to the sdist + a Rust toolchain. Best-effort in CI: a musl cross flake
  cannot block the manylinux publish.

## [0.2.6] - 2026-08-25

### Fixed
- No more `RuntimeWarning: overflow encountered in exp` from the smooth-kNN membership step
  (`graph.py`) on inputs with zero-distance neighbours. `np.where` evaluated `exp(-offset/mid)`
  on the discarded `offset <= 0` slots too, where the exponent is large positive; now the
  discarded slots carry 0.0 into a single `exp`, so it never overflows. Output is bit-identical
  (surfaced by the hypothesis property tests).

## [0.2.5] - 2026-08-25

### Added
- `choose_k(embeddings, *, k_min=2, k_max=10, ...)` — pick `n_clusters` automatically by the
  inertia elbow (kneedle knee of the inertia-vs-k curve). Runs k-means across the range and
  returns the sharpest bend; on more than `max_samples` rows it searches a deterministic
  subsample to bound cost. A heuristic — recovers the true k give or take one on clear cluster
  structure (verified on separated blobs at k=3/5/7). Exported at the top level.

## [0.2.4] - 2026-08-25

### Changed
- Docs content pass for the 0.2.x merge: removed the last "pure numpy + scipy" claims from the
  `fastumap` and `fastumap.clustering` module docstrings and the blog draft, and added a
  `fastumap-accel` → `fastumap` migration note to the README (the package is gone; its kernels
  are built in). No code change.

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
