# fastumap

<!-- --8<-- [start:pitch] -->
A UMAP and clustering library for environments where a JIT compiler at import time is not
acceptable: serverless functions, small or CPU-limited containers, and autoscaled workers that pay
a cold start on every new process.

umap-learn compiles its kernels with numba and LLVM the first time the package is imported. That
costs seconds on a workstation and minutes on a CPU-throttled container, and it recurs on every
cold start rather than once per machine. NumPy and SciPy are equally compiled code, but they ship
their machine code prebuilt in the wheel, so they load immediately. fastumap reimplements the UMAP
algorithm on top of them, with precompiled Rust kernels (also shipped in the wheel) for the parts
that need to be fast. Nothing is left to compile at import.

| | cold import | added to image |
|---|---:|---:|
| `import umap` (umap-learn) | ~21 s | ~172 MB |
| `import fastumap` | ~12 ms | 0 MB |

Projection quality stays within 0.01 neighbour overlap of umap-learn; the measurements are below.

<!-- --8<-- [end:pitch] -->
## Installation
<!-- --8<-- [start:install] -->

```bash
pip install fastumap                 # one wheel, native kernels included
pip install fastumap[ann]            # adds an approximate kNN backend for large inputs (see below)
```

<!-- --8<-- [end:install] -->
## Usage
<!-- --8<-- [start:usage] -->

```python
from fastumap import umap_project, spectral_project

xy  = umap_project(x, 2)                     # (n, 2)
xyz = umap_project(x, 3)                     # (n, 3)
red = umap_project(x, 10)                    # ~10-D for clustering (not just 2/3)
cos = umap_project(x, 2, metric="cosine")    # text / CLS embeddings
dm  = umap_project(x, 2, densmap=True)       # densMAP: keep dense/sparse regions distinct
```

- `metric="cosine"` is recommended for encoder embeddings. Euclidean distance on unnormalised
  vectors is dominated by magnitude rather than by the direction that carries meaning.
- `pca_dim=100` pre-reduces very wide inputs (for example 1024-dimensional embeddings) before the
  nearest-neighbour search. Neighbour overlap is preserved to within about 0.01. Off by default.

`umap_project` also accepts `n_neighbors`, `min_dist`, `spread`, `n_epochs`,
`negative_sample_rate`, `random_state`, and `chunk_count`.

<!-- --8<-- [end:usage] -->
## Supervised projection
<!-- --8<-- [start:supervised] -->

Categorical labels passed as `y` inform the layout: same-label points attract, so structure that
is separable in the input space stays separable in 2-D instead of interleaving.

```python
xy = umap_project(x, 2, y=labels)                     # labels: one int per point, -1 = unlabelled
xy = umap_project(x, 2, y=labels, target_weight=0.9)  # lean harder on the labels
```

`target_weight` is in [0, 1] and defaults to 0.5. It trades geometry against labels: 0.0 barely
weakens inter-class edges, 1.0 cuts them entirely. Labels of `-1` count as unlabelled
(semi-supervised). The default `y=None` is ordinary unsupervised UMAP, unchanged.

This ports umap-learn's categorical intersection. The UMAP paper only sketches the idea
(arXiv:1802.03426, §7 Future Work), so umap-learn is the reference; see its
[supervised docs](https://umap-learn.readthedocs.io/en/latest/supervised.html).

<!-- --8<-- [end:supervised] -->
## Clustering
<!-- --8<-- [start:clustering] -->

`cluster` groups points directly, implemented here in numpy and scipy plus the native k-means
kernel, with no external clustering library. It does not compute a 2-D layout first. That layout
is an iterative SGD and takes minutes at a million points, and clustering does not need it.

```python
from fastumap import cluster

labels = cluster(embeddings, method="kmeans", n_clusters=20)     # k-means++ / Lloyd
labels = cluster(embeddings, method="spectral", n_clusters=20)   # non-convex clusters
labels = cluster(embeddings, method="dbscan")                    # density: finds k, noise = -1
labels = cluster(embeddings, method="hdbscan")                   # density, no eps to pick
```

`hdbscan` is the hierarchical form of `dbscan`. It takes no `eps`, so clusters of different
densities are found together. The root of the hierarchy is never a cluster, so a dataset
containing a single group returns all `-1`.

`kmeans` and `spectral` take `n_clusters` and `metric` (`"euclidean"` or `"cosine"`) and are
deterministic given `random_state`. k-means uses k-means++ seeding and Lloyd iterations with a
blocked assignment step, so the n×k distance matrix is never materialised. `spectral` reuses
fastumap's normalised-eigenvector machinery and runs k-means on the result. For a visualisation,
call `umap_project` separately, usually on a sample; clustering and visualisation are separate
operations.

Further options:

- `return_centroids=True` returns a `ClusterResult(labels, centroids, inertia)` instead of labels
  alone, which is what is needed to compare runs or choose k.
- `choose_k(x, k_min=2, k_max=10)` selects `n_clusters` by the inertia elbow. It is a heuristic;
  it lands on the true k give or take one when the clusters are clear.
- `assign_clusters(x_new, centroids)` labels new points against already-fitted centroids, without
  re-clustering.

Above 50,000 points the native k-means kernel (matrixmultiply SIMD GEMM with rayon over row
blocks, compiled into the wheel) replaces the inner loop. It clusters 1,000,000 × 128 into 64
groups in about 10 s on a 16-core machine, measured on SIFT1M, against minutes for the numpy path.

Above 200,000 points it also switches to mini-batch k-means, which samples `batch_size` rows per
iteration rather than sweeping all of them. Measured on SIFT1M inside a `docker --cpus=2`
container: 5.2 s against full Lloyd's 25.3 s, at 1.010× the inertia.

```python
from fastumap import cluster, kmeans

labels = cluster(x, method="kmeans", n_clusters=64)   # mini-batch above 200k rows
labels, cent = kmeans(x, 64, batch_size=0)            # force full Lloyd at any size
labels, cent = kmeans(x, 64, batch_size=5000)         # force mini-batch, with an explicit batch
```

The switch is on `n` alone and never on the core count, so the same input and seed produce the
same labels on any machine. One limitation: on perfectly separated clusters, mini-batch can leave
two centroids inside one group, measured at 4.8× worse inertia on synthetic blobs 8σ apart. Its
running-mean update cannot move a centroid back across an empty gap, where full Lloyd's recompute
can. Real embeddings overlap and do not trigger this (MNIST measures 1.013× Lloyd), and
`batch_size=0` disables the switch.

<!-- --8<-- [end:clustering] -->
## Quality relative to umap-learn
<!-- --8<-- [start:quality] -->

fastumap stays within 0.01 neighbour overlap of umap-learn. Measured on MNIST (784-dimensional,
seed 42). These numbers are generated rather than transcribed: `make bench-report` regenerates them
into a dated report under
[docs/bench/](https://gitlab.com/jorgeecardona/fastumap/-/tree/main/docs/bench), which records the
machine, the load average and every package version they were measured on.

| n | method | wall time | overlap@15 | global corr |
|---|---|---:|---:|---:|
| 5000 | fastumap | 22 s | 0.341 | 0.320 |
| 5000 | umap-learn | 79 s | 0.344 | 0.329 |
| 5000 | pca | 87 s | 0.058 | 0.523 |
| 10000 | fastumap | 46 s | 0.271 | 0.361 |
| 10000 | umap-learn | 50 s | 0.268 | 0.387 |
| 10000 | pca | 120 s | 0.038 | 0.487 |
| 20000 | fastumap | 57 s | 0.207 | 0.358 |
| 20000 | umap-learn | 69 s | 0.204 | 0.368 |
| 20000 | pca | 237 s | 0.024 | 0.504 |

The quality columns are the comparable ones. overlap@15 and global corr are deterministic; the
wall times came from a loaded machine (recorded in the report) and are inflated for every method.
PCA is included as a control: it preserves global structure best and local structure worst, which
is the trade-off UMAP addresses.

umap-learn's times reuse the numba compilation from its first fit in the same process. A fresh
process pays roughly 20 s of compilation on every invocation, which fastumap does not pay. Above
about 50,000 points fastumap's exact O(n²) neighbour search becomes the bottleneck;
`pip install fastumap[ann]` addresses that, described below. Raising `chunk_count` (default 1)
recovers most of the remaining local-overlap gap at proportional cost, and remains deterministic.

> overlap@15 is the fraction of each point's 15 input-space neighbours retained after projection,
> read against a random baseline. global corr is the Spearman correlation of all pairwise
> distances, before against after.

<!-- --8<-- [end:quality] -->
## Speed per call
<!-- --8<-- [start:speed] -->

Import and cold-start cost are where fastumap differs. Per-call latency is close to umap-learn
rather than better than it.

- The native kernel ships by default, so per-call speed on large batches is close to umap-learn.
  It runs umap-learn's in-place SGD, in Rust.
- The numpy fallback, which only an unbuilt source checkout uses, is roughly 2× slower at 1024
  dimensions and n=5000 (about 40 s against 20 s). A vectorised numpy SGD cannot match numba's
  compiled loop.

fastumap is the appropriate choice when import and cold-start cost dominate. On per-call latency
for large, high-dimensional batches it is roughly even with umap-learn on the kernel path.

### BLAS thread cap under a CPU quota

Some containers cap CPU (`docker --cpus`, Kubernetes and EKS limits, EC2 cgroups) while still
reporting the host's full core count. BLAS then starts more threads than the quota supports and
thrashes under load.

fastumap caps the BLAS pool to the quota automatically, with no configuration. Measured under
`docker --cpus=0.5` on a 16-core host with 4 workers, per-call time drops from about 70 s to about
43 s (roughly 1.6×). Where no quota is visible, nothing is capped.

This is a no-op on AWS Fargate, which meters CPU outside the cgroup, leaving the quota invisible.
Fargate also reports a low core count, so BLAS does not oversubscribe there in the first place.

<!-- --8<-- [end:speed] -->
## Server usage
<!-- --8<-- [start:server] -->

`umap_project` is thread-safe. It holds no module-level mutable state and seeds a fresh RNG per
call, so it can be called from a worker thread (`await asyncio.to_thread(umap_project, x, 2)`).

Recomputing the whole layout on every request is unnecessary. Fit once, then place new points into
the existing layout:

```python
from fastumap import fit, transform
from fastumap.projection import UMAPModel

model = fit(window, 2)                                 # cache it
xy, fit_distance = transform(model, pts, return_distances=True)   # coords + per-point fit

model.to_npz("layout.npz")                             # persist across restarts (versioned)
model = UMAPModel.from_npz("layout.npz")
```

- `transform` places new points without a refit. It retains about 72% of a full fit's local
  overlap, and the layout stays stable across requests.
- `return_distances=True` returns each point's distance to its nearest training neighbour, which
  serves as a fit score. A point 2 to 3× further out than the training mean is an extrapolation.
  A rising batch mean indicates that a refit is due.
- `to_npz` and `from_npz` persist the model in a versioned numpy format. It survives releases,
  where a raw `pickle` would break on any dataclass change.
- `init=previous` is an alternative: `umap_project(window, 2, init=previous)` reuses the previous
  coordinates so carried-over points start where they were, and new rows are placed by the caller.
  It refreshes a view without the fit/transform split.

A cached 5000×1024 model is about 20 MB, with training data stored as float32. Rolling windows and
sparse input are not supported: densify sparse input first, and refit when the window slides.

<!-- --8<-- [end:server] -->
## The native accelerator
<!-- --8<-- [start:accelerator] -->

fastumap ships a compiled Rust extension, `fastumap._accel`, inside the wheel. It reimplements the
SGD layout kernel and the k-means kernel. The prebuilt wheels (abi3, Python 3.11+; Linux x86_64
and aarch64/Graviton, musllinux for Alpine, and Windows x86_64) carry it, so it is used
automatically. macOS has no prebuilt wheel yet and builds from the sdist, which requires a Rust
toolchain, as does any other platform without a prebuilt wheel.

The numpy path remains as a runtime fallback. If the extension is absent, for instance in an
unbuilt source checkout, fastumap still works, more slowly. The SGD kernel is roughly 1.7 to 1.9×
faster, with slightly higher overlap.

Because the kernel performs umap-learn's in-place walk rather than the fallback's per-epoch
approximation, it produces a different, higher-quality layout for the same seed. The active path
can be queried:

```python
import fastumap
fastumap.accelerator_active()   # True if the native kernel is present (the normal case)
```

Upgrading from `fastumap-accel`: that package no longer exists separately. Its kernels are built
into `fastumap` itself from 0.2.1 onward. Run `pip install -U fastumap` and drop any
`fastumap-accel` dependency; nothing else changes.

<!-- --8<-- [end:accelerator] -->
## Large inputs (approximate kNN)
<!-- --8<-- [start:ann] -->

The exact neighbour search is O(n²) and dominates runtime above roughly 20,000 points.
`pip install fastumap[ann]` adds an approximate backend, [faiss](https://faiss.ai) HNSW, which
ships prebuilt wheels for Linux x86_64 and aarch64, macOS and Windows, so it installs without a
compiler.

```python
xy = umap_project(x, 2, knn="auto")   # default: exact for small n, approximate above 16k
xy = umap_project(x, 2, knn="approx") # force approximate (needs fastumap[ann])
xy = umap_project(x, 2, knn="exact")  # force the exact brute force
```

`"auto"`, the default, switches to approximate only when the extra is installed and n is at least
16,384, so small inputs remain bit-identical. It is deterministic and retains at least 0.86
neighbour recall against exact. Measured on the neighbour search alone, at 256 dimensions:

| n | exact | approx | speedup | recall@15 |
|---|---:|---:|---:|---:|
| 20000 | 71 s | 29 s | 2.4× | 0.91 |
| 30000 | 142 s | 50 s | 2.8× | 0.86 |

`fastumap.ann_available()` reports whether the backend is installed.

<!-- --8<-- [start:reproducibility] -->
For a given input and seed, output is bit-identical across processes and machines in the same
environment. Two things change it across different environments:

- The native kernel against the numpy fallback. The wheels carry the kernel and use it by default;
  an unbuilt checkout falls back to numpy and differs.
- `fastumap[ann]`, which changes the neighbour graph above 16,384 points.

Both are properties of the environment. `accelerator_active()` and `ann_available()` report which
paths a run used, so a stored projection can record how it was produced.
<!-- --8<-- [end:reproducibility] -->

<!-- --8<-- [end:ann] -->
## Guarantees
<!-- --8<-- [start:guarantees] -->

Each of these is enforced by a test:

- Nothing compiles at import: no numba or llvmlite JIT. The base is NumPy, SciPy and the
  pure-Python `threadpoolctl`, and the native kernels ship precompiled in the wheel.
- Import completes in under 200 ms, output is deterministic (bit-identical), and the API is
  thread-safe.
- Memory is bounded: the full n×n distance matrix is never materialised (blocked kNN), staying
  under 200 MB at 5000×1024.
- Any output dimension is supported, 2-D and 3-D for visualisation and around 10-D for clustering.
  The codebase is fully type-checked under pyright strict.

<!-- --8<-- [end:guarantees] -->
## Development
<!-- --8<-- [start:development] -->

```bash
make check        # lint, type-check, and tests
make tox          # the suite across Python 3.11, 3.12, and 3.13
make bench-report # regenerate the quality + speed tables into a dated docs/bench/ report
make fargate      # import and fit timing under one CPU cap (docker --cpus=0.5 --memory=2g)
make constrained  # fit and cluster timing under 0.5, 1 and 2 vCPU, into a dated report
```

<!-- --8<-- [end:development] -->
## License and attribution
<!-- --8<-- [start:license] -->

fastumap is an independent reimplementation of the UMAP algorithm
([McInnes, Healy & Melville, arXiv:1802.03426](https://arxiv.org/abs/1802.03426)). It is not
affiliated with or endorsed by the UMAP authors, and it is not a drop-in replacement; the public
API is deliberately small. MIT licensed.

<!-- --8<-- [end:license] -->
