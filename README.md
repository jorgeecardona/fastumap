# fastumap

A UMAP and clustering library built for one thing: **importing it costs milliseconds, not
seconds** — nothing compiles when you import it. Mostly NumPy and SciPy, with precompiled native
kernels (Rust, built into the wheel) for speed at scale.

| | cold import | added to image |
|---|---:|---:|
| `import umap` (umap-learn) | ~21 s | ~172 MB |
| `import fastumap` | ~12 ms | 0 MB |

umap-learn's kernels are compiled by numba with LLVM the first time the package is imported —
seconds on a laptop, minutes on a small CPU-throttled container. NumPy and SciPy are equally
compiled, but they ship their machine code precompiled in the wheel, so they load immediately.
fastumap reimplements the UMAP algorithm on top of them: the same mathematics, with nothing
left to compile at import time.

## Installation

```bash
pip install fastumap                 # one wheel — native kernels included
pip install fastumap[ann]            # adds an approximate kNN backend for large inputs (see below)
```

## Usage

```python
from fastumap import umap_project, spectral_project

xy  = umap_project(x, 2)                     # (n, 2)
xyz = umap_project(x, 3)                     # (n, 3)
cos = umap_project(x, 2, metric="cosine")    # text / CLS embeddings
```

- `metric="cosine"` is recommended for encoder embeddings; Euclidean distance on unnormalised
  vectors is dominated by magnitude rather than the direction that carries meaning.
- `pca_dim=100` pre-reduces very wide inputs (e.g. 1024-dimensional embeddings) before the
  nearest-neighbour search. Neighbour overlap is preserved to within ~0.01. Off by default.

`umap_project` also accepts `n_neighbors`, `min_dist`, `spread`, `n_epochs`,
`negative_sample_rate`, `random_state`, and `chunk_count`.

## Supervised projection

Pass categorical labels as `y` to let the classes inform the layout — same-label points attract,
so structure that is separable in the input space stays separable in 2-D instead of interleaving:

```python
xy = umap_project(x, 2, y=labels)                     # labels: one int per point, -1 = unlabelled
xy = umap_project(x, 2, y=labels, target_weight=0.9)  # lean harder on the labels
```

`target_weight` ∈ [0, 1] (default 0.5) trades geometry against labels: 0.0 barely weakens
inter-class edges, 1.0 cuts them entirely. `-1` labels count as unlabelled (semi-supervised), and
the default `y=None` is ordinary unsupervised UMAP, unchanged.

This ports umap-learn's categorical intersection. The UMAP paper only sketches the idea
(arXiv:1802.03426, §7 Future Work), so umap-learn is the reference — see its
[supervised docs](https://umap-learn.readthedocs.io/en/latest/supervised.html).

## Clustering

`cluster` groups points directly — implemented ourselves (numpy and scipy plus the native k-means
kernel), with no external clustering library. It does **not** compute a 2-D layout first: that
layout is an iterative SGD and takes
minutes at a million points, and clustering doesn't need it.

```python
from fastumap import cluster

labels = cluster(embeddings, method="kmeans", n_clusters=20)     # k-means++ / Lloyd
labels = cluster(embeddings, method="spectral", n_clusters=20)   # non-convex clusters
labels = cluster(embeddings, method="dbscan")                    # density: finds k, noise = -1
```

Both take `n_clusters` and `metric` (`"euclidean"` / `"cosine"`) and are deterministic given
`random_state`. k-means is our own k-means++ seeding and Lloyd iterations with a **blocked**
assignment step, so the n×k distance matrix is never materialised — it holds to the low-memory
constraint at scale. `spectral` reuses fastumap's own normalised-eigenvector machinery, then
k-means those. For a picture, call `umap_project` separately (usually on a sample) — clustering
and visualisation are different jobs.

More knobs:

- **`return_centroids=True`** returns a `ClusterResult(labels, centroids, inertia)` instead of just
  labels — the numbers you need to compare runs or pick k.
- **`choose_k(x, k_min=2, k_max=10)`** picks `n_clusters` for you, by the inertia elbow. A
  heuristic: it lands on the true k give or take one when the clusters are clear.
- **`assign_clusters(x_new, centroids)`** labels new points against already-fitted centroids — the
  predict half, no re-clustering.

For large inputs fastumap's native k-means kernel — compiled into the wheel (matrixmultiply SIMD
GEMM + rayon over row-blocks) — replaces the inner loop: it clusters **1,000,000 × 128 into 64
groups in ~10 s** on a 16-core machine (measured on SIFT1M), versus minutes for the numpy path,
and engages automatically above 50k points. That is a **multi-core batch**
figure — on a throttled container (0.5–2 vCPU) the same job is CPU-bound and takes far longer, so
run clustering on a real machine, not a fractional-vCPU serving box.

## Quality relative to umap-learn

fastumap stays within 0.01–0.02 neighbour overlap of umap-learn and is marginally ahead on
global structure. Measured on MNIST (784-dimensional, single-threaded, `make bench`):

| n | method | wall time | overlap@15 | global corr |
|---|---|---:|---:|---:|
| 5000 | fastumap | 26 s | 0.333 | 0.310 |
| 5000 | umap-learn | 73 s | 0.344 | 0.329 |
| 10000 | fastumap | 72 s | 0.267 | 0.279 |
| 10000 | umap-learn | 83 s | 0.274 | 0.286 |
| 20000 | fastumap | 110 s | 0.188 | 0.323 |
| 20000 | umap-learn | 31 s | 0.205 | 0.316 |

fastumap is faster at 5k and 10k and slower at 20k, where its exact O(n²) neighbour search
becomes the bottleneck (`pip install fastumap[ann]` addresses this — see below). umap-learn's times
reuse the numba compilation from its first fit — a fresh process pays roughly 20 s of
compilation on every invocation. Raising `chunk_count` (default 1) recovers most of the
remaining local-overlap gap at proportional cost and stays deterministic.

> overlap@15 is the fraction of each point's 15 input-space neighbours retained after
> projection (read against a random baseline). global corr is the Spearman correlation of all
> pairwise distances, before versus after.

## Speed per call

Cold start is fastumap's win. Per-call latency is the trade-off.

- The **native kernel ships by default**, so per-call speed is close to umap-learn on large
  batches (it runs umap-learn's true in-place SGD, in Rust).
- The **numpy fallback** (only an unbuilt source checkout uses it) is the slow one: ~2× slower at
  1024 dims, n=5000 (~40 s vs ~20 s). A vectorised numpy SGD can't match numba's compiled loop.

So: pick fastumap when import and cold-start cost dominate. On raw per-call latency for big,
high-dimensional batches, it's roughly even with umap-learn on the kernel path.

### BLAS thread cap under a CPU quota

Some containers cap CPU (`docker --cpus`, Kubernetes/EKS limits, EC2 cgroups) but still report the
host's full core count. BLAS then starts too many threads for the quota and thrashes under load.

fastumap caps the BLAS pool to the quota automatically — no configuration. Measured under
`docker --cpus=0.5`, 16-core host, 4 workers: per-call time drops ~70 s → ~43 s (≈1.6×). Where no
quota is visible, nothing is capped.

**AWS Fargate is a no-op here**: it meters CPU outside the cgroup, so the quota is invisible. But
Fargate also reports a low core count, so BLAS doesn't oversubscribe there anyway.

## Server usage

`umap_project` is thread-safe — it holds no module-level mutable state and seeds a fresh RNG
per call — so it may be called from a worker thread (`await asyncio.to_thread(umap_project,
x, 2)`).

Don't recompute the whole layout on every request. Fit once, then place new points into it:

```python
from fastumap import fit, transform
from fastumap.projection import UMAPModel

model = fit(window, 2)                                 # cache it
xy, fit_distance = transform(model, pts, return_distances=True)   # coords + per-point fit

model.to_npz("layout.npz")                             # persist across restarts (versioned)
model = UMAPModel.from_npz("layout.npz")
```

- **`transform`** places new points without a refit. It keeps ~72% of a full fit's local overlap,
  and the layout stays stable across requests.
- **`return_distances=True`** gives each point's distance to its nearest training neighbour — a
  fit score. A point 2–3× further out than the training mean is an extrapolation. A rising batch
  mean means: time to refit.
- **`to_npz` / `from_npz`** persists the model in a versioned numpy format. It survives releases;
  a raw `pickle` would break on any dataclass change.
- **`init=previous`** is the other option: `umap_project(window, 2, init=previous)` reuses the old
  coordinates so carried-over points start where they were (you place any new rows). Use it to
  refresh a view without the fit/transform split.

A cached 5000×1024 model is about 20 MB (training data stored as float32). Not supported: rolling
windows and sparse input — densify sparse input first, and refit when the window slides.

## The native accelerator

fastumap ships a compiled Rust extension (`fastumap._accel`) **inside the wheel** — a
reimplementation of the SGD layout kernel and the k-means kernel. The prebuilt wheels (abi3,
Python 3.11+; Linux x86_64 and aarch64/Graviton) carry it, so it is used automatically. On a
platform without a prebuilt wheel, pip builds from the sdist, which needs a Rust toolchain. The
NumPy path remains as a runtime fallback — if the extension is ever not present (for instance an
unbuilt source checkout), fastumap still works, just slower. The SGD kernel is roughly 1.7–1.9×
faster with slightly higher overlap.

Because it performs umap-learn's true in-place walk rather than the fallback's per-epoch
approximation, it produces a **different — higher-quality — layout for the same seed.** Query
the active path with:

```python
import fastumap
fastumap.accelerator_active()   # True if the native kernel is present (the normal case)
```

**Upgrading from `fastumap-accel`?** It no longer exists as a separate package — its kernels are
built into `fastumap` itself (0.2.1+). Just `pip install -U fastumap` and drop any
`fastumap-accel` dependency; nothing else changes.

## Large inputs (approximate kNN)

The exact neighbour search is O(n²) and dominates runtime above ~20k points. `pip install
fastumap[ann]` adds an approximate backend — [faiss](https://faiss.ai) HNSW, which ships
prebuilt wheels (Linux x86_64/aarch64, macOS, Windows), so it installs without a compiler.

```python
xy = umap_project(x, 2, knn="auto")   # default: exact for small n, approximate above 16k
xy = umap_project(x, 2, knn="approx") # force approximate (needs fastumap[ann])
xy = umap_project(x, 2, knn="exact")  # force the exact brute force
```

`"auto"` (the default) only switches to approximate when the extra is installed *and* n ≥
16384, so small inputs stay bit-identical. It's deterministic and keeps ≥0.86 neighbour recall
against exact. Measured on the neighbour search alone (256-dim):

| n | exact | approx | speedup | recall@15 |
|---|---:|---:|---:|---:|
| 20000 | 71 s | 29 s | **2.4×** | 0.91 |
| 30000 | 142 s | 50 s | **2.8×** | 0.86 |

`fastumap.ann_available()` reports whether the backend is installed.

For a given input and seed, output is bit-identical across processes and machines in the same
environment. Two things change it across *different* environments:

- **The native kernel vs the numpy fallback** — the wheels carry the kernel and use it by default;
  an unbuilt checkout falls back to numpy and differs.
- **`fastumap[ann]`** — the approximate neighbour graph above 16k points.

Treat both as part of the environment. `accelerator_active()` and `ann_available()` report which
paths a run used, so a stored projection can record how it was made.

## Guarantees

Each is enforced by a test:

- **Nothing compiles at import** — no numba/llvmlite JIT. The base is NumPy, SciPy and the
  pure-Python `threadpoolctl`; the native kernels ship precompiled in the wheel, so they import in
  milliseconds too.
- **Import under 200 ms**, deterministic (bit-identical) output, and thread-safe.
- **Bounded memory** — the full n×n distance matrix is never materialised (blocked kNN); under
  200 MB at 5000×1024.
- **Any output dimension** — 2-D/3-D for pictures, ~10-D for clustering; fully type-checked under
  pyright strict.

## Development

```bash
make check     # lint, type-check, and tests
make tox       # the suite across Python 3.11, 3.12, and 3.13
make fargate   # import and fit timing under a constrained CPU cap (docker --cpus=0.5 --memory=2g)
```

## License and attribution

fastumap is an independent reimplementation of the UMAP algorithm
([McInnes, Healy & Melville, arXiv:1802.03426](https://arxiv.org/abs/1802.03426)). It is not
affiliated with or endorsed by the UMAP authors and is not a drop-in replacement; the public
API is intentionally small. MIT licensed.
