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

`target_weight` ∈ [0, 1] (default 0.5) trades geometry against labels: 0.0 attenuates inter-class
edges the least, 1.0 severs them entirely. `-1` labels are treated as unlabelled (semi-supervised).
Default `y=None` is ordinary unsupervised UMAP, unchanged. This is a faithful port of umap-learn's
categorical `discrete_metric_simplicial_set_intersection`; the original UMAP paper only proposes
the simplicial-set-intersection idea (arXiv:1802.03426, §7 Future Work), so the reference
implementation is the source. See umap-learn's [supervised docs](https://umap-learn.readthedocs.io/en/latest/supervised.html).

## Clustering

`cluster` groups points directly — implemented ourselves (numpy and scipy plus the native k-means
kernel), with no external clustering library. It does **not** compute a 2-D layout first: that
layout is an iterative SGD and takes
minutes at a million points, and clustering doesn't need it.

```python
from fastumap import cluster

labels = cluster(embeddings, method="kmeans", n_clusters=20)     # k-means++ / Lloyd
labels = cluster(embeddings, method="spectral", n_clusters=20)   # non-convex clusters
```

Both take `n_clusters` and `metric` (`"euclidean"` / `"cosine"`) and are deterministic given
`random_state`. k-means is our own k-means++ seeding and Lloyd iterations with a **blocked**
assignment step, so the n×k distance matrix is never materialised — it holds to the low-memory
constraint at scale. `spectral` reuses fastumap's own normalised-eigenvector machinery, then
k-means those. For a picture, call `umap_project` separately (usually on a sample) — clustering
and visualisation are different jobs.

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

## Performance characteristics

At 1024 dimensions and n=5000, fastumap is roughly 2× slower per call than umap-learn
(~40 s versus ~20 s). This is inherent rather than a missing optimisation: the layout SGD
dominates runtime, and a vectorised NumPy SGD cannot match numba's compiled in-place optimiser.
fastumap is therefore the appropriate choice when import and cold-start cost dominate, and less
so when per-call latency on large, high-dimensional batches is the constraint. The optional
accelerator narrows this gap.

When a CPU quota is visible inside the container — `docker --cpus`, Kubernetes/EKS CPU limits,
EC2 cgroups — and the container still reports the host's full core count, an unconstrained BLAS
pool sized to those cores oversubscribes the quota and thrashes under concurrent load. fastumap
caps the pool to the quota automatically: measured under `docker --cpus=0.5` on a 16-core host
with four workers, per-call time drops from ~70 s to ~43 s (≈1.6×). No configuration is needed,
and nothing is capped where no quota is visible. (AWS Fargate is a known exception: it meters
CPU outside the container's cgroup, so the quota is invisible and the cap is a no-op there — but
Fargate's micro-VM also reports a low core count, so BLAS doesn't oversubscribe there anyway.)

## Server usage

`umap_project` is thread-safe — it holds no module-level mutable state and seeds a fresh RNG
per call — so it may be called from a worker thread (`await asyncio.to_thread(umap_project,
x, 2)`).

For long-running services, avoid recomputing the full layout on every request. Fit once and
place new points into the existing layout:

```python
from fastumap import fit, transform
from fastumap.projection import UMAPModel

model = fit(window, 2)                                 # cache it
xy, fit_distance = transform(model, pts, return_distances=True)   # coords + per-point fit

model.to_npz("layout.npz")                             # persist across restarts (versioned)
model = UMAPModel.from_npz("layout.npz")
```

`transform` approximates a full refit (roughly 72% of its local overlap) while keeping the
embedding stable across requests. `return_distances=True` returns each new point's distance to
its nearest training neighbour — a per-point measure of fit: points landing 2–3× further out
than the training set's own mean are extrapolations, and a rising batch mean is the signal to
refit. Persist a model with `to_npz` / `from_npz`, a versioned numpy format that survives
releases where a raw `pickle` would break on any dataclass change. To keep a *refreshed* view
comparable without the fit/transform split, pass the previous coordinates as
`umap_project(window, 2, init=previous)` so carried-over points start where they were (you place
any new rows). A cached 5000×1024 model occupies about 20 MB (training data stored as float32).
Rolling windows and sparse input are not supported — densify sparse input first, and refit when
the window slides.

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

For a given input and seed, output is bit-identical across processes and machines within one
environment. Two things change the layout across *different* environments: whether the native
kernel is present (the wheels carry it and it is used by default, but a run that falls back to the
numpy optimiser — an unbuilt source checkout — differs) and `fastumap[ann]` (an approximate
neighbour graph above 16k points). Treat both as part of the environment's dependency set;
`accelerator_active()` and `ann_available()` report which paths a run used, so a stored projection
can record how it was produced.

## Guarantees

Each is enforced by a test:

- **Nothing compiles at import** — no numba/llvmlite JIT. The base is NumPy, SciPy and the
  pure-Python `threadpoolctl`; the optional native kernels ship as precompiled wheels, so they
  import in milliseconds too.
- **Import under 200 ms**, deterministic (bit-identical) output, and thread-safe.
- **Bounded memory** — the full n×n distance matrix is never materialised (blocked kNN); under
  200 MB at 5000×1024.
- **2-D and 3-D** first-class; fully type-checked under pyright strict.

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
