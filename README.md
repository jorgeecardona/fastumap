# fastumap

A UMAP implementation in pure **NumPy and SciPy**, built so that importing it costs
milliseconds rather than seconds.

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
pip install fastumap                 # NumPy and SciPy only
pip install fastumap[accel]          # adds the optional native accelerator (see below)
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

model = fit(window, 2)          # UMAPModel; picklable, cache it
xy    = transform(model, pts)   # inexpensive; the layout stays fixed across requests
```

`transform` approximates a full refit, retaining roughly 72% of the local overlap that
refitting would produce while keeping the embedding stable across requests. Refit when the
input distribution shifts; a practical trigger is the distance from new points to their nearest
training neighbour rising to 2–3× the training mean. A cached 5000×1024 model occupies about
20 MB (training data stored as float32). Rolling windows and sparse input are not supported —
densify sparse input first, and refit when the window slides.

## The optional accelerator

`fastumap-accel` is a small Rust reimplementation of the SGD, distributed as a separate abi3
wheel (Python 3.11+; x86_64 and aarch64/Graviton, with an sdist for other platforms). When it
is installed, fastumap uses it automatically; the base package remains pure NumPy and SciPy and
serves as the fallback. It is roughly 1.7–1.9× faster with slightly higher overlap.

Because it performs umap-learn's true in-place walk rather than the fallback's per-epoch
approximation, it produces a **different — higher-quality — layout for the same seed.** Query
the active path with:

```python
import fastumap
fastumap.accelerator_active()   # True if the native kernel is installed and will be used
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
environment. Two optional packages change the layout, so reproducibility *across* environments
requires pinning them: `fastumap-accel` (a different optimiser) and `fastumap[ann]` (an
approximate neighbour graph above 16k points). Treat them as part of the environment's
dependency set; `accelerator_active()` and `ann_available()` report which paths a run used, so a
stored projection can record how it was produced.

## Guarantees

Each is enforced by a test:

- **No JIT or compiled stack at runtime** — NumPy, SciPy, and the small pure-Python
  `threadpoolctl`; never numba, llvmlite, scikit-learn, or first-party compiled code.
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
