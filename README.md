# fastumap

A numba-free UMAP in pure **numpy + scipy**. It exists for one reason: import cost.

| | import (cold) | wheels added to image |
|---|---:|---:|
| `import umap` (umap-learn) | **~21 s** on a laptop; **148 s** at 0.5 vCPU ×4 | ~172 MB (llvmlite alone 113 MB) |
| `import fastumap` | **~12 ms** | 0 MB (numpy + scipy already present) |

umap-learn is excellent, but its kernels are numba, and numba compiles with LLVM at
*import* time. numpy and scipy ship precompiled kernels, so they import in milliseconds.
That is the whole thesis: **the problem is not numba, it is compilation at import time**,
and a vectorised numpy implementation avoids it without giving up much quality.

## Why this exists — the incident

The first consumer is an internal ML-platform API on AWS Fargate: **0.5 vCPU, 4 granian
workers, 2 GB**, shared with a Datadog sidecar. Every worker pays every import, and the
cgroup meters CPU, so four numba compilations run through a half-vCPU straw at once:

- `from umap import UMAP` cost ~4.1 CPU-seconds; `pynndescent` alone declares 46
  eagerly-compiled `@njit` functions.
- In that container the import took **148 s wall** (126 s and 206 s on two Fargate tasks).
  The health check kills the container at ~120 s, so the service crash-looped and **never
  served a request**. At 1 CPU it was 37.8 s, at 2 CPUs 16.1 s — compilation is serial, so
  more cores barely help.
- Baking a numba cache into the image did not fix it (11.9 s vs 12.0 s): the eager-signature
  functions that dominate import have no `cache=True`.
- `NUMBA_DISABLE_JIT=1` works but the pure-Python kernels are ~100× slower.

fastumap imports in ~12 ms, four times over, and leaves the health-check budget intact.

## Quality vs umap-learn

Graded on separated Gaussian clusters (dimensions noted per table), against umap-learn on
the same data. We assert we are *within a tolerance* of the reference, never that layouts
match: different initialisation and update order rotate and reflect equally-good embeddings.

<!-- QUALITY_TABLE -->
**n=1000, 64-dim, 8 clusters, 3 seeds** (`make grade`):

| layout | overlap@15 ↑ | global dist corr ↑ |
|---|---:|---:|
| fastumap | 0.236 ± 0.003 | **0.400 ± 0.086** |
| umap-learn | **0.250 ± 0.003** | 0.206 ± 0.033 |
| pca | 0.165 ± 0.001 | 0.843 ± 0.012 |
| random | 0.015 ± 0.001 | −0.003 ± 0.003 |

fastumap (default) trails umap-learn by ~0.014 on local neighbourhood overlap and is well
ahead on global structure. Both beat PCA on local overlap and crush the random control.

**Quality knob — `chunk_count`.** The optimiser defaults to a per-epoch snapshot update
(`chunk_count=1`), the fast path the 0.5-vCPU service needs. Raising `chunk_count` processes
each epoch's edges in that many shuffled chunks so later chunks see earlier moves (partial
in-epoch feedback, toward umap-learn's in-place walk). It trades speed for local overlap:

| chunk_count | overlap@15 (n=1000) | vs default | rel. cost |
|---:|---:|---:|---:|
| 1 (default) | 0.234 | — | 1× |
| 10 | 0.244 | +0.010 | ~5× |
| 40 | 0.248 | +0.014 (≈ umap-learn) | ~10× |

`umap_project(x, 2, chunk_count=10)` closes most of the gap; it stays deterministic.

### Real data — MNIST (`make bench`)

Single-thread, 784-dim MNIST via `bench/mnist.py` (defaults, `chunk_count=1`):

| n | method | wall-clock | overlap@15 ↑ | global dist corr ↑ |
|---|---|---:|---:|---:|
| 5000 | **fastumap** | 26 s | 0.333 | 0.310 |
| 5000 | umap-learn | 73 s | 0.344 | 0.329 |
| 5000 | pca | 2 s | 0.058 | 0.523 |
| 10000 | **fastumap** | 72 s | 0.267 | 0.279 |
| 10000 | umap-learn | 83 s | 0.274 | 0.286 |
| 10000 | pca | 4 s | 0.039 | 0.471 |
| 20000 | fastumap | 110 s | 0.188 | 0.323 |
| 20000 | umap-learn | 31 s | 0.205 | 0.316 |
| 20000 | pca | 10 s | 0.024 | 0.497 |

fastumap stays within **0.01–0.02** overlap of umap-learn at every size, and beats PCA on
local structure by a wide margin (PCA keeps global distances best, as a linear method does).
On speed it wins at 5k and 10k. Two honest caveats: (1) wall-clock is measured in one
process, so umap-learn's later fits reuse the numba compile paid on the first — its 20k time
excludes the ~20 s compile a *fresh* process pays every time, which is the cost fastumap
exists to avoid; (2) at 20k fastumap's brute-force O(n²) kNN and Python-loop SGD lose to
pynndescent — the case the approximate-kNN backend (roadmap #15) addresses.

Metrics:
- **overlap@k** — share of each point's k input-space neighbours still among its k nearest
  after projection. Chance is ~`k/(n-1)`; always read against the random control.
- **global dist corr** — Spearman correlation of all pairwise distances, before vs after.

## Install

```bash
pip install fastumap        # numpy + scipy, nothing else
```

## Use

```python
from fastumap import umap_project, spectral_project

xy  = umap_project(embeddings, dimensions=2)                    # (n, 2)
xyz = umap_project(embeddings, dimensions=3)                    # (n, 3)
cos = umap_project(embeddings, dimensions=2, metric="cosine")  # for text/CLS embeddings
init = spectral_project(embeddings, dimensions=2)              # just the spectral init
```

`metric` is `"euclidean"` (default) or `"cosine"`. Cosine is usually what you want for
encoder/CLS embeddings — euclidean on unnormalised output is dominated by vector length,
not the direction that carries the meaning.

**Incremental placement.** Fit once, then place new points into the frozen layout without
refitting — so a drift view stays stable instead of reshuffling every request:

```python
from fastumap import fit, transform

model = fit(window, dimensions=2)      # UMAPModel; model.embedding is the layout
new_xy = transform(model, new_points)  # (n_new, 2), placed against the fixed layout
```

Same input and seed give **bit-identical** output across processes (the layout must be
stable across reloads so people can compare the picture over time). `umap_project` takes
`n_neighbors`, `min_dist`, `spread`, `n_epochs`, `negative_sample_rate`, `random_state`,
`metric`, and `chunk_count`.

## Using this in a server

fastumap is safe to call from a worker thread of an async server (`asyncio.to_thread`):
arguments in, array out, a fresh seeded RNG per call, no module-level mutable state. Two
projections running concurrently in different threads each produce their single-threaded
result — pinned by a test — so `await asyncio.to_thread(umap_project, matrix, 2)` is fine.

For a long-lived server, don't recompute the whole layout every request. Fit once per
window, cache the model, and place newly-arrived points with `transform`:

```python
model = fit(window, dimensions=2)     # once per window; cache the UMAPModel
xy = transform(model, new_points)     # per request — cheap, and stable across requests
```

**Is a transform comparable to a refit?** A `transform` layout is an approximation of a
full refit of the same window: new points are placed against the frozen training embedding,
so the picture stays put across requests (a full refit would rotate/reflect the whole
thing). It preserves each new point's neighbourhood well when the new points resemble the
training distribution; it does not re-discover global structure. Refit when the window
itself shifts.

**When does a cached model go stale?** Watch the distance from new points to their nearest
training neighbour. When the batch mean of that distance drifts well above the training
set's own mean nearest-neighbour distance (say 2–3×), or a rising share of new points have
every neighbour far from any training point, the transform is extrapolating and it is time
to refit. A timer is a weak proxy; this distance signal reacts to the actual drift, and both
quantities are cheap to compute from the kNN you already run.

**Persisting a model.** `UMAPModel` is a plain frozen dataclass of numpy arrays, so it
pickles: fit once, `pickle.dumps(model)`, and every worker — or a worker after restart —
can `pickle.loads` it instead of refitting. Picklability is a supported contract (a test
pins it). The footprint is dominated by `train`, which the model keeps for the transform
kNN: roughly `n_train × n_features × 8` bytes, about **40 MB** for a 5000 × 1024 model,
times the number of cached copies. (Storing `train` as float32 would halve that; tracked.)

**Rolling window — out of scope.** "Add these points, drop those older than X" is not a
supported operation: dropping training points would require rebuilding the graph, i.e. a
refit. Use `fit`/`transform` for the append-only case (new points against a fixed window),
and refit when the window slides.

## Sparse input

Not supported — `umap_project`/`fit` take a **dense** numpy array. The blocked kNN relies
on dense BLAS matmul, and the target workload is dense encoder/CLS embeddings, so a native
sparse distance path is out of scope. Passing a `scipy.sparse` matrix raises a `TypeError`
telling you to densify first (`matrix.toarray()`). Revisit if a sparse workload actually
shows up.

## Guarantees (enforced by tests)

- **numpy + scipy only** at runtime — no numba, llvmlite, scikit-learn, or compiled
  extension of our own. A test asserts the JIT stack is never imported.
- **Import under 200 ms** — a subprocess test measures it.
- **Deterministic** — pinned ARPACK start vector, seeded negative sampler; identical bytes
  across processes.
- **Thread-safe** — no module-level mutable state, a fresh RNG per call; concurrent
  projections in different threads each match their single-threaded result (a test pins it),
  so it is safe to call from `asyncio.to_thread`.
- **Memory bounded** — the n-by-n distance matrix is never materialised (512-row blocked
  kNN). At the worst case (5000 × 1024) the fit adds ~102 MB over the import+input baseline
  (175 MB total peak); the test enforces a 200 MB ceiling.
- **Time budgeted** — single-thread wall-clock ~5.3 s at 1000 × 1024, ~32 s at 5000 × 1024;
  a CI test measures under a single-core cap and fails on a regression past a generous
  ceiling (25 s / 120 s). The SGD dominates — the number is honest, not yet fast; speeding
  it up further is tracked.
- **2-D and 3-D**, both first-class. **Typed**, pyright strict.

## Releasing

CI (GitLab, moon + uv + proto) runs lint, format, typecheck, tests, build on every push.
On `main` it runs `:release`: the same checks, then publishes to PyPI via **OIDC Trusted
Publishing** (no stored token, pending publisher configured) whenever the version in
`pyproject.toml` is not yet on PyPI — idempotent, so it is a no-op on every other pipeline.

## Not affiliated with UMAP

fastumap is an **independent reimplementation** of the UMAP algorithm
([McInnes, Healy, Melville, arXiv:1802.03426](https://arxiv.org/abs/1802.03426);
reference implementation [lmcinnes/umap](https://github.com/lmcinnes/umap)). It is not
affiliated with or endorsed by the UMAP authors, and it is **not a drop-in replacement** —
the public surface is deliberately small. MIT licensed.
