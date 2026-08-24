# fastumap

UMAP in pure **numpy + scipy**, built so that importing it costs almost nothing.

| | cold import | added to image |
|---|---:|---:|
| `import umap` (umap-learn) | **~21 s** (laptop) · **148 s** (0.5 vCPU ×4) | ~172 MB |
| `import fastumap` | **~12 ms** | 0 MB |

That gap is the reason this library exists. umap-learn's kernels are numba, and numba compiles
them with LLVM the first time you import it — seconds on a laptop, minutes on a small
CPU-throttled container. Here's the catch: numpy and scipy are just as compiled underneath,
they just ship **precompiled** in the wheel, so they load in milliseconds. So the fix isn't to
drop compiled code — it's to stop compiling *at import time*. fastumap keeps the UMAP math and
leaves nothing to JIT when it loads.

## Why import cost matters

It bites hardest on a small, CPU-throttled container — a fraction of a vCPU, a few workers, a
health check that polls from the first second. Every worker pays the import, so the compile
stacks up fast:

- `import umap` ≈ 4 CPU-seconds, and `pynndescent` alone JIT-compiles 46 `@njit` functions.
- Through a **0.5 vCPU, 4-worker** cgroup that becomes **~148 s** — past a ~120 s health
  check, so the container is killed before it serves one request.
- More cores barely help (37.8 s at 1 CPU, 16.1 s at 2), because the compile is serial.
- A baked numba cache doesn't save you (eager signatures skip it), and `NUMBA_DISABLE_JIT=1`
  runs ~100× slower.

fastumap sidesteps all of it — it imports in **~12 ms**, on every worker.

## Quality vs umap-learn

Within **0.01–0.02** neighbour-overlap of umap-learn, ahead on global structure, and it buries
PCA and the random control. Real MNIST (784-dim, single thread, `make bench`):

| n | method | wall | overlap@15 ↑ | global corr ↑ |
|---|---|---:|---:|---:|
| 5000 | **fastumap** | 26 s | 0.333 | 0.310 |
| 5000 | umap-learn | 73 s | 0.344 | 0.329 |
| 10000 | **fastumap** | 72 s | 0.267 | 0.279 |
| 10000 | umap-learn | 83 s | 0.274 | 0.286 |
| 20000 | fastumap | 110 s | 0.188 | 0.323 |
| 20000 | umap-learn | **31 s** | 0.205 | 0.316 |

- **Wins at 5k/10k, loses at 20k** — the brute-force O(n²) kNN catches up with it (approximate
  kNN is roadmap #15). PCA sits at ~0.02–0.06 overlap: a linear method can't compete locally.
- umap-learn's times here **reuse the numba compile** paid on the first fit; a fresh process
  pays ~20 s every time — the cost fastumap exists to avoid.
- **`chunk_count=10`** closes most of the local-overlap gap (~5× slower, still deterministic).
  Default `1` is the fast path.

> overlap@k = share of each point's k input neighbours still neighbours after projection
> (read it against random). global corr = Spearman corr of all pairwise distances, before vs after.

## Install

```bash
pip install fastumap        # numpy + scipy, nothing else
```

## Use

```python
from fastumap import umap_project, spectral_project

xy  = umap_project(x, 2)                     # (n, 2)
xyz = umap_project(x, 3)                     # (n, 3)
cos = umap_project(x, 2, metric="cosine")    # text / CLS embeddings
```

- **cosine** for encoder embeddings — euclidean on unnormalised vectors is dominated by
  length, not the direction that carries meaning.
- **`pca_dim=100`** for wide inputs (e.g. 1024-dim): pre-reduces before the kNN, overlap holds
  within ~0.01, off by default.

Same input + seed → **bit-identical** output across processes. `umap_project` also takes
`n_neighbors`, `min_dist`, `spread`, `n_epochs`, `negative_sample_rate`, `random_state`.

## In a server

Thread-safe — fresh RNG per call, no module-level state, so
`await asyncio.to_thread(umap_project, x, 2)` is fine.

Don't refit every request. Fit once, place new points:

```python
model = fit(window, 2)          # cache it — UMAPModel pickles
xy    = transform(model, pts)   # per request: cheap, stable across requests
```

- **transform ≈ refit?** Keeps ~**72%** of a full refit's local overlap. Good for placing
  in-distribution points against a fixed window; refit when the window itself shifts.
- **Gone stale?** Watch new-point distance to nearest training neighbour. Drifts to 2–3× the
  training mean → time to refit. (A timer is a weak proxy.)
- **Memory:** ~**20 MB** per cached 5000×1024 model (`train` kept as float32).
- **Rolling window** ("add these, drop old ones") and **sparse input**: not supported — both
  would need a rebuild. Densify sparse first; `fit`/`transform` cover the append-only case.

## Speed, honestly

At 1024-dim, n=5000, fastumap is ~**2× slower per call** than umap-learn (~38–53 s vs ~22 s).
That's inherent — a vectorised numpy SGD can't match numba's in-place walk. Use fastumap when
**cold-start / import cost dominates**; reach for umap-learn when **per-call latency on big
high-dimensional batches** does.

**Optional Rust accelerator** ([`rust/`](rust/)) — ~**1.7–1.9× faster with better overlap**,
one abi3 wheel for Python 3.11+. Auto-detected when installed; the base stays pure numpy+scipy
and is the fallback.

## Guarantees (each pinned by a test)

- **numpy + scipy only** at runtime — no numba, llvmlite, sklearn, or compiled code of ours.
- **Import < 200 ms** · **deterministic** (bit-identical) · **thread-safe**.
- **Memory-bounded** — the n×n distance matrix is never built (blocked kNN); < 200 MB at 5000×1024.
- **2-D and 3-D**, both first-class · **typed**, pyright strict.

## Testing

```bash
make check     # lint + typecheck + tests (mirrors CI's gate)
make tox       # across Python 3.11 / 3.12 / 3.13
make fargate   # import + fit timing under docker --cpus=0.5 --memory=2g
```

`--cpus` is a CFS quota (throttles total CPU-time like a container cgroup), so the numbers
mean what they will on 0.5 vCPU.

## Not affiliated with UMAP

An **independent reimplementation** of the UMAP algorithm
([McInnes, Healy, Melville, arXiv:1802.03426](https://arxiv.org/abs/1802.03426)) — not endorsed
by the authors, and **not a drop-in replacement**. MIT licensed.
