# fastumap

I kept needing to drop a pile of embeddings into 2-D for a quick plot, and every time I
reached for umap-learn the *import* alone took about twenty seconds — almost all of it numba
compiling its kernels with LLVM. So I wrote the same UMAP in plain numpy and scipy. It imports
in a few milliseconds.

| | cold import | added to your image |
|---|---:|---:|
| `import umap` (umap-learn) | **~21 s** | ~172 MB |
| `import fastumap` | **~12 ms** | 0 MB |

numpy and scipy are every bit as compiled as numba — they just ship the machine code
*precompiled* in the wheel, so they load instantly. numba compiles on your machine, the first
time you import it, and on a small or CPU-throttled box that stretches from seconds into
minutes. That's the only thing fastumap really changes: same UMAP math, nothing left to
compile when it loads.

## Using it

```bash
pip install fastumap        # just numpy + scipy
```

```python
from fastumap import umap_project, spectral_project

xy  = umap_project(x, 2)                     # (n, 2)
xyz = umap_project(x, 3)                     # (n, 3)
cos = umap_project(x, 2, metric="cosine")    # for text / CLS embeddings
```

- Use **cosine** for encoder embeddings — plain euclidean is dominated by vector length, not
  the direction that carries the meaning.
- For very wide inputs (say 1024-dim), **`pca_dim=100`** pre-reduces before the nearest-
  neighbour search: overlap barely moves and it's cheaper. Off by default.

The same input and seed give **bit-identical** output every time, across processes — so a
plot you regenerate tomorrow looks the same as today's.

## How close is it to umap-learn?

Close on local neighbourhoods, a little ahead on global structure. Here's what I get on MNIST
(784-dim, one thread, `make bench`):

| n | | wall | overlap@15 | global corr |
|---|---|---:|---:|---:|
| 5000 | **fastumap** | 26 s | 0.333 | 0.310 |
| 5000 | umap-learn | 73 s | 0.344 | 0.329 |
| 10000 | **fastumap** | 72 s | 0.267 | 0.279 |
| 10000 | umap-learn | 83 s | 0.274 | 0.286 |
| 20000 | fastumap | 110 s | 0.188 | 0.323 |
| 20000 | umap-learn | **31 s** | 0.205 | 0.316 |

A few honest notes:

- It wins on wall-clock at 5k and 10k and loses at 20k, where the brute-force neighbour search
  starts to hurt (a faster approximate one is on my list). PCA trails far behind on local
  structure — a linear method can't fold the space the way UMAP does.
- umap-learn's times above reuse the numba compile from its first fit; a *fresh* process pays
  that ~20 s every single time, which is the cost I was trying to get rid of.
- If you want the last bit of local overlap back, `chunk_count=10` recovers most of it (slower,
  and still deterministic).

> overlap@15 = how many of each point's 15 input-space neighbours are still neighbours after
> the projection. global corr = how well all the pairwise distances survive (Spearman).

## In a long-running server

It's safe to call from a worker thread — a fresh RNG per call and no module-level state — so
`await asyncio.to_thread(umap_project, x, 2)` is fine.

You usually don't want to refit the whole layout on every request. Fit once, then drop new
points into the layout you already have:

```python
model = fit(window, 2)          # cache this — it pickles
xy    = transform(model, pts)   # cheap, and the picture stays put between requests
```

That `transform` is an approximation of a full refit — it keeps about **72%** of the local
overlap you'd get by refitting — so it's great for placing in-distribution points, and worth a
real refit once the data drifts. (A good "time to refit" signal: watch how far new points land
from their nearest training neighbour; 2–3× the usual distance means you've drifted.) A cached
5000×1024 model is about **20 MB**. Rolling windows ("add these, drop the old ones") and sparse
input aren't supported — densify first, and refit when the window slides.

## The honest catch

Per call, at 1024-dim and n=5000, fastumap is about **2× slower** than umap-learn (~40 s vs
~20 s). That isn't a missing optimisation — a vectorised numpy SGD just can't match numba's
in-place inner loop. So this is the right tool when the *import / cold-start* cost is what
hurts, and the wrong one when you're pushing big high-dimensional batches through it all day.

That gap is also why there's an optional native piece: I rewrote just the SGD in Rust.

```bash
pip install fastumap[accel]     # adds the compiled kernel; base stays pure numpy + scipy
```

It compiles once into a small wheel — no runtime JIT, so the fast import stays — and runs
about **1.7–1.9× faster** with slightly better overlap. fastumap picks it up automatically when
it's installed; without it you get the pure numpy path. Wheels are prebuilt for x86_64 and
aarch64 (Graviton).

## What I made sure of (there's a test for each)

- Nothing but numpy and scipy at runtime — no numba, no LLVM, no compiled code of my own.
- Imports in under 200 ms, deterministic to the byte, safe across threads.
- Never builds the full n×n distance matrix, so memory stays bounded even at 5000×1024.
- 2-D and 3-D both work, and the whole thing is type-checked (pyright strict).

## Trying it locally

```bash
make check     # lint, type-check, tests
make tox       # run the suite on Python 3.11 / 3.12 / 3.13
make fargate   # time import + fit under a tight CPU cap (docker --cpus=0.5 --memory=2g)
```

## About UMAP

This is an independent reimplementation of the UMAP algorithm
([McInnes, Healy & Melville, arXiv:1802.03426](https://arxiv.org/abs/1802.03426)) — not
affiliated with or endorsed by its authors, and not a drop-in replacement; the API here is
deliberately small. MIT licensed.
