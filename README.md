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

Graded on separated Gaussian clusters (1024-dim unless noted), five seeds, against
umap-learn on the same data. We assert we are *within a tolerance* of the reference, never
that layouts match: different initialisation and update order rotate and reflect
equally-good embeddings.

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

`umap_project(x, 2, chunk_count=10)` closes most of the gap; it stays deterministic. The
real-data benchmark (MNIST / Fashion-MNIST at 5k/10k/20k) lands here next — roadmap #7.

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

xy  = umap_project(embeddings, dimensions=2)          # (n, 2)
xyz = umap_project(embeddings, dimensions=3)          # (n, 3)
init = spectral_project(embeddings, dimensions=2)     # just the spectral initialisation
```

Same input and seed give **bit-identical** output across processes (the layout must be
stable across reloads so people can compare the picture over time). `umap_project` takes
`n_neighbors`, `min_dist`, `spread`, `n_epochs`, `negative_sample_rate`, `random_state`.

## Guarantees (enforced by tests)

- **numpy + scipy only** at runtime — no numba, llvmlite, scikit-learn, or compiled
  extension of our own. A test asserts the JIT stack is never imported.
- **Import under 200 ms** — a subprocess test measures it.
- **Deterministic** — pinned ARPACK start vector, seeded negative sampler; identical bytes
  across processes.
- **Memory bounded** — the n-by-n distance matrix is never materialised (512-row blocked
  kNN). At the worst case (5000 × 1024) the fit adds ~102 MB over the import+input baseline
  (175 MB total peak); the test enforces a 200 MB ceiling.
- **2-D and 3-D**, both first-class. **Typed**, pyright strict.

## Releasing

CI (GitLab, moon + uv + proto) runs lint, format, typecheck, tests, build on every push.
On `main`, once the PyPI Trusted Publisher is configured and CI/CD variable `PYPI_READY=1`
is set, it publishes to PyPI via **OIDC Trusted Publishing** (no stored token) whenever the
version in `pyproject.toml` is not yet on PyPI.

## Not affiliated with UMAP

fastumap is an **independent reimplementation** of the UMAP algorithm
([McInnes, Healy, Melville, arXiv:1802.03426](https://arxiv.org/abs/1802.03426);
reference implementation [lmcinnes/umap](https://github.com/lmcinnes/umap)). It is not
affiliated with or endorsed by the UMAP authors, and it is **not a drop-in replacement** —
the public surface is deliberately small. MIT licensed.
