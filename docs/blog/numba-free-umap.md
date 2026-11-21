---
title: "The problem was never numba: it was *when* it compiles"
draft: true
---

# The problem was never numba: it was *when* it compiles

A projection endpoint took down our service, and the fix turned into a small library and a
lesson about where compilation belongs.

## The incident

An internal API renders UMAP projections of 1024-dim embeddings. It runs on AWS Fargate:
**0.5 vCPU, four `granian` workers, 2 GB**, shared with a Datadog sidecar. `granian` binds
the port immediately, so the health check starts polling from the first second. But nothing
answers until the app module finishes importing.

`from umap import UMAP` costs ~4 CPU-seconds to *import*, because its kernels are numba and
numba compiles with LLVM **at import time**. `pynndescent` alone declares 46 eagerly-compiled
`@njit` functions. On four workers metered through a half-vCPU cgroup, that import took **148
seconds of wall clock**. The health check kills the container at ~120 s. So the service
crash-looped and never served a request; ECS replaced the task, the replacement paid the same
cost, and the deployment rolled back.

Things that did *not* fix it: baking a warm numba cache into the image (11.9 s vs 12.0 s: the
eager-signature functions ignore the cache). `NUMBA_DISABLE_JIT=1` works but the pure-Python
kernels are ~100× slower.

## The insight

The problem was never numba the compiler. It was **import-time** compilation. numpy and scipy
are just as compiled. But they ship **precompiled** in the wheel, so they import in
milliseconds. The whole 172 MB of `llvmlite` + numba exists to compile *on the machine, at
import*, which is exactly what a cold, CPU-throttled container can't afford.

So: reimplement UMAP in pure **numpy + scipy**. No JIT, nothing to compile at runtime.

## The build

`fastumap` is a faithful port of the algorithm: `smooth_knn_dist`, the fuzzy-union
symmetrisation, `find_ab_params` (a = 1.5769, b = 0.8951, verified equal), a spectral init via
`scipy.sparse.linalg.eigsh`, and a vectorised attract/repel SGD, all in numpy.

| | import (cold) | wheels added |
|---|---:|---:|
| `import umap` | **~21 s** on a laptop; **148 s** at 0.5 vCPU ×4 | ~172 MB |
| `import fastumap` | **~12 ms** | 0 MB |

On real MNIST it stays within **0.01–0.02** neighbour-overlap of umap-learn (0.333 vs 0.344 at
5k), and boots the container in ~10 s instead of 148. The service came back.

## The honest tradeoff

Reproducing quality was the easy part; per-call speed was not. At 1024-dim, n = 5000, fastumap
is roughly **2× slower per call** than umap-learn. That's inherent: the SGD dominates wall
clock, and a vectorised-numpy SGD can't match numba's compiled *in-place* (Hogwild) walk.

I tried the obvious micro-optimisations and mostly reverted them: float32 distances (regressed
quality at high dim), batched negative sampling (neutral, unprovable under load). PCA
pre-reduction helped the kNN by ~9%, but the kNN wasn't the bottleneck. Honest dead-ends beat
imaginary wins.

## The twist

Here's the thing the incident actually taught me: **numba's problem was the *timing* of
compilation, not compilation itself.** So compile the hot kernel: but at *build* time.

A ~150-line Rust reimplementation of the SGD (PyO3 + maturin) does umap-learn's true in-place
walk. It compiles once, into a tiny `.so` that imports in milliseconds: no runtime JIT, no
llvmlite. Measured against the numpy SGD:

| n | rust | numpy | speedup | overlap (rust / numpy) |
|---|---:|---:|---:|---:|
| 1000 | 2.4 s | 4.5 s | **1.9×** | 0.244 / 0.233 |
| 2000 | 5.0 s | 8.6 s | **1.7×** | 0.151 / 0.140 |

Faster *and* higher quality: because it's the real in-place walk, not a snapshot
approximation. It builds as an **abi3** wheel (`cp311-abi3`): one binary that works on Python
3.11 and every version after, no per-version rebuild. It ships built into the `fastumap` wheel;
the numpy path stays as the fallback for a source build without a Rust toolchain.

So the false choice (fast import *or* native speed) dissolves. You can have both. You just
have to compile at the right moment.

## The takeaway

When a dependency's install or import is mysteriously slow on constrained infra, ask *when it
compiles*, not *whether*. Import-time JIT (numba) trades every cold start for peak throughput: a terrible trade on Lambda/Fargate. Build-time compilation (Rust, Cython, C) gives you the
native speed with a millisecond import. Same machine code; a very different bill.

*fastumap is MIT-licensed and an independent reimplementation of UMAP
([McInnes et al., arXiv:1802.03426](https://arxiv.org/abs/1802.03426)), not affiliated, not a
drop-in replacement.*
