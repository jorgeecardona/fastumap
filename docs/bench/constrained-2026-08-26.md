---
title: fastumap constrained-vCPU probe
date: 2026-08-26
generated_by: bench/constrained.sh
---

# Constrained-vCPU probe — 2026-08-26

Generated, not hand-written. Regenerate with:

```
CAPS="0.5 1 2" SIZES=250,500,1000,5000 DIM=1024 sh bench/constrained.sh
```

Each row is one `docker run --cpus=<cap> --memory=2g` on `python:3.13-slim-trixie`, installing the
wheel `fastumap-0.2.15-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` built from this tree. `--cpus` sets a CFS quota, the same mechanism
AWS Fargate's cgroup uses to meter CPU.

`fit` is `umap_project(x, 2)`; `cluster` is `cluster(x, method="kmeans", n_clusters=8)`
on the same input. Clustering never computes a layout first, which is why it is timed
separately rather than folded into the fit.

| vCPU cap | n | dims | fit | cluster |
|---|---:|---:|---:|---:|
| 0.5 | import | | 8 ms | |
| 0.5 | 250 | 1024 | 3.5 s | 0.20 s |
| 0.5 | 500 | 1024 | 2.8 s | 1.49 s |
| 0.5 | 1000 | 1024 | 5.6 s | 1.77 s |
| 0.5 | 5000 | 1024 | 33.2 s | 20.69 s |
| 1 | import | | 9 ms | |
| 1 | 250 | 1024 | 1.7 s | 0.11 s |
| 1 | 500 | 1024 | 1.5 s | 1.13 s |
| 1 | 1000 | 1024 | 2.9 s | 1.57 s |
| 1 | 5000 | 1024 | 16.7 s | 14.52 s |
| 2 | import | | 7 ms | |
| 2 | 250 | 1024 | 1.6 s | 0.02 s |
| 2 | 500 | 1024 | 1.4 s | 0.59 s |
| 2 | 1000 | 1024 | 2.7 s | 1.02 s |
| 2 | 5000 | 1024 | 15.8 s | 8.74 s |

## Against production

The reason this probe exists. The live AWS Fargate ECS numbers for the SGD path are the
ground truth it is checked against:

| n | Fargate (observed, 0.5 vCPU) | this run (0.5 vCPU) |
|---|---:|---:|
| 250 | 7.6 s | 3.5 s |
| 500 | 28 s | 2.8 s |
| 1000 | 83 s | 5.6 s |
| 5000 | 270 s | 33.2 s |

A local `--cpus` cap reproduces Fargate's *mechanism* (a CFS quota) but not its
*hardware*: half a core of the host CPU below is far more compute than half a shared
Fargate vCPU, and the container still sees every host core, so BLAS and rayon spread the
quota over many threads instead of one. Expect this probe to be faster in absolute terms
and flatter at small n, where fixed startup cost dominates. What it is good for is
**drift**: the same command on the same host, run again after a change.

## Host

| | |
|---|---|
| host CPU | Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz |
| host cores | 4 |
| host load at start | 4.68 5.05 5.21 |
| image | python:3.13-slim-trixie |
| memory cap | 2g |

The host load matters: `--cpus` bounds the container's CPU-time but cannot reserve it,
so a busy host inflates every row here. Compare runs whose host rows match.
