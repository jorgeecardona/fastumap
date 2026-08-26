---
title: fastumap benchmark report
date: 2026-08-26
generated_by: bench/report.py
---

# fastumap benchmark report — 2026-08-26

> **This run was measured on a busy machine.** The load average below is a large fraction of the core count, so every wall time here is inflated and the inflation is not even across methods. The quality columns (overlap@15, global corr) are deterministic and unaffected. Re-run on an idle machine before quoting the timings.

Generated, not hand-written. Regenerate with:

```
uv run python bench/report.py --dataset mnist --sizes 5000,10000,20000 --kmeans-n 200000 --k 64 --iter 20
```

Every number below comes from that one run on the machine described here. Comparing two reports
only means something when the machine rows match.

## Machine

| | |
|---|---|
| CPU | Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz |
| logical cores | 16 |
| load average (1/5/15 min) | 14.65 / 10.91 / 9.02 (over 16 cores) |
| RAM | 31.0 GB |
| platform | Linux-6.19.14+deb14-amd64-x86_64-with-glibc2.43 |
| python | 3.12.7 |
| native kernel active | True |

## Versions

| package | version |
|---|---|
| fastumap | 0.2.13 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| umap-learn | 0.5.12 |
| scikit-learn | 1.9.0 |
| faiss-cpu | 1.15.0 |

## Projection quality — mnist (784-dimensional)

fastumap vs umap-learn vs PCA at n = 5000, 10000, 20000, seed 42.

- **overlap@15** — the fraction of each point's 15 input-space neighbours still among its 15
  nearest neighbours in the embedding. Local structure. Higher is better.
- **global corr** — Spearman correlation between input-space and embedding pairwise distances.
  Global structure. Higher is better.

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

umap-learn's wall times in a single process reuse the numba compilation from its first fit; a
fresh process pays that compilation (roughly 20 s) on every invocation. fastumap has no
compilation step.

## Clustering — k-means

synthetic 128-d blobs. Inertia is the k-means objective (sum of squared distances to the assigned
centroid): lower is tighter. It sits beside the wall time on purpose — a faster k-means that
optimises less is not faster.

| implementation | n | k | wall time | inertia |
|---|---|---:|---:|---:|
| accel (Rust) | 200000 | 64 | 1.56 s | 1.105e+08 |
| numpy | 200000 | 64 | 2.52 s | 1.105e+08 |
