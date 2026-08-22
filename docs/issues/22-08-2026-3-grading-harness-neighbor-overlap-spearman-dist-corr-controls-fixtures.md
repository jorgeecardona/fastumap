---
title: "#3 Grading harness: neighbor_overlap + Spearman dist-corr, controls, fixtures"
status: open
area: harness
owner: null
filed: 2026-08-22
scope: M
verify: test
---

`neighbor_overlap(original, projected, k)` and Spearman pairwise-distance correlation, a random-layout control, and fixtures: separated Gaussian clusters, curved 1-D manifold in high dim, collinear, identical, disconnected kNN graph, n just above minimum.

Done when: one command prints a table of both metrics for fastumap, umap-learn, PCA, random over five seeds with mean and std.

MEASURED: metrics in src/fastumap/metrics.py; `python -m fastumap.grade` prints the 4-layout table over N seeds; fixtures in tests/fixtures.py. Verify the printed table looks right (inspect) as a follow-up.