---
title: "#7 Real-data benchmark: MNIST + Fashion-MNIST via cached downloader"
status: open
area: quality
owner: null
filed: 2026-08-22
scope: M
verify: test
---

Small cached downloader for MNIST and Fashion-MNIST (bench/data/, gitignored). Committed script produces a table of both metrics and wall-clock for fastumap, umap-learn, PCA at 5k, 10k, 20k points.

Done when: README carries that table, produced by bench/mnist.py.