---
title: An explicit n_iter=50 is replaced by 100 on the mini-batch path
status: open
area: capability
owner: null
filed: 2026-11-11
scope: S
verify: test
---

`_kmeans.kmeans_single` uses `n_iter == 50` to mean "the caller left the default", and then runs
100 mini-batch iterations. A caller who passes `n_iter=50` on purpose gets 100. The fix needs
`n_iter: int | None = None` on `kmeans` and `cluster`, which changes a public default.
