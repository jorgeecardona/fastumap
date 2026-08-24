---
title: "#9 Memory ceiling test: peak RSS at 5000x1024 under a stated bound"
status: done
area: production
owner: null
filed: 2026-08-22
fixed_in: 9a536f6
scope: S
verify: test
---

Assert peak RSS for a 5000-point, 1024-dim fit stays under a documented bound (brief: +121MB over baseline with 512-row blocking; was 733MB before blocking). Measure with psutil (dev dep) in a subprocess; state the bound in the README.

Done when: the test runs in CI and the bound is documented. knn_euclidean already blocks at block_size=512 (src/fastumap/neighbors.py) so the n-by-n matrix is never materialised.