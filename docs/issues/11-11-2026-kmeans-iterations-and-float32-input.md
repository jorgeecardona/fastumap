---
title: k-means does not report iterations and copies float32 input to float64
status: open
area: capability
owner: null
filed: 2026-11-11
scope: M
verify: test
---

From GitLab #18 and #29. Two parts, both need the Rust binding:

1. Return the number of iterations the kernel ran, so a caller can budget a 1M job.
2. Accept float32 in `_accel.kmeans` and `minibatch_kmeans`. Today Python copies the input to
   float64 first: 3 GB at 500k x 768, 8 GB at 1M x 1024.
