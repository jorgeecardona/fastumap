---
title: k-means auto-select k (elbow / silhouette)
status: done
area: capability
owner: null
filed: 2026-08-25
fixed_in: 90b439b
scope: M
verify: test
---

cluster() and kmeans() require an explicit n_clusters (src/fastumap/clustering.py:67 raises otherwise). Add an optional automatic k: run k over a small range and pick by the inertia elbow (knee) or silhouette on a subsample. Keep it deterministic and bounded — subsample for silhouette, reuse the accel kmeans for each candidate, cap the range.

Done when: on well-separated blobs generated with a known true k, the auto path recovers k within ±1 in a test, and the docstring/README state it is a heuristic (approximate).