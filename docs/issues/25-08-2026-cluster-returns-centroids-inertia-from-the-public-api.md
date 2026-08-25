---
title: cluster() returns centroids + inertia from the public API
status: open
area: capability
owner: null
filed: 2026-08-25
scope: S
verify: test
---

cluster() (src/fastumap/clustering.py:46) returns labels only. The standalone kmeans() returns (labels, centroids) but neither exposes inertia. Model selection (elbow/silhouette, comparing k) needs centroids and the final inertia without recomputing them.

Add an opt-in (e.g. return_centroids=True, or a small result object) that exposes centroids (k, d) and the final inertia from BOTH paths: the accel path recovers centroids via _centroids_from_labels (line 104); inertia is one blocked pass reusing _assign's distance math (line 77).

Done when: a test asserts cluster(..., return_centroids=True) yields centroids of shape (k, d) and a finite inertia equal (within tolerance) to a manual sum of squared distances to the assigned centroid.