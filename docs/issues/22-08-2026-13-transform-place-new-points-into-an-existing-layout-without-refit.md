---
title: "#13 transform(): place new points into an existing layout without refit"
status: done
area: capability
owner: null
filed: 2026-08-22
fixed_in: c555a18
scope: L
verify: test
---

Add transform() so new points are placed into an existing layout without recomputing it — what makes a drift view incremental rather than a full recompute per request. Follows umap-learn's transform: kNN of new points against the training set, membership, then a few optimisation epochs holding the trained embedding fixed.

Done when: transform(new) is deterministic, keeps the base layout stable, and a test shows new points land near their training neighbours.