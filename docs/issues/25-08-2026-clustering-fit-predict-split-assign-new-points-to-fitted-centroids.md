---
title: "Clustering fit/predict split: assign new points to fitted centroids"
status: done
area: capability
owner: null
filed: 2026-08-25
fixed_in: d891268
scope: S
verify: test
---

There is no way to assign NEW points to already-fitted centroids without re-clustering everything. Mirrors the projection fit/transform split that already exists (UMAPModel.transform).

Add a predict step: given centroids from a prior cluster()/kmeans() run and new rows, return nearest-centroid labels using the existing blocked _assign (src/fastumap/clustering.py:77) so peak memory stays bounded. No new dependency.

Done when: a test fits on n rows, predicts on m held-out rows, and the predicted labels equal a full nearest-centroid assignment against the same centroids.