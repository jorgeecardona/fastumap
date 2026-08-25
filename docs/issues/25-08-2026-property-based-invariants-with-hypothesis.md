---
title: Property-based invariants with hypothesis
status: open
area: quality
owner: null
filed: 2026-08-25
scope: M
verify: test
---

Tests use fixed fixtures (gaussian_clusters, blobs). Add hypothesis property tests for invariants that must hold for ARBITRARY valid input, catching edge cases fixtures miss: cluster() labels are always in [0, k) and use ≤ k distinct values; projection output is finite and shaped (n, out_dim); cosine metric == euclidean on unit-normalised rows; determinism holds for any generated seed.

Done when: hypothesis-driven tests cover these invariants and pass over generated inputs (bounded sizes to keep CI fast).