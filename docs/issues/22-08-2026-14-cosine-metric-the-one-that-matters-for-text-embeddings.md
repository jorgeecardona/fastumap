---
title: "#14 Cosine metric (the one that matters for text embeddings)"
status: done
area: capability
owner: null
filed: 2026-08-22
fixed_in: 6c3a41f
scope: M
verify: test
---

Add metric="cosine" to umap_project/spectral_project. euclidean on unnormalised encoder output is rarely what you want for 1024-dim CLS embeddings (the first consumer). Implement cosine distance in the blocked kNN (normalise rows, then 1 - cosine sim; the block matmul already gives dot products).

Done when: metric="cosine" works end to end, is deterministic, and a test shows it beats euclidean on a normalised-embedding fixture.