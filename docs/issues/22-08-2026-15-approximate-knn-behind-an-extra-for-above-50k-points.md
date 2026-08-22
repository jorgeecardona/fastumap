---
title: "#15 Approximate kNN behind an extra, for above ~50k points"
status: open
area: capability
owner: null
filed: 2026-08-22
scope: L
verify: test
---

Brute force is O(n^2); blocking bounds memory, not time. For >~50k points add an approximate kNN backend behind an optional extra, keeping the base install pure numpy+scipy. Candidate: a pure-numpy random-projection forest, or an optional dependency gated so it never touches the base closure.

Done when: an approximate path exists behind an extra, the base install stays numpy+scipy, and a test shows recall vs brute force above a threshold.
## Update 2026-08-23 — motivating evidence + needs a design decision

MEASURED (MNIST, single-thread, one process): at n=20000 fastumap's brute-force kNN + SGD
took 110s vs umap-learn 31s (umap's numba was warm from earlier sizes). fastumap wins at
5k/10k but its O(n^2) kNN loses at 20k — exactly this issue. Quality stays within 0.02
overlap at all sizes, so only speed-at-scale is the gap.

DECISION NEEDED (structural — for Jorge, not to be made autonomously): the approximate
backend "behind an extra" is one of:
  A) Optional compiled dep, e.g. `fastumap[ann]` -> hnswlib (C++, not numba). Fast, mature,
     high recall; base install stays pure numpy+scipy. Adds a build/runtime dep for opt-in
     users. Cleanest path to real >50k performance.
  B) Pure-numpy random-projection forest / LSH implemented in-repo. No new dep, but a real
     chunk of code to get recall + speed right, and still Python-loop-bound.
  C) scipy.spatial.cKDTree (already a dep) — exact, compiled, but degrades badly at 784+
     dims (curse of dimensionality); not a real fix for high-dim embeddings.

Recommendation: A (hnswlib behind an extra) — it matches "keep the base pure numpy+scipy"
and actually beats O(n^2). Left OPEN pending that call. Not started in code.
