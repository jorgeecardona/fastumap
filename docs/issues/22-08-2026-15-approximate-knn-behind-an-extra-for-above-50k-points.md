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