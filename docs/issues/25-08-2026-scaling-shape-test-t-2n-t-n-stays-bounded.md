---
title: "Scaling-shape test: t(2n)/t(n) stays bounded"
status: open
area: quality
owner: null
filed: 2026-08-25
scope: S
verify: test
---

No test guards algorithmic complexity. A change turning the blocked O(n·k) assignment (src/fastumap/clustering.py:77) or a linear-in-n stage quadratic would only surface as a loose wall-clock miss, if at all.

Add a scaling test: measure work (operation count, or median time of several repeats) at n and 2n and assert the ratio t(2n)/t(n) < ~2.5 for the stages that must be linear in n. Machine-independent because it is the curve's ratio, not the value; use medians + generous bounds to absorb noise.

Done when: a test asserts the ratio bound for kmeans assignment (and a linear projection stage) and fails when an O(n^2) loop is injected.