---
title: In-process ratio test for the native-vs-numpy speedup
status: done
area: quality
owner: null
filed: 2026-08-25
fixed_in: 750ad0b
scope: S
verify: test
---

The native speedup claims (rust/README: ~1.7–1.9x for the SGD; ~10s vs minutes for k-means) have no test — nothing would catch the compiled path regressing to slower than the numpy fallback.

Add an in-process ratio test: time the native path and the numpy fallback back-to-back on the same input and seed, assert native ≤ numpy (or speedup ≥ a loose factor), best-of-k to damp noise. The machine cancels in the ratio, so no absolute budget is needed. Force the numpy path by the chunk_count>1 lever (layout) or by temporarily nulling _accel_kmeans (clustering).

Done when: a test compares the two paths in one process and asserts the native path is not slower; skipped when fastumap._accel is absent.