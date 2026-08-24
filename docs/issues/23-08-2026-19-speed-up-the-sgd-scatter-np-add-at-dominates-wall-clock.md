---
title: "#19 Speed up the SGD scatter (np.add.at dominates wall-clock)"
status: done
area: production
owner: null
filed: 2026-08-23
fixed_in: 017a0c0
scope: M
verify: test
---

MEASURED (single-thread, dim=1024): umap_project wall-clock is 7.7s at n=1000 and 39.5s at n=5000 (src/fastumap/layout.py optimize_layout, default chunk_count=1, 500 epochs). On 0.5 vCPU (throttled ~4-8x) that is ~30-60s / ~150-300s per request — too slow for the async request path the library targets.

The bottleneck is `np.add.at(delta, idx, grad)` in optimize_layout (attraction to head+tail, plus negative_sample_rate repulsions), i.e. ~7 scatters/epoch x 500 epochs. np.add.at is a slow element-by-element scatter.

Fix to try: since the output is 2-3 dims, replace each np.add.at with per-dimension `np.bincount(idx, weights=grad[:, d], minlength=n)`, which is a compiled reduction and typically 5-20x faster than add.at. Verify: (a) overlap@15 unchanged over 5 seeds at n=1000/2000 (bincount sums in a different float order, so bytes differ but quality must not), (b) determinism still holds run-to-run, (c) the time-budget test (#12) tightens. Then lower the #12 ceilings to match.

Done when: n=1000 wall-clock drops materially (target <3s single-thread) with overlap@15 within noise of current, all tests green, #12 ceilings updated.
## Update 2026-08-23 — bincount scatter landed (partial win, kept open)

MEASURED (single-thread, dim=1024): np.add.at -> per-dim np.bincount in optimize_layout.
- n=1000: 7.7s -> 5.3s (-31%)
- n=5000: 39.5s -> 32.1s (-19%)
overlap@15 unchanged (n=1000 0.2338->0.2331, n=2000 0.1387->0.1381, within noise);
run-to-run determinism preserved. Committed in 0.1.3.

The scatter is no longer the sole bottleneck — the remaining cost is the per-epoch Python
overhead (500 iterations) and the 5x inner negative-sample loop (einsum/where/power on all
due edges, five times). The <3s target is not met (5.3s). NEXT: batch the negative samples
(draw negative_sample_rate*m at once instead of a Python loop of 5), and/or reduce per-epoch
overhead. Keeping OPEN for that follow-up.
