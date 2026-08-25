---
title: Machine-independent perf tests via operation / iteration counts
status: open
area: quality
owner: null
filed: 2026-08-25
scope: M
verify: test
---

The speed tests assert absolute wall-clock with 3x-loose ceilings (tests/test_time_budget.py: 25s/120s at n=1000/5000; tests/test_import_cost.py: 200ms). These are machine-dependent — they flake on a slow or loaded runner (they nearly did under today's swap-thrashed box) and only catch gross regressions.

Replace the speed-regression intent with deterministic invariants that count WORK, not time: k-means converges in ≤ N Lloyd iterations; the blocked _assign allocates exactly ⌈n/_BLOCK⌉ blocks (src/fastumap/clustering.py:82, _BLOCK=4096); the SGD performs exactly n_epochs × n_edges updates. Instrument with counters or tracemalloc. tests/test_memory.py is the existing model (asserts peak MB, an invariant, not seconds).

Done when: tests assert these counts and pass byte-identically regardless of host speed; a regression that doubles the iteration/allocation count fails them.