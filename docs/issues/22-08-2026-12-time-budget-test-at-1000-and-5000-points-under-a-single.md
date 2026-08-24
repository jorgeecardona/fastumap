---
title: "#12 Time-budget test at 1000 and 5000 points under a single-core cap"
status: done
area: production
owner: null
filed: 2026-08-22
fixed_in: 3f054f9
scope: M
verify: test
---

Assert wall clock at n=1000 and n=5000 (1024 dim) against a documented ceiling, measured under a single-core cap (taskset/OMP/threadpool) so the number means something on 0.5 vCPU Fargate.

Done when: CI fails if a change makes the 1000-point case slower than the stated budget. Pick the budget from the worst case: 5000 points of 1024 dim.