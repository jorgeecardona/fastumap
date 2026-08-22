---
title: "#17 README leads with the two numbers that justify the library"
status: open
area: release
owner: null
filed: 2026-08-22
scope: S
verify: inspect
---

README must lead with: import cost vs umap-learn, and the quality table showing what that import speed costs. Include the failure story (0.5 vCPU x 4 workers Fargate, numba LLVM compile at import = 148s wall, health check kills the task at ~120s) — it is the clearest statement of who should use this.

Done when: README opens with measured import cost and the umap-learn quality table, and tells the incident story. inspect.