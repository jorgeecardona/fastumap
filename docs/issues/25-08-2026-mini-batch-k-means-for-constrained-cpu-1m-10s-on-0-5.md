---
title: Mini-batch k-means for constrained CPU (1M/<10s on 0.5–2 vCPU)
status: open
area: capability
owner: null
filed: 2026-08-25
scope: L
verify: probe
---

Full Lloyd k-means at 1M×128 is ~154 core-seconds (MEASURED: ~9.6s on a 16-core box on SIFT1M via fastumap._accel.kmeans). On Fargate ECS at 0.5–2 vCPU that is minutes, not seconds — Jorge's live SGD numbers show the shape: 250 pts 7.6s, 500 pts 28s, 1000 pts 83s, 5000 pts ~270s. Mini-batch k-means (sample B rows per iteration, update only touched centroids) cuts the per-iteration work ~10x (INFERRED ~15 core-seconds total), so 2–4 vCPU could reach <10s at 1M.

Implement in rust/src/lib.rs beside the existing kmeans_impl (blocked GEMM assignment already there); gate by n and core count in src/fastumap/clustering.py:kmeans (near the _ACCEL_MIN branch at line 129). This is the optimization deferred on 2026-08-25.

Done when: a probe under a 2-vCPU cap (cgroup / docker --cpus=2) clusters 1M×128 into ~64 groups in <10s with inertia within ~5% of full Lloyd.