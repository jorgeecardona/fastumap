---
title: Mini-batch k-means for constrained CPU (1M/<10s on 0.5–2 vCPU)
status: done
area: capability
owner: bronze-sentinel-79061
filed: 2026-08-25
fixed_in: 3d8842c3e819
scope: L
verify: probe
---

Full Lloyd k-means at 1M×128 is ~154 core-seconds (MEASURED: ~9.6s on a 16-core box on SIFT1M via fastumap._accel.kmeans). On Fargate ECS at 0.5–2 vCPU that is minutes, not seconds — Jorge's live SGD numbers show the shape: 250 pts 7.6s, 500 pts 28s, 1000 pts 83s, 5000 pts ~270s. Mini-batch k-means (sample B rows per iteration, update only touched centroids) cuts the per-iteration work ~10x (INFERRED ~15 core-seconds total), so 2–4 vCPU could reach <10s at 1M.

Implement in rust/src/lib.rs beside the existing kmeans_impl (blocked GEMM assignment already there); gate by n and core count in src/fastumap/clustering.py:kmeans (near the _ACCEL_MIN branch at line 129). This is the optimization deferred on 2026-08-25.

Done when: a probe under a 2-vCPU cap (cgroup / docker --cpus=2) clusters 1M×128 into ~64 groups in <10s with inertia within ~5% of full Lloyd.
## Update 2026-08-26 — bar met on SIFT1M under a real 2-vCPU cap

MEASURED (docs/bench/minibatch-1m-2026-08-26.md), `docker --cpus=2 --memory=6g`, SIFT1M
1,000,000 x 128, k=64, the wheel built from this tree:

| implementation | wall time | inertia |
|---|---:|---:|
| mini-batch | 5.2 s | 6.6601e+10 |
| full Lloyd | 25.3 s | 6.5935e+10 |
| ratio | 0.21x | 1.010x |

Both halves of the done-condition hold: under 10 s (5.2 s) and within ~5% of full Lloyd (1.010x).
Regenerate with `make minibatch-1m`.

DEVIATION from this issue's plan, deliberately: the issue said to gate "by n and core count". The
gate is on **n alone** (`_MINIBATCH_MIN = 200_000`). Gating on core count would make the same input
and seed produce different labels on different machines, which contradicts the determinism the
library sells; the environment switches already documented (native kernel vs numpy, `fastumap[ann]`)
are properties of the install, not of how many cores the box happens to have. `batch_size=0` forces
full Lloyd and a positive int forces mini-batch, so the constrained-CPU case is still reachable
explicitly at any n.

KNOWN WEAKNESS, pinned by test_minibatch_struggles_on_well_separated_blobs rather than hidden: on
perfectly separated clusters mini-batch can leave two centroids inside one blob and none in
another — MEASURED 4.8x worse inertia on 20000x32 with 16 blobs 8 sigma apart. The running-mean
update has no force to walk a centroid back across an empty gap; Lloyd's recompute-from-scratch
does. Real overlapping data does not trigger it (MNIST 60000x784: 1.013x, SIFT1M: 1.010x), which
is why the auto-switch is on. Tried and did NOT fix it: initialising the per-centroid count to 1
to avoid the first-point teleport, and raising iterations to 4000 — the objective did not move.
A proper fix is best-of-n_init seeding or low-count centroid reassignment, worth its own issue if
someone hits this on real data.
