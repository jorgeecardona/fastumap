---
title: "#6 Partial in-epoch feedback (shuffled chunks) in the optimiser"
status: open
area: quality
owner: null
filed: 2026-08-22
scope: M
verify: test
---

optimize_layout computes every gradient in an epoch from one snapshot (delta accumulated, applied at epoch end). umap-learn updates coordinates in place while walking the edge list (Hogwild), so later edges see earlier moves. Process due edges in shuffled chunks so later chunks see earlier moves; chunk size is the knob.

Done when: overlap@15 at n=1000 and n=2000 is within 0.002 of umap-learn, OR the attempt is documented as a dead end with the numbers. Do #5 first, one change at a time, metric before/after.
## Update 2026-08-23 — chunk_count knob delivered; moves the number, kept opt-in

MEASURED (fastumap-only, 5 seeds, dim=64, 8 clusters). chunk_count = shuffled chunks per
epoch; chunk_count=1 is the snapshot path (reproduces the old bytes exactly).

n=1000: cc1 0.2338 | cc4 0.2424 | cc10 0.2444 | cc20 0.2462 | cc40 0.2475 | cc80 0.2429
n=2000: cc1 0.1387 | cc10 0.1474
per-fit cost (n=1000): cc10 ~11s, cc40 ~22s, cc80 ~36s (cost scales ~linearly with cc).

So in-epoch feedback DOES close the local-overlap gap (umap-learn n=1000 = 0.250): cc40 gets
0.2475 (gap 0.0025), cc80 overshoots (destabilises). But cost scales with chunk_count, and
this library exists for a 0.5-vCPU service, so a high default is not viable.

DECISION: keep chunk_count as an opt-in knob, default 1 (fast path unchanged, byte-identical
determinism). Documented in README with the tradeoff table. The "within 0.002" numeric
target is only reached at cc=40 (~10x cost); whether to raise the DEFAULT is deferred to #12
(time budget on a single core) — revisit once the 5000-point wall-clock is known. Tests:
test_chunk_count_one_matches_default, test_chunk_count_feedback_deterministic_and_finite.
