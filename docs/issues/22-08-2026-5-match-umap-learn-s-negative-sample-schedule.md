---
title: "#5 Match umap-learn's negative-sample schedule"
status: open
area: quality
owner: null
filed: 2026-08-22
scope: S
verify: test
---

Current optimiser (src/fastumap/layout.py optimize_layout) uses a FLAT negative_sample_rate per due edge. umap-learn derives the per-edge count from a second schedule: epochs_per_negative_sample = epochs_per_sample / negative_sample_rate, with a per-edge counter (epoch_of_next_negative_sample). Replace the flat rate with that schedule.

Done when: the grading harness shows overlap@15 at n=1000 and n=2000 before and after over five seeds, and the change is kept ONLY if it moves the number. Revert if it does not.

INFERRED gap (from brief): mine trails umap-learn by ~0.004 overlap, reproducible across seeds.