---
title: "#5 Match umap-learn's negative-sample schedule"
status: done
area: quality
owner: null
filed: 2026-08-22
fixed_in: 8d02639
scope: S
verify: test
---

Current optimiser (src/fastumap/layout.py optimize_layout) uses a FLAT negative_sample_rate per due edge. umap-learn derives the per-edge count from a second schedule: epochs_per_negative_sample = epochs_per_sample / negative_sample_rate, with a per-edge counter (epoch_of_next_negative_sample). Replace the flat rate with that schedule.

Done when: the grading harness shows overlap@15 at n=1000 and n=2000 before and after over five seeds, and the change is kept ONLY if it moves the number. Revert if it does not.

INFERRED gap (from brief): mine trails umap-learn by ~0.004 overlap, reproducible across seeds.
## Update 2026-08-22 — schedule alone is neutral, reverted

MEASURED (fastumap-only, 5 seeds, dim=64, 8 clusters), flat rate vs per-edge schedule:
- n=1000: 0.2338 ± 0.0039  →  0.2335 ± 0.0051  (Δ within noise)
- n=2000: 0.1387 ± 0.0031  →  0.1380 ± 0.0017  (Δ within noise)

The per-edge negative-sample schedule alone does NOT move overlap@15 in this regime, so it was reverted per the brief's "revert if it does not move the number". Keep OPEN: the brief pairs it with #6 (in-epoch feedback); the schedule may only pay off in combination. Next: try #6, then re-test #5 on top of a winning #6.
