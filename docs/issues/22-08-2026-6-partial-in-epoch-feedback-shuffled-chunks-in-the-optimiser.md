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