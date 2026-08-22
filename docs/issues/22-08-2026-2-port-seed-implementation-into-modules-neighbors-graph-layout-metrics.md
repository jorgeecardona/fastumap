---
title: "#2 Port seed implementation into modules (neighbors, graph, layout, metrics)"
status: open
area: harness
owner: null
filed: 2026-08-22
scope: L
verify: test
---

The seed at ~/Documents/repos/classifier/.../projection.py is GONE (repo deleted; confirmed absent on disk 2026-08-22). Jorge chose "build from the brief" — the brief fully specifies the algorithm and constants. Implement directly against umap-learn as reference, split into neighbors/graph/layout/metrics with `umap_project`/`spectral_project` as the public API.

Done when: public API is `umap_project` and `spectral_project`; the projection tests pass.

MEASURED: implemented in src/fastumap/{neighbors,graph,layout,projection,metrics}.py; 17 core tests pass incl. shapes, determinism, degenerate inputs.