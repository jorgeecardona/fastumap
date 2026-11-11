---
title: pca_reduce lives in projection.py but clustering uses it too
status: open
area: quality
owner: null
filed: 2026-11-11
scope: XS
verify: inspect
---

`cluster(pca_dim=...)` imports `pca_reduce` from `projection.py` lazily. Move it to a small shared
module (for example `_pca.py`) so clustering does not depend on the UMAP module.
