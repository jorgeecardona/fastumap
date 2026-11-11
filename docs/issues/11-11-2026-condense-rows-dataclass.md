---
title: hdbscan condensed tree rows are a bare tuple
status: open
area: quality
owner: null
filed: 2026-11-11
scope: XS
verify: test
---

`_density._condense` returns `list[tuple[int, int, float, int]]` (parent, child, lambda, size).
Make it a frozen dataclass so the fields have names.
