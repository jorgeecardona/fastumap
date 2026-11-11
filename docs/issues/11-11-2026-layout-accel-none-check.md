---
title: layout.py still checks _accel_optimize is not None
status: open
area: quality
owner: null
filed: 2026-11-11
scope: XS
verify: test
---

The kernel is required since 0.2.33, so `_accel_optimize` is never None in a real run.
`tests/test_speedup_ratio.py` sets it to None to time the numpy path. Replace that with an explicit
switch (for example `chunk_count > 1`, which already forces numpy) and drop the None check.
