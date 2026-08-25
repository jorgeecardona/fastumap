---
title: graph.py exp overflow on extreme inputs (compute-then-select)
status: done
area: quality
owner: null
filed: 2026-08-25
fixed_in: 0790371
scope: XS
verify: test
---

The hypothesis property test surfaced this. `fuzzy_simplicial_set`'s membership-strength step at src/fastumap/graph.py:72 does `val = np.where(offset > 0.0, np.exp(-offset / mid[:, None]), 1.0)`. `np.where` evaluates BOTH branches for every element, so `np.exp(-offset/mid)` is computed even where `offset <= 0` (there `-offset/mid >= 0`, and for a tiny `mid` the exponent is large positive) → MEASURED `RuntimeWarning: overflow encountered in exp` during `test_projection_is_finite_and_shaped` (tests/test_properties.py) on small random inputs.

The result is still correct — the overflowed values are in the discarded branch (`offset <= 0` selects 1.0), and the projection output is finite (the test passes) — so this is cosmetic, but it litters logs with a scary warning. Fix by only exponentiating the kept elements (mask first, or clip the exponent), so no overflow is computed.

Done when: the projection runs on the property-test inputs with no numpy overflow warning, and output stays bit-identical to now.