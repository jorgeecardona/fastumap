---
title: "#8 Import-cost test: subprocess import under 200ms"
status: done
area: production
owner: null
filed: 2026-08-22
fixed_in: d2a1e69
scope: XS
verify: test
---

Spawn a subprocess, import the package, assert wall time under 200ms. Fails loudly if a heavy import sneaks into __init__.

Done when: the test exists and enforces the budget.

MEASURED: tests/test_import_cost.py asserts <200ms (min of 5 warmed runs) and also that numpy is NOT imported by `import fastumap` (lazy __getattr__). Passes.