---
title: Coverage measurement + threshold gate
status: open
area: quality
owner: null
filed: 2026-08-25
scope: S
verify: inspect
---

No line coverage is measured. With 87 tests across 25 files the suite is broad, but nothing reports which branches are exercised, and the numpy fallback branches (used only when fastumap._accel is absent) may be under-covered now that CI always builds the extension.

Add pytest-cov to the gate, report line coverage, and set a floor starting at the current number and ratcheting up. Justify any explicit exclusions (bench/, the accel-absent fallbacks).

Done when: CI prints a coverage number and fails below the configured threshold (INSPECTED in the job log).