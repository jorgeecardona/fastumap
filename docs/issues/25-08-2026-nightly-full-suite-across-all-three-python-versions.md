---
title: Nightly full suite across all three Python versions
status: open
area: harness
owner: null
filed: 2026-08-25
scope: S
verify: test
---

CI runs the full suite once (ci job) and only a SUBSET across 3.11/3.12/3.13 (test-matrix job in .gitlab-ci.yml runs test_projection, test_find_ab_params, test_no_heavy_deps, test_version, test_import_cost), kept small to spare the memory-limited runner. Version-specific breakage outside that subset is uncaught until a release.

Add a scheduled (nightly) pipeline that runs the FULL tox matrix (all tests × 3.11/3.12/3.13) where the schedule can tolerate the longer run.

Done when: a scheduled pipeline exists and runs the full tox suite green on all three interpreters.