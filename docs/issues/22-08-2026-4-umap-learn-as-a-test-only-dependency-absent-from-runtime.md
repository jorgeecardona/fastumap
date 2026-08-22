---
title: "#4 umap-learn as a test-only dependency, absent from runtime"
status: open
area: harness
owner: null
filed: 2026-08-22
scope: S
verify: test
---

umap-learn lives in the dev/bench extra, never in runtime deps. Comparison tests skip (not fail) when it is absent.

Done when: a test asserts umap/numba are not imported by `import fastumap` and not in [project.dependencies]; comparison tests use pytest.importorskip.

MEASURED: tests/test_no_heavy_deps.py asserts both; tests/test_umap_compare.py uses importorskip. Runtime deps are numpy+scipy only.