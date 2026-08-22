---
title: "#1 Repository skeleton (uv, src layout, ruff, pyright strict, CI)"
status: open
area: harness
owner: null
filed: 2026-08-22
scope: M
verify: test
---

Stand up the project: uv + pyproject.toml, src layout, ruff lint/format, pytest, pyright strict, and a GitLab CI workflow (moon+uv+proto, replicated from musil) running lint, typecheck, tests.

Done when: `uv run pytest`, `uv run ruff check`, and `uv run pyright` all pass, and CI runs them on push.

MEASURED (2026-08-22): scaffold written; pyright 0 errors, ruff clean, core tests pass. Note: pyright strict keeps `reportMissingTypeStubs` off (scipy/sklearn/umap ship no stubs) and the three numpy `reportUnknown*Type` reports off (numpy 2.x `floating[Any]` stubs) — documented in pyproject.toml. This deviates from the brief's "no suppressions" and was flagged to Jorge.