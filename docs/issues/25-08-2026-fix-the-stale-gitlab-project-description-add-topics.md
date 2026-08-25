---
title: Fix the stale GitLab project description + add topics
status: done
area: release
owner: gentle-atlas-13a06
filed: 2026-08-25
fixed_in: 1ef2cea
scope: XS
verify: inspect
---

The GitLab project description still reads "A numba-free UMAP in pure numpy + scipy: imports in milliseconds, no LLVM/JIT at runtime." (MEASURED via `glab api projects/:id`) — stale after the 0.2.x merge: it still claims "pure numpy + scipy", omits clustering, and omits the native kernels. Project topics are empty ([]).

The PyPI summary is already current ("Fast, import-light UMAP and clustering with native kernels — milliseconds to import."), so align the GitLab description with the PyPI summary + README tagline, and add topics (e.g. umap, dimensionality-reduction, clustering, embeddings, rust, python) so the repo is discoverable.

Done when: the GitLab description matches the 0.2.x reality and topics are set (INSPECTED via `glab api projects/:id`).

MEASURED (2026-08-25, glab api PUT projects/:id, verified via GET): description is now "Fast, import-light UMAP and clustering — milliseconds to import, native Rust kernels, no import-time JIT." (no more "pure numpy + scipy"); topics = [clustering, dimensionality-reduction, embeddings, k-means, python, rust, uMap] (GitLab canonicalised umap→uMap). External GitLab-metadata change, so fixed_in marks the mainline HEAD at the time, not a code commit.