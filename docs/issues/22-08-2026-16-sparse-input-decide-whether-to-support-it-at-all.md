---
title: "#16 Sparse input: decide whether to support it at all"
status: done
area: capability
owner: null
filed: 2026-08-22
fixed_in: abec612
scope: S
verify: inspect
---

Decide whether to accept scipy.sparse input. The first consumer is dense 1024-dim pgvector CLS embeddings, so sparse may be out of scope. Write the decision down either way; if yes, the blocked kNN needs a sparse distance path.

Done when: a short decision note in the README (or CHANGELOG) states supported/not, with the reason. inspect.