---
title: hdbscan() — hierarchical density clustering over the kNN graph
status: done
area: capability
owner: bronze-sentinel-79061
filed: 2026-08-26
fixed_in: 49ff821cd554
scope: L
verify: test
---

UMAP -> HDBSCAN is the standard pairing (same author, Leland McInnes), and it is the clusterer
people reach for after a reduction. fastumap has `dbscan()` (src/fastumap/clustering.py:187) but
not HDBSCAN, so a user still has to pick `eps` — which is the one parameter DBSCAN is bad at and
HDBSCAN removes.

MEASURED: the pieces already exist in this repo.
- core distance = distance to the `min_samples`-th neighbour: already computed at
  src/fastumap/clustering.py:222 (`dists[:, col]`) for the eps heuristic.
- union-find with path halving: src/fastumap/clustering.py:232-244, inside `_dbscan`.
- kNN with the blocked/faiss backends: `knn_graph` in src/fastumap/neighbors.py:201.

INFERRED (the work that is not there yet), in order:
1. mutual reachability on each kNN edge: max(core_i, core_j, d_ij).
2. MST over those edges — Kruskal, reusing the union-find.
3. single-linkage tree: sorted MST edges, recording merge height and component sizes.
4. condensed tree: walk down, a child under `min_cluster_size` falls out as noise, otherwise
   it is a real split.
5. cluster extraction by stability (excess of mass); unselected points get -1.

Steps 4 and 5 are the real work, roughly 200 lines. 1-3 are short.

KNOWN LIMITATION to state in the docstring, not to hide: building the MST from the kNN graph
rather than the full mutual-reachability graph makes this APPROXIMATE HDBSCAN, and the kNN graph
can be disconnected, so the MST is a forest and the code must handle several components. Same
tradeoff RAPIDS makes; it is the only version that fits the existing kNN pipeline.

Out of scope for a first cut: soft membership and GLOSH outlier scores.

Done when: `hdbscan()` is exported from `fastumap`, reachable as `cluster(method="hdbscan")`
alongside the existing dbscan route, and a test shows it recovers clusters of unequal density
that DBSCAN with a single global eps cannot (two blobs at different densities, plus noise
labelled -1), with no `eps` argument anywhere in the call.