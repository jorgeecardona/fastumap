"""Density clustering internals: dbscan and hdbscan labels from a kNN graph.

The public ``dbscan`` and ``hdbscan`` live in ``fastumap.clustering``. Only numpy is imported here
at module load; anything heavier is imported inside the function that needs it.
"""

from __future__ import annotations

import numpy as np

from ._types import FloatArray, IntArray


def _uf_find(parent: list[int], a: int) -> int:
    """Union-find root of ``a`` with path halving; ``parent`` is mutated in place. Shared by dbscan,
    the mutual-reachability MST, and single linkage, which otherwise each reimplemented it."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def dbscan_from_knn(indices: IntArray, dists: FloatArray, eps: float, min_samples: int) -> IntArray:  # noqa: C901
    """DBSCAN over a precomputed kNN. Core points are unioned to their eps-neighbours; borders take
    a neighbouring core's label; the rest are noise (-1). Complexity 13: the union pass, the
    core-relabel pass and the border pass are one algorithm; splitting them would not read better."""
    n = indices.shape[0]
    within = dists <= eps  # (n, k); column 0 is self at distance 0, so always within
    core = within.sum(axis=1) >= min_samples  # (n,) bool

    parent = list(range(n))

    def union(a: int, b: int) -> None:
        ra, rb = _uf_find(parent, a), _uf_find(parent, b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    core_points = np.where(core)[0]
    for i in core_points:
        ii = int(i)
        row_idx = indices[ii]
        row_within = within[ii]
        for pos in range(row_idx.shape[0]):
            if row_within[pos]:
                j = int(row_idx[pos])
                if j != ii and core[j]:
                    union(ii, j)

    labels = np.full(n, -1, dtype=np.int64)
    remap: dict[int, int] = {}
    for i in core_points:
        root = _uf_find(parent, int(i))
        if root not in remap:
            remap[root] = len(remap)
        labels[int(i)] = remap[root]
    # border points: a non-core point within eps of a core takes that core's cluster label.
    for i in range(n):
        if core[i] or labels[i] != -1:
            continue
        row_idx = indices[i]
        row_within = within[i]
        for pos in range(row_idx.shape[0]):
            if row_within[pos] and core[int(row_idx[pos])]:
                labels[i] = labels[int(row_idx[pos])]
                break
    return labels


def hdbscan_from_knn(
    indices: IntArray, dists: FloatArray, core: FloatArray, min_cluster_size: int
) -> IntArray:
    """HDBSCAN over a precomputed kNN and its core distances: MST -> single-linkage tree ->
    condensed tree -> stability selection -> labels."""
    n = indices.shape[0]
    edges = _mutual_reachability_mst(indices, dists, core)
    left, right, height, size = _single_linkage(edges, n)
    rows = _condense(left, right, height, size, n, min_cluster_size)
    return _labels_from_condensed(rows, n)


def _mutual_reachability_mst(
    indices: IntArray, dists: FloatArray, core: FloatArray
) -> list[tuple[float, int, int]]:
    """Kruskal over the kNN edges, weighted by mutual reachability. Returns exactly ``n - 1``
    edges ``(weight, a, b)`` ascending. The kNN graph can be disconnected, so whatever components
    survive are joined at ``inf``, a merge that costs infinite density, i.e. lambda 0."""
    n, k = indices.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k - 1)
    dst = np.ascontiguousarray(indices[:, 1:]).ravel()
    d = np.ascontiguousarray(dists[:, 1:]).ravel()
    weight = np.maximum(np.maximum(core[src], core[dst]), d)
    order = np.argsort(weight, kind="stable")
    # .tolist() once: iterating numpy scalars in the Kruskal loop below is several times slower.
    ws = weight[order].tolist()
    as_ = src[order].tolist()
    bs = dst[order].tolist()

    parent = list(range(n))

    mst: list[tuple[float, int, int]] = []
    for pos in range(len(ws)):
        ra, rb = _uf_find(parent, as_[pos]), _uf_find(parent, bs[pos])
        if ra == rb:
            continue
        parent[rb] = ra
        mst.append((ws[pos], as_[pos], bs[pos]))
        if len(mst) == n - 1:
            break
    if len(mst) < n - 1:  # disconnected kNN graph: bridge the components at infinite distance
        roots = [i for i in range(n) if _uf_find(parent, i) == i]
        for other in roots[1:]:
            parent[_uf_find(parent, other)] = _uf_find(parent, roots[0])
            mst.append((float("inf"), roots[0], other))
    return mst


def _single_linkage(
    mst: list[tuple[float, int, int]], n: int
) -> tuple[list[int], list[int], list[float], list[int]]:
    """MST edges (ascending) -> single-linkage tree. Node ``i < n`` is a point; node ``n + j`` is
    the merge made by edge ``j``, joining ``left[j]`` and ``right[j]`` at ``height[j]``."""
    parent = list(range(n))
    node_of = list(range(n))  # union-find root -> the tree node that root currently stands for
    size = [1] * (2 * n - 1)
    left = [0] * (n - 1)
    right = [0] * (n - 1)
    height = [0.0] * (n - 1)

    for j, (w, a, b) in enumerate(mst):
        ra, rb = _uf_find(parent, a), _uf_find(parent, b)
        na, nb = node_of[ra], node_of[rb]
        node = n + j
        left[j], right[j], height[j] = na, nb, w
        size[node] = size[na] + size[nb]
        parent[rb] = ra
        node_of[ra] = node
    return left, right, height, size


def _condense(
    left: list[int],
    right: list[int],
    height: list[float],
    size: list[int],
    n: int,
    min_cluster_size: int,
) -> list[tuple[int, int, float, int]]:
    """Walk the single-linkage tree down from the root, dropping every branch smaller than
    ``min_cluster_size``. Returns rows ``(cluster, child, lambda, child_size)``: ``child < n`` is a
    point leaving ``cluster`` at that lambda, ``child >= n`` is a sub-cluster born there. Cluster
    ids start at ``n`` (the root) and always exceed their parent's."""

    def subtree_points(node: int) -> list[int]:
        points: list[int] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur < n:
                points.append(cur)
            else:
                stack.append(left[cur - n])
                stack.append(right[cur - n])
        return points

    rows: list[tuple[int, int, float, int]] = []
    next_label = n + 1
    stack = [(2 * n - 2, n)]  # (tree node, the cluster id it belongs to)
    while stack:
        node, label = stack.pop()
        if node < n:
            continue
        j = node - n
        lam = 1.0 / height[j] if height[j] > 0.0 else float("inf")
        lc, rc = left[j], right[j]
        big_l, big_r = size[lc] >= min_cluster_size, size[rc] >= min_cluster_size
        if big_l and big_r:  # a real split: two new clusters born at lam
            for child in (lc, rc):
                rows.append((label, next_label, lam, size[child]))
                stack.append((child, next_label))
                next_label += 1
        elif not big_l and not big_r:  # the whole branch falls out; nothing below is a cluster
            for point in subtree_points(node):
                rows.append((label, point, lam, 1))
        else:  # one side is too small: it falls out, the other stays this same cluster
            small, big = (lc, rc) if not big_l else (rc, lc)
            for point in subtree_points(small):
                rows.append((label, point, lam, 1))
            stack.append((big, label))
    return rows


def _labels_from_condensed(rows: list[tuple[int, int, float, int]], n: int) -> IntArray:
    """Score every cluster in the condensed tree by stability and keep the best non-overlapping
    set (excess of mass), then label each point by the deepest kept cluster above it.

    Stability of a cluster is ``sum over its children of (lambda_leave - lambda_birth) * size``:
    how much density range it survives, weighted by how many points survive it. A cluster is kept
    only if it beats the total of its descendants; otherwise the descendants win. The root is
    never a cluster (it holds everything, so "one big cluster" is never an answer here)."""
    labels = np.full(n, -1, dtype=np.int64)
    if not rows:
        return labels
    root = n
    birth: dict[int, float] = {root: 0.0}
    children: dict[int, list[int]] = {}
    for cluster, child, lam, _ in rows:
        if child >= n:
            birth[child] = lam
            children.setdefault(cluster, []).append(child)
    stability: dict[int, float] = dict.fromkeys(birth, 0.0)
    for cluster, _, lam, count in rows:
        stability[cluster] += (lam - birth[cluster]) * count

    # Bottom-up: a child always has a larger id than its parent, so descending order is leaves-up.
    selected = {c for c in stability if c != root}
    for cluster in sorted(selected, reverse=True):
        below = sum(stability[c] for c in children.get(cluster, ()))
        if below > stability[cluster]:
            selected.discard(cluster)
            stability[cluster] = below  # the descendants' score propagates up in its place
        else:
            stack = list(children.get(cluster, ()))
            while stack:  # this cluster wins, so nothing under it can also be a cluster
                sub = stack.pop()
                selected.discard(sub)
                stack.extend(children.get(sub, ()))
    if not selected:
        return labels

    label_of = {c: i for i, c in enumerate(sorted(selected))}
    up = {child: cluster for cluster, child, _, _ in rows if child not in selected}
    for point in range(n):
        node = point
        while node in up:
            node = up[node]
        if node in label_of:
            labels[point] = label_of[node]
    return labels
