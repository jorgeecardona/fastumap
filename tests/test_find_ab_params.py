"""find_ab_params must reproduce UMAP's canonical output-kernel constants."""

from __future__ import annotations

from fastumap.graph import find_ab_params


def test_default_params_match_umap() -> None:
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    assert abs(a - 1.5769) < 1e-3
    assert abs(b - 0.8951) < 1e-3


def test_larger_min_dist_lowers_a() -> None:
    a_tight, _ = find_ab_params(spread=1.0, min_dist=0.1)
    a_loose, _ = find_ab_params(spread=1.0, min_dist=0.5)
    assert a_loose < a_tight
