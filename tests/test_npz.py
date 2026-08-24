"""Versioned save/load for UMAPModel (GitLab #11)."""

import io
from pathlib import Path

import numpy as np
import pytest

from fastumap import fit, transform
from fastumap.projection import UMAPModel


def _fit(pca: bool) -> UMAPModel:
    x = np.random.default_rng(0).standard_normal((300, 32))
    return fit(x, 2, random_state=42, pca_dim=16 if pca else None)


@pytest.mark.parametrize("pca", [False, True])
def test_npz_round_trip_matches(pca: bool) -> None:
    model = _fit(pca)
    buf = io.BytesIO()
    model.to_npz(buf)
    buf.seek(0)
    restored = UMAPModel.from_npz(buf)
    new = np.random.default_rng(1).standard_normal((10, 32))
    assert np.array_equal(transform(model, new), transform(restored, new))
    assert restored.train.dtype == np.float32
    assert restored.metric == model.metric
    assert restored.n_neighbors == model.n_neighbors
    assert (restored.pca_mean is not None) == pca


def test_npz_file_path(tmp_path: Path) -> None:
    model = _fit(False)
    path = tmp_path / "model.npz"
    model.to_npz(path)
    restored = UMAPModel.from_npz(path)
    assert restored.dimensions == 2 and restored.n_features == 32


def test_npz_version_mismatch_names_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fit(False)
    path = tmp_path / "m.npz"
    model.to_npz(path)  # written as format 1
    import fastumap.projection as P

    monkeypatch.setattr(P, "_MODEL_FORMAT", 2)  # a newer reader
    with pytest.raises(ValueError, match=r"format 1.*format 2"):
        UMAPModel.from_npz(path)
