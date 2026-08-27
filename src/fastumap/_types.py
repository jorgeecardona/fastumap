"""Shared array type aliases.

Keeping these in one place makes the numpy dtype contract explicit at every
boundary: fastumap works in float64 for coordinates and int64 for indices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
# float32 is used only for the cached UMAPModel.train, to halve its footprint; transform
# upcasts to FloatArray for the actual distance compute.
Float32Array = NDArray[np.float32]


def unit_rows(x: FloatArray) -> FloatArray:
    """Rows scaled to unit L2 norm; a zero row is left unchanged (its norm is treated as 1). Used
    by the cosine metric, where squared-L2 on unit rows equals 2*(1 - cosine similarity)."""
    norms = np.sqrt(np.einsum("ij,ij->i", x, x))
    return x / np.where(norms > 0.0, norms, 1.0)[:, None]
