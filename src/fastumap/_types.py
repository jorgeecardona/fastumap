"""Shared array type aliases.

Keeping these in one place makes the numpy dtype contract explicit at every
boundary: fastumap works in float64 for coordinates and int64 for indices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
