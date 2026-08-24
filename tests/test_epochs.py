"""make_epochs_per_sample: the -1 sentinel contract (GitLab #7)."""

import warnings

import numpy as np

from fastumap.graph import make_epochs_per_sample


def test_denormal_weight_returns_sentinel_not_inf() -> None:
    # 1e-320 is denormal; its reciprocal overflows. The edge must come back as the documented
    # -1 sentinel, never inf, so the optimiser's `> 0.0` mask skips it as intended.
    out = make_epochs_per_sample(np.array([1.0, 1e-320]), 500)
    assert np.isfinite(out).all()
    assert out[1] == -1.0
    assert out[0] == 1.0  # the max-weight edge fires every epoch


def test_no_warning_from_library_code() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        make_epochs_per_sample(np.array([1.0, 1e-320]), 500)


def test_no_raise_under_seterr_raise() -> None:
    # A numeric service may run with np.seterr(all="raise"); library code must not trip it.
    old = np.seterr(all="raise")
    try:
        out = make_epochs_per_sample(np.array([1.0, 1e-320]), 500)
    finally:
        np.seterr(**old)
    assert out[1] == -1.0
