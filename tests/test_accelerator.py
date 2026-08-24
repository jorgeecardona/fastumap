"""fastumap.accelerator_active() introspection (GitLab #9)."""

import importlib

import fastumap


def test_returns_bool_matching_install() -> None:
    active = fastumap.accelerator_active()
    assert isinstance(active, bool)
    # It must agree with whether the compiled kernel actually imports — the same dynamic
    # import layout.py uses to pick the SGD path.
    try:
        importlib.import_module("fastumap_accel")
        imported = True
    except ImportError:
        imported = False
    assert active == imported
