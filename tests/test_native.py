"""A missing compiled kernel raises with install instructions, never a silent numpy fallback."""

from __future__ import annotations

import importlib

import pytest

from fastumap._native import load_accel


def test_missing_extension_raises_with_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> object:
        if name == "fastumap._accel":
            raise ImportError("no module")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(ImportError, match="uv sync"):
        load_accel()


def test_present_extension_loads() -> None:
    assert hasattr(load_accel(), "kmeans")
