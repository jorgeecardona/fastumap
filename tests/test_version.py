"""Guard against the __version__ drift of GitLab issue #1.

pyproject is the single source of truth; __version__ derives from the installed
distribution metadata. These assert the whole chain agrees, so a bump can't ship a
module attribute that still reports the old version.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import tomllib

import fastumap


def test_version_matches_distribution_metadata() -> None:
    assert fastumap.__version__ == importlib.metadata.version("fastumap")


def test_version_matches_pyproject() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert fastumap.__version__ == pyproject["project"]["version"], (
        "fastumap.__version__ (from installed metadata) is out of sync with pyproject; "
        "run `uv sync` after a version bump so the metadata is rebuilt"
    )
