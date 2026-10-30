"""Unit tests for agrobot_perception.utils.repo_root."""

from pathlib import Path

import pytest

from agrobot_perception.utils.repo_root import find_repo_root


# Check that a valid setting is returned as the repository root
def test_returns_root_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGROBOT_ROOT", str(tmp_path))
    assert find_repo_root() == tmp_path


# Check that a missing setting fails instead of guessing a location
def test_fails_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGROBOT_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="AGROBOT_ROOT is not set"):
        find_repo_root()


# Check that a stale setting fails before any file lookup
def test_fails_when_not_a_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGROBOT_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not a directory"):
        find_repo_root()
