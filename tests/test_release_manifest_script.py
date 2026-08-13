"""Regression tests for the HACS release-manifest updater."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "update_hacs_manifest.py"
)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v6.2.2-powmr.10-dev.2", "6.2.2-powmr.10-dev.2"),
        ("6.2.2-powmr.10-dev.2", "6.2.2-powmr.10-dev.2"),
    ],
)
def test_updater_removes_only_the_leading_tag_prefix(
    tmp_path: Path, tag: str, expected: str
) -> None:
    """The letter v inside a prerelease identifier must be preserved."""
    component = tmp_path / "component"
    component.mkdir()
    manifest_path = component / "manifest.json"
    manifest_path.write_text(
        json.dumps({"domain": "hsem", "name": "HSEM PowMr", "version": "old"}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--version",
            tag,
            "--path",
            "/component/",
        ],
        cwd=tmp_path,
        check=True,
    )

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["version"] == expected
