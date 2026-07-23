"""Packaging and CLI configuration smoke tests."""

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_backend_uses_supported_setuptools_backend():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["build-system"]["build-backend"] == "setuptools.build_meta"


def test_console_script_uses_shared_main_entrypoint():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["scripts"]["animalta"] == "main:main"
