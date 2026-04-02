"""Smoke tests for Linux/macOS compatibility.

Run with: python -m pytest test_linux_compat.py -v
"""
import sys
import ast
import os


ANIMALTA_DIR = os.path.join(os.path.dirname(__file__), "AnimalTA")


def iter_py_files():
    for root, _, files in os.walk(ANIMALTA_DIR):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def test_no_winsound_direct_import():
    """Ensure no module directly imports winsound outside compat.py."""
    for path in iter_py_files():
        if path.endswith("compat.py"):
            continue
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "winsound", f"{path} directly imports winsound"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "winsound", f"{path} imports from winsound"


def test_no_django_import():
    """Ensure django has been removed from all source files."""
    for path in iter_py_files():
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("django"), (
                        f"{path} imports django (should have been removed)"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("django"), (
                    f"{path} imports from django (should have been removed)"
                )


def test_no_unconditional_windll_import():
    """windll must only be imported inside a sys.platform == 'win32' guard."""
    violations = []
    for path in iter_py_files():
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "windll" in stripped and "import" in stripped:
                # Check the previous non-blank line for a platform guard
                context = "".join(lines[max(0, i - 3):i])
                if "win32" not in context:
                    violations.append(f"{path}:{i}: {stripped}")
    assert not violations, "Unconditional windll imports found:\n" + "\n".join(violations)


def test_no_hardcoded_windows_paths():
    """Check for hardcoded Windows drive paths."""
    violations = []
    for path in iter_py_files():
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "C:\\" in line or "D:\\" in line:
                    violations.append(f"{path}:{i}: {stripped}")
    assert not violations, "Hardcoded Windows paths found:\n" + "\n".join(violations)


def test_no_simsun_font():
    """simsun.ttc (Windows-only font) must not be referenced directly."""
    violations = []
    for path in iter_py_files():
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if "simsun" in line.lower() and not line.strip().startswith("#"):
                    violations.append(f"{path}:{i}: {line.strip()}")
    assert not violations, "simsun.ttc references found:\n" + "\n".join(violations)


def test_compat_module_importable():
    """Verify the compatibility layer is importable and has required attributes."""
    from AnimalTA import compat
    assert hasattr(compat, "beep")
    assert hasattr(compat, "play_sound")
    assert hasattr(compat, "open_file_external")
    assert hasattr(compat, "get_font_path")


def test_compat_get_font_path_returns_something():
    """get_font_path should always return a (path_or_None, size) tuple."""
    from AnimalTA.compat import get_font_path
    result = get_font_path(20)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1] == 20


def test_animalta_importable():
    """Verify the package imports without pulling in Windows-only modules."""
    try:
        from AnimalTA.Main_interface import start_mainframe  # noqa: F401
    except ImportError as e:
        if "winsound" in str(e) or "winreg" in str(e) or "windll" in str(e):
            raise AssertionError(f"Windows-only import at package level: {e}")
        # Other ImportErrors (e.g. no display) are acceptable in headless CI
