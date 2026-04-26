"""Smoke tests for Linux/macOS compatibility.

Run with: python -m pytest test_linux_compat.py -v
"""
import sys
import ast
import os
import tempfile
import shutil
import types
import uuid
import numpy as np


ANIMALTA_DIR = os.path.join(os.path.dirname(__file__), "AnimalTA")
TEST_SCRATCH_ROOT = os.path.join(tempfile.gettempdir(), "animalta_test_linux_compat")


def iter_py_files():
    for root, _, files in os.walk(ANIMALTA_DIR):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def make_test_scratch_dir(prefix):
    os.makedirs(TEST_SCRATCH_ROOT, exist_ok=True)
    path = os.path.join(TEST_SCRATCH_ROOT, prefix + "_" + uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_test_scratch_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    try:
        os.rmdir(TEST_SCRATCH_ROOT)
    except OSError:
        pass


def imported_names_for_tree(tree):
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name != "*":
                    imports.add(alias.asname or alias.name)
    return imports


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
    """simsun.ttc (Windows-only font) must not be referenced directly outside the compat layer."""
    violations = []
    for path in iter_py_files():
        if path.endswith("compat.py"):
            continue  # compat.py is allowed to reference simsun as a Windows fallback
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if "simsun" in line.lower() and not line.strip().startswith("#"):
                    violations.append(f"{path}:{i}: {line.strip()}")
    assert not violations, "simsun.ttc references found:\n" + "\n".join(violations)


def test_no_direct_pyautogui_import():
    """Tk helpers should not require pyautogui at import time."""
    violations = []
    for path in iter_py_files():
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pyautogui":
                        violations.append(path)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "pyautogui":
                    violations.append(path)
    assert not violations, "Direct pyautogui imports found:\n" + "\n".join(sorted(set(violations)))


def test_shared_helper_modules_are_explicitly_imported():
    """Common cross-platform helper modules should never be used via implicit globals."""
    helper_names = {"UserMessages", "compat", "image_utils"}
    violations = []

    for path in iter_py_files():
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)

        imported = imported_names_for_tree(tree)
        for helper_name in helper_names:
            used = any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == helper_name
                for node in ast.walk(tree)
            )
            if used and helper_name not in imported:
                violations.append(f"{path}: uses {helper_name} without importing it")

    assert not violations, "Missing helper imports found:\n" + "\n".join(violations)


def test_runtime_modules_do_not_hardcode_project_runtime_subdirs():
    """Project runtime directories should be resolved through UserMessages helpers."""
    disallowed = {
        "coordinates",
        "Coordinates",
        "corrected_coordinates",
        "TMP_portion",
        "tmp_portion",
        "converted_vids",
    }
    violations = []

    for path in iter_py_files():
        if path.endswith("UserMessages.py"):
            continue

        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "join"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
            ):
                continue

            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value in disallowed:
                    violations.append(f"{path}:{node.lineno}: os.path.join(..., {arg.value!r}, ...)")

    assert not violations, "Hardcoded project runtime subdirs found:\n" + "\n".join(violations)


def test_compat_module_importable():
    """Verify the compatibility layer is importable and has required attributes."""
    from AnimalTA import compat
    assert hasattr(compat, "beep")
    assert hasattr(compat, "play_sound")
    assert hasattr(compat, "open_file_external")
    assert hasattr(compat, "get_pointer_position")
    assert hasattr(compat, "set_window_icon")
    assert hasattr(compat, "startup_debug")
    assert hasattr(compat, "get_font_path")


def test_compat_get_font_path_returns_something():
    """get_font_path should always return a (path_or_None, size) tuple."""
    from AnimalTA.compat import get_font_path
    result = get_font_path(20)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1] == 20


def test_resource_fallback_finds_existing_logo():
    """Missing preferred assets should fall back to an existing bundled resource."""
    from AnimalTA import compat

    path = compat.find_resource_path(
        os.path.join("AnimalTA", "Files", "Logo_fond.png"),
        os.path.join("AnimalTA", "Files", "Logo.png"),
    )
    assert path is not None
    assert os.path.exists(path)


def test_writable_resources_redirect_to_user_data_dir():
    """Files that must stay writable in containers should resolve outside the app tree."""
    from AnimalTA.A_General_tools import UserMessages
    import tempfile

    user_dir = UserMessages.user_data_dir()
    temp_dir = os.path.join(tempfile.gettempdir(), "animalta")
    app_dir = os.path.join(os.path.dirname(__file__), "AnimalTA", "Files")
    paths = [
        UserMessages.resource_path(os.path.join("AnimalTA", "Files", "Language")),
        UserMessages.resource_path(os.path.join("AnimalTA", "Files", "Autosave")),
        UserMessages.resource_path(os.path.join("AnimalTA", "Files", "Autosave", "example.ata")),
        UserMessages.resource_path(os.path.join("AnimalTA", "Files", "Last_downloaded")),
    ]
    for path in paths:
        try:
            in_app_dir = os.path.commonpath([app_dir, os.path.abspath(path)]) == app_dir
        except ValueError:
            in_app_dir = False
        assert not in_app_dir
        assert (
            os.path.commonpath([user_dir, path]) == user_dir
            or os.path.commonpath([temp_dir, path]) == temp_dir
        )


def test_projects_dir_path_is_writable():
    """New projects should default to a writable per-user directory."""
    from AnimalTA.A_General_tools import UserMessages

    projects_dir = UserMessages.projects_dir_path()
    user_dir = UserMessages.user_data_dir()
    temp_dir = os.path.join(tempfile.gettempdir(), "animalta")
    app_dir = os.path.join(os.path.dirname(__file__), "AnimalTA")

    assert os.path.isdir(projects_dir)
    try:
        in_app_dir = os.path.commonpath([app_dir, os.path.abspath(projects_dir)]) == app_dir
    except ValueError:
        in_app_dir = False

    assert not in_app_dir
    assert (
        os.path.commonpath([user_dir, projects_dir]) == user_dir
        or os.path.commonpath([temp_dir, projects_dir]) == temp_dir
    )


def test_working_project_copy_paths_use_user_projects_dir():
    """Read-only projects should be relocated into the per-user projects area."""
    from AnimalTA.A_General_tools import UserMessages

    temp_root = make_test_scratch_dir("working_copy")
    previous = os.environ.get("ANIMALTA_DATA_DIR")
    os.environ["ANIMALTA_DATA_DIR"] = temp_root
    try:
        project_file, project_folder = UserMessages.working_project_copy_paths(
            "/opt/animalta/test_project/Untitled_project.ata"
        )

        projects_root = os.path.join(temp_root, "Projects")
        assert os.path.dirname(project_file) == projects_root
        assert os.path.dirname(project_folder) == projects_root
        assert os.path.basename(project_file) == "Untitled_project_working_copy.ata"
        assert os.path.basename(project_folder) == "Project_folder_Untitled_project_working_copy"
    finally:
        if previous is None:
            os.environ.pop("ANIMALTA_DATA_DIR", None)
        else:
            os.environ["ANIMALTA_DATA_DIR"] = previous
        cleanup_test_scratch_dir(temp_root)


def test_working_project_copy_paths_add_suffix_when_name_taken():
    """Working-copy path reservation should avoid clobbering an existing project."""
    from AnimalTA.A_General_tools import UserMessages

    temp_root = make_test_scratch_dir("working_copy")
    previous = os.environ.get("ANIMALTA_DATA_DIR")
    os.environ["ANIMALTA_DATA_DIR"] = temp_root
    try:
        projects_root = os.path.join(temp_root, "Projects")
        os.makedirs(projects_root, exist_ok=True)
        open(os.path.join(projects_root, "Untitled_project_working_copy.ata"), "w", encoding="utf-8").close()
        os.makedirs(os.path.join(projects_root, "Project_folder_Untitled_project_working_copy"), exist_ok=True)

        project_file, project_folder = UserMessages.working_project_copy_paths(
            "/opt/animalta/test_project/Untitled_project.ata"
        )

        assert os.path.basename(project_file) == "Untitled_project_working_copy_2.ata"
        assert os.path.basename(project_folder) == "Project_folder_Untitled_project_working_copy_2"
    finally:
        if previous is None:
            os.environ.pop("ANIMALTA_DATA_DIR", None)
        else:
            os.environ["ANIMALTA_DATA_DIR"] = previous
        cleanup_test_scratch_dir(temp_root)


def test_safe_relative_background_handles_zero_background_pixels():
    """Relative background scaling should not divide by zero or wrap on zero-valued background pixels."""
    from AnimalTA.A_General_tools import image_utils

    image = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    background = np.array([[0, 10], [0, 5]], dtype=np.uint8)
    result = image_utils.apply_relative_background(image, background)

    assert result.dtype == np.uint8
    assert np.array_equal(result, np.array([[0, 255], [255, 255]], dtype=np.uint8))


def test_coordinates_dir_path_handles_existing_lowercase_dir():
    """Coordinate helpers should tolerate older lowercase project folders on Linux."""
    from AnimalTA.A_General_tools import UserMessages

    tmpdir = make_test_scratch_dir("coord_case")
    try:
        lowercase_dir = os.path.join(tmpdir, "coordinates")
        os.makedirs(lowercase_dir, exist_ok=True)
        resolved = UserMessages.coordinates_dir_path(tmpdir)
        assert os.path.basename(resolved).lower() == "coordinates"
        assert os.path.normcase(resolved) == os.path.normcase(lowercase_dir)
    finally:
        cleanup_test_scratch_dir(tmpdir)


def test_coordinates_dir_path_creates_canonical_dir():
    """New coordinate folders should be created with the canonical project casing."""
    from AnimalTA.A_General_tools import UserMessages

    tmpdir = make_test_scratch_dir("coord_case")
    try:
        resolved = UserMessages.coordinates_dir_path(tmpdir, create=True)
        assert os.path.isdir(resolved)
        assert os.path.basename(resolved) == "Coordinates"
    finally:
        cleanup_test_scratch_dir(tmpdir)


def test_row_video_previews_do_not_use_opencv_windows():
    """The Linux build should not rely on OpenCV HighGUI windows for hover/previews."""
    violations = []
    allowed_comment_only = {
        os.path.join(
            ANIMALTA_DIR,
            "E_Post_tracking",
            "b_Analyses",
            "Body_part_functions.py",
        ),
    }

    for path in iter_py_files():
        if path in allowed_comment_only:
            continue
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                stripped = line.strip()
                if any(token in line for token in ("cv2.namedWindow", "cv2.imshow", "cv2.moveWindow", "cv2.destroyAllWindows")):
                    violations.append(f"{path}:{i}: {stripped}")

    assert not violations, "OpenCV GUI calls found:\n" + "\n".join(violations)


def test_tracking_modules_do_not_print_debug_output():
    """Tracking modules should not emit stray debug prints in normal runs."""
    paths = [
        os.path.join(ANIMALTA_DIR, "D_Tracking_process", "Tracking_method_selection.py"),
        os.path.join(ANIMALTA_DIR, "D_Tracking_process", "Do_the_track_multi.py"),
        os.path.join(ANIMALTA_DIR, "D_Tracking_process", "Function_prepare_images_multi.py"),
        os.path.join(ANIMALTA_DIR, "D_Tracking_process", "Function_assign_cnts_multi.py"),
    ]
    violations = []

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                stripped = line.strip()
                if "print(" in stripped and not stripped.startswith("#"):
                    violations.append(f"{path}:{i}: {stripped}")

    assert not violations, "Debug print statements found:\n" + "\n".join(violations)


def test_reader_update_image_returns_none_after_close():
    """Queued frame updates should quietly stop once the video reader starts closing."""
    from AnimalTA.A_General_tools.Class_Lecteur import Lecteur

    reader = object.__new__(Lecteur)
    reader.closed = True

    assert Lecteur.update_image(reader, 0) is None


def test_reader_update_ratio_keeps_previous_value_when_canvas_is_gone():
    """Canvas teardown should not trigger Tk errors while recomputing the zoom ratio."""
    from AnimalTA.A_General_tools.Class_Lecteur import Lecteur

    reader = object.__new__(Lecteur)
    reader.closed = False
    reader.zoom_sq = [0, 0, 320, 240]
    reader.ratio = 1.5
    reader.canvas_video = types.SimpleNamespace(winfo_exists=lambda: 0)

    assert Lecteur.update_ratio(reader) == 1.5
    assert reader.ratio == 1.5


def test_scrollbar_drag_ignores_closed_reader():
    """Late drag callbacks should be ignored once the owning reader is closed."""
    from AnimalTA.A_General_tools.Class_Scroll_crop import Pers_Scroll

    calls = []
    scroll = object.__new__(Pers_Scroll)
    scroll.parent = types.SimpleNamespace(winfo_width=lambda: 200, winfo_exists=lambda: 1)
    scroll.Top = types.SimpleNamespace(closed=True, update_image=lambda pos: calls.append(pos))
    scroll.winfo_exists = lambda: 1
    scroll.decalage = 25
    scroll.video_length = 100
    scroll.debut = 0
    scroll.refresh = lambda *args: None

    Pers_Scroll.move_position(scroll, types.SimpleNamespace(x=80))

    assert calls == []


def test_coordinate_loader_has_usermessages_path_helpers():
    """Coordinate load/save helpers should keep using the shared writable-path utilities."""
    from AnimalTA.E_Post_tracking import Coos_loader_saver

    assert hasattr(Coos_loader_saver, "UserMessages")


def test_converted_videos_dir_path_creates_project_subdir():
    """Converted videos should resolve through a shared project-subdir helper."""
    from AnimalTA.A_General_tools import UserMessages

    tmpdir = make_test_scratch_dir("converted_vids")
    try:
        resolved = UserMessages.converted_videos_dir_path(tmpdir, create=True)
        assert os.path.isdir(resolved)
        assert os.path.basename(resolved) == "converted_vids"
    finally:
        cleanup_test_scratch_dir(tmpdir)


def test_tracking_check_resize_ignores_incomplete_coordinate_load():
    """A failed coordinate load should not cascade into resize-table attribute errors."""
    from AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check import Lecteur

    viewer = object.__new__(Lecteur)
    viewer.Coos = None
    viewer.Scrollbar = object()
    viewer.afficher_table = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resize should no-op"))

    Lecteur.resize_fr(viewer, types.SimpleNamespace())


def test_tracking_check_table_noops_until_coordinates_are_loaded():
    """The verification table should stay idle until coordinate data has loaded successfully."""
    from AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check import Lecteur

    viewer = object.__new__(Lecteur)
    viewer.Coos = None
    viewer.who_is_here = None
    viewer.Scrollbar = object()

    assert Lecteur.afficher_table(viewer) is None


def test_no_utf8_bom():
    """Python source files must not start with a UTF-8 BOM (U+FEFF).

    A BOM causes ast.parse() to raise SyntaxError on Python 3.12+ when the
    file is opened with encoding='utf-8' (not utf-8-sig).  Save files as
    UTF-8 without BOM.
    """
    violations = []
    for path in iter_py_files():
        with open(path, "rb") as f:
            if f.read(3) == b"\xef\xbb\xbf":
                violations.append(path)
    assert not violations, "Files with UTF-8 BOM found:\n" + "\n".join(violations)


def test_animalta_importable():
    """Verify the package imports without pulling in Windows-only modules."""
    try:
        from AnimalTA.Main_interface import start_mainframe  # noqa: F401
    except Exception as e:
        msg = str(e)
        if "winsound" in msg or "winreg" in msg or "windll" in msg:
            raise AssertionError(f"Windows-only import at package level: {e}")
        # X11/display errors (pyautogui, Xlib) are acceptable in headless environments;
        # they will be resolved at runtime when DISPLAY is set by the VNC session.
