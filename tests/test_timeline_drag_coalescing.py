"""Headless regression tests for the shared video timeline and image canvas."""

import ast
from pathlib import Path
from types import SimpleNamespace

from AnimalTA.A_General_tools.Class_Scroll_crop import Pers_Scroll
from AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check import Lecteur as CheckLecteur


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeWidget:
    def __init__(self, width=200):
        self.width = width

    def winfo_exists(self):
        return 1

    def winfo_width(self):
        return self.width


def _make_scrollbar():
    scheduled = []
    cancelled = []
    updates = []
    refreshes = []

    scroll = object.__new__(Pers_Scroll)
    scroll.parent = _FakeWidget()
    scroll.Top = SimpleNamespace(closed=False, update_image=updates.append)
    scroll.winfo_exists = lambda: 1
    scroll.decalage = 25
    scroll.video_length = 100
    scroll.debut = 0
    scroll.fin = 100
    scroll.active_pos = 0
    scroll._closed = False
    scroll._drag_after_id = None
    scroll._pending_drag_pos = None
    scroll._drag_debounce_ms = 16
    scroll.refresh = lambda *args: refreshes.append(scroll.active_pos)
    scroll.delete = lambda *_args: None
    scroll.unbind = lambda *_args: None
    scroll.after = lambda _delay, callback: scheduled.append(callback) or "drag-1"
    scroll.after_cancel = lambda callback_id: cancelled.append(callback_id)
    return scroll, scheduled, cancelled, updates, refreshes


def test_drag_coalesces_motion_events_until_one_tk_callback():
    scroll, scheduled, _cancelled, updates, refreshes = _make_scrollbar()

    Pers_Scroll.move_position(scroll, SimpleNamespace(x=40))
    Pers_Scroll.move_position(scroll, SimpleNamespace(x=90))
    Pers_Scroll.move_position(scroll, SimpleNamespace(x=140))

    assert len(scheduled) == 1
    assert updates == []
    assert refreshes == []

    scheduled[0]()

    assert updates == [82]
    assert refreshes == [82]
    assert scroll._pending_drag_pos is None


def test_pending_drag_is_cancelled_and_ignored_after_reader_close():
    scroll, scheduled, cancelled, updates, refreshes = _make_scrollbar()

    Pers_Scroll.move_position(scroll, SimpleNamespace(x=140))
    Pers_Scroll.close_N_destroy(scroll)
    scheduled[0]()

    assert cancelled == ["drag-1"]
    assert updates == []
    assert refreshes == []


def test_release_flushes_latest_drag_without_waiting_for_timer():
    scroll, scheduled, cancelled, updates, refreshes = _make_scrollbar()

    Pers_Scroll.move_position(scroll, SimpleNamespace(x=40))
    Pers_Scroll.move_position(scroll, SimpleNamespace(x=140))
    Pers_Scroll._finish_drag(scroll)

    assert cancelled == ["drag-1"]
    assert updates == [82]
    assert refreshes == [82]


def test_reader_reuses_one_canvas_image_item_for_display_updates():
    path = REPO_ROOT / "AnimalTA/A_General_tools/Class_Lecteur.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lecteur = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Lecteur")
    afficher_img = next(node for node in lecteur.body if isinstance(node, ast.FunctionDef) and node.name == "afficher_img")

    create_image_calls = [
        node for node in ast.walk(afficher_img)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_image"
    ]
    assert len(create_image_calls) == 1
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "itemconfig"
        for node in ast.walk(afficher_img)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "coords"
        for node in ast.walk(afficher_img)
    )


def test_verification_table_scale_coalesces_motion_and_flushes_on_release():
    scheduled = []
    cancelled = []
    updates = []
    refreshes = []

    reader = object.__new__(CheckLecteur)
    reader.vsb = SimpleNamespace(get=lambda: 12)
    reader.Scrollbar = SimpleNamespace(
        active_pos=None,
        refresh=lambda: refreshes.append(reader.Scrollbar.active_pos),
    )
    reader.Vid_Lecteur = SimpleNamespace(update_image=updates.append)
    reader._table_scroll_after_id = None
    reader._pending_table_pos = None
    reader._table_scroll_debounce_ms = 16
    reader.after = lambda _delay, callback: scheduled.append(callback) or "table-1"
    reader.after_cancel = lambda callback_id: cancelled.append(callback_id)

    CheckLecteur.move_tree(reader, SimpleNamespace())
    reader.vsb.get = lambda: 42
    CheckLecteur.move_tree(reader, SimpleNamespace())

    assert len(scheduled) == 1
    assert updates == []
    assert refreshes == []

    CheckLecteur.finish_tree_scroll(reader)

    assert cancelled == ["table-1"]
    assert reader.Scrollbar.active_pos == 42
    assert refreshes == [42]
    assert updates == [42]
