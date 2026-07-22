"""Focused tests for project-list scrolling and tracking completion cleanup."""

import ast
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_method(relative_path, class_name, method_name, globals_=None):
    """Load one method without importing the GUI's optional video dependencies."""
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.Module(body=[method_node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


class _Scale:
    def __init__(self, value=0):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Flag:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_project_list_scrolls_with_x11_wheel_buttons():
    on_mousewheel = _load_method(
        "AnimalTA/B_Project_organisation/Interface_pretracking.py",
        "Interface",
        "on_mousewheel",
    )
    display_calls = []
    interface = SimpleNamespace(vsv=_Scale(3), afficher_projects=lambda: display_calls.append(True))

    on_mousewheel(interface, SimpleNamespace(num=4, delta=0))
    assert interface.vsv.get() == 2

    on_mousewheel(interface, SimpleNamespace(num=5, delta=0))
    assert interface.vsv.get() == 3
    assert len(display_calls) == 2


def test_project_list_scrolls_with_mousewheel_delta():
    on_mousewheel = _load_method(
        "AnimalTA/B_Project_organisation/Interface_pretracking.py",
        "Interface",
        "on_mousewheel",
    )
    interface = SimpleNamespace(vsv=_Scale(3), afficher_projects=lambda: None)

    on_mousewheel(interface, SimpleNamespace(num=None, delta=120))
    assert interface.vsv.get() == 2

    on_mousewheel(interface, SimpleNamespace(num=None, delta=-120))
    assert interface.vsv.get() == 3


def test_project_list_registers_and_removes_all_wheel_events():
    bind_mousewheel = _load_method(
        "AnimalTA/B_Project_organisation/Interface_pretracking.py",
        "Interface",
        "bind_mousewheel",
    )
    unbind_mousewheel = _load_method(
        "AnimalTA/B_Project_organisation/Interface_pretracking.py",
        "Interface",
        "unbind_mousewheel",
    )
    bound = []
    unbound = []
    interface = SimpleNamespace(
        bind_all=lambda sequence, callback: bound.append((sequence, callback)),
        unbind_all=lambda sequence: unbound.append(sequence),
        on_mousewheel=lambda _event: None,
    )

    bind_mousewheel(interface)
    unbind_mousewheel(interface)

    expected = {"<MouseWheel>", "<Button-4>", "<Button-5>"}
    assert {sequence for sequence, _callback in bound} == expected
    assert set(unbound) == expected


def test_tracking_completion_closes_after_alert():
    calls = []
    tracking_done = _load_method(
        "AnimalTA/A_General_tools/Interface_selection_track_and_analyses.py",
        "Extend",
        "_tracking_done",
        {
            "time": SimpleNamespace(time=lambda: 10),
            "pymsgbox": SimpleNamespace(
                alert=lambda message, title: calls.append(("alert", message, title))
            ),
        },
    )
    tracker = SimpleNamespace(
        manual_track=_Flag(False),
        urgent_close=False,
        Params={"Sound_alert_track": False, "Pop_alert_track": True},
        Messages={"Do_track3": "Finished in {} sec", "Do_track4": "Finished"},
        cancel=lambda: calls.append(("cancel",)),
    )

    tracking_done(tracker, 0)

    assert calls[0][0] == "alert"
    assert calls[1] == ("cancel",)


def test_tracking_completion_closes_when_alert_fails():
    def broken_alert(*_args):
        raise RuntimeError("notification failed")

    tracking_done = _load_method(
        "AnimalTA/A_General_tools/Interface_selection_track_and_analyses.py",
        "Extend",
        "_tracking_done",
        {
            "time": SimpleNamespace(time=lambda: 10),
            "pymsgbox": SimpleNamespace(alert=broken_alert),
        },
    )
    calls = []
    tracker = SimpleNamespace(
        manual_track=_Flag(False),
        urgent_close=False,
        Params={"Sound_alert_track": False, "Pop_alert_track": True},
        Messages={"Do_track3": "Finished in {} sec", "Do_track4": "Finished"},
        cancel=lambda: calls.append("cancel"),
    )

    tracking_done(tracker, 0)

    assert calls == ["cancel"]


def test_background_tracking_captures_tk_state_before_worker_starts():
    """Worker lambdas must not read Tk variables or update Tk widgets."""
    path = REPO_ROOT / "AnimalTA/A_General_tools/Interface_selection_track_and_analyses.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Choose_method"
    ]

    assert len(calls) == 2
    for call in calls:
        update_ui = next(keyword.value for keyword in call.keywords if keyword.arg == "update_ui")
        assert isinstance(update_ui, ast.Constant) and update_ui.value is False
        progress = next(keyword.value for keyword in call.keywords if keyword.arg == "progress")
        assert isinstance(progress, ast.Name) and progress.id == "_progress"

    worker_lambdas = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Lambda)
        and any(call in set(ast.walk(node)) for call in calls)
    ]
    assert len(worker_lambdas) == 2
    assert all(
        not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        )
        for worker in worker_lambdas
        for node in ast.walk(worker)
    )


def test_tracking_poll_does_not_enter_a_nested_tk_event_loop():
    path = REPO_ROOT / "AnimalTA/A_General_tools/Interface_selection_track_and_analyses.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    extend = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Extend"
    )
    method = next(
        node for node in extend.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_tracking_threaded"
    )
    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "show_load"
    ]

    assert len(calls) == 1
    process_events = next(
        keyword.value for keyword in calls[0].keywords
        if keyword.arg == "process_events"
    )
    assert isinstance(process_events, ast.Constant)
    assert process_events.value is False
