"""Focused tests for tracking orchestration and UI progress isolation."""

import inspect
import threading
from types import SimpleNamespace

import pytest

from AnimalTA.D_Tracking_process import Tracking_method_selection


def _video(back_mode=0):
    return SimpleNamespace(
        Frame_rate=[30, 30],
        Cropped=[True, [0, 2999]],
        Back=[back_mode],
        Fusion=[[0, "video.avi"]],
        User_Name="video",
    )


def _parent():
    return SimpleNamespace(timer=None, show_load=lambda: None)


@pytest.mark.parametrize("update_ui, expected_updates", [(True, 1), (False, 0)])
def test_choose_method_respects_ui_update_policy(monkeypatch, update_ui, expected_updates):
    updates = []
    parent = SimpleNamespace(timer=None, show_load=lambda: updates.append(True))
    tracker_calls = []

    def fake_tracking(**kwargs):
        tracker_calls.append(kwargs)
        return "tracked"

    monkeypatch.setattr(Tracking_method_selection.Do_the_track_multi, "Do_tracking", fake_tracking)

    result = Tracking_method_selection.Choose_method(
        parent, _video(), folder="project", type="fixed", head_tail=False, update_ui=update_ui,
    )

    assert result == "tracked"
    assert len(updates) == expected_updates
    assert tracker_calls[0]["update_ui"] is update_ui


def test_non_mog2_always_uses_parallel(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track_multi, "Do_tracking",
        lambda **_: calls.append("parallel") or True,
    )
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track, "Do_tracking",
        lambda **_: (_ for _ in ()).throw(AssertionError("single-process called")),
    )

    Tracking_method_selection.Choose_method(_parent(), _video(back_mode=0), "project", "fixed", False)
    assert calls == ["parallel"]


def test_mog2_uses_single_process(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track, "Do_tracking",
        lambda **_: calls.append("single") or True,
    )
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track_multi, "Do_tracking",
        lambda **_: (_ for _ in ()).throw(AssertionError("parallel called for MOG2")),
    )

    Tracking_method_selection.Choose_method(_parent(), _video(back_mode=2), "project", "fixed", False)
    assert calls == ["single"]


@pytest.mark.parametrize("cpus", [1, 2, 3, 4, 5, 8, 16, 64])
def test_cpu_count_never_triggers_single_process(monkeypatch, cpus):
    monkeypatch.setattr(Tracking_method_selection, "_allocated_cpus", lambda: cpus)
    calls = []
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track_multi, "Do_tracking",
        lambda **_: calls.append("parallel") or True,
    )
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track, "Do_tracking",
        lambda **_: (_ for _ in ()).throw(AssertionError("single-process called")),
    )

    Tracking_method_selection.Choose_method(_parent(), _video(), "project", "fixed", False)
    assert calls == ["parallel"]


def test_parallel_exception_is_reraised(monkeypatch):
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track_multi, "Do_tracking",
        lambda **_: (_ for _ in ()).throw(RuntimeError("worker died")),
    )
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track, "Do_tracking",
        lambda **_: (_ for _ in ()).throw(AssertionError("fallback must not start")),
    )

    with pytest.raises(RuntimeError, match="worker died"):
        Tracking_method_selection.Choose_method(_parent(), _video(), "project", "fixed", False)


def test_direct_tracking_entrypoints_keep_ui_updates_enabled_by_default():
    single_default = inspect.signature(
        Tracking_method_selection.Do_the_track.Do_tracking
    ).parameters["update_ui"].default
    multi_default = inspect.signature(
        Tracking_method_selection.Do_the_track_multi.Do_tracking
    ).parameters["update_ui"].default

    assert single_default is True
    assert multi_default is True


def test_background_progress_never_mutates_the_tk_parent(monkeypatch):
    class MainThreadOnlyParent:
        def __setattr__(self, name, value):
            if name == "timer":
                raise AssertionError("background tracking touched its Tk parent")
            super().__setattr__(name, value)

        def show_load(self):
            raise AssertionError("background tracking updated Tk")

    progress = Tracking_method_selection.TrackingProgress()

    def fake_tracking(**kwargs):
        assert kwargs["progress"] is progress
        progress.set(0.5)
        return "tracked"

    monkeypatch.setattr(Tracking_method_selection.Do_the_track_multi, "Do_tracking", fake_tracking)

    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(
            Tracking_method_selection.Choose_method(
                MainThreadOnlyParent(), _video(), folder="project", type="fixed",
                head_tail=False, update_ui=False, progress=progress,
            )
        )
    )
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result_box == ["tracked"]
    assert progress.get() == 0.5


def test_parallel_tracking_reserves_capacity_for_the_gui(monkeypatch):
    module = Tracking_method_selection.Do_the_track_multi
    monkeypatch.setattr(module, "_job_memory_limit_bytes", lambda: 10**12)
    monkeypatch.setattr(module, "_reader_backend_request", lambda: "opencv")

    # Eight CPUs minus one reader, one assignment process, and one GUI core.
    assert module._safe_worker_count(8, (480, 640, 3), reader_count=1) == 5


def test_parallel_tracking_uses_spawn_from_the_background_thread():
    source = inspect.getsource(Tracking_method_selection.Do_the_track_multi.Do_tracking)
    assert 'multiprocessing.get_context("spawn")' in source


def test_ten_cpu_ten_gb_job_reserves_gui_capacity(monkeypatch):
    module = Tracking_method_selection.Do_the_track_multi
    monkeypatch.setattr(module, "_job_memory_limit_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(module, "_reader_backend_request", lambda: "opencv")

    workers = module._safe_worker_count(10, (2160, 3840, 3), reader_count=1)

    assert workers == 7
    assert workers + 1 + 1 < 10  # preparation + reader + assignment leaves one GUI CPU


def test_multiprocess_cleanup_terminates_and_reaps_every_resource():
    module = Tracking_method_selection.Do_the_track_multi

    class FakeProcess:
        def __init__(self, alive, pid):
            self.alive = alive
            self.pid = pid
            self.terminated = False
            self.joined = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True

        def join(self, timeout):
            assert timeout == 2
            self.joined = True

    class FakeQueue:
        def __init__(self):
            self.cancelled = False
            self.closed = False

        def cancel_join_thread(self):
            self.cancelled = True

        def close(self):
            self.closed = True

    class FakeManager:
        def __init__(self):
            self.stopped = False

        def shutdown(self):
            self.stopped = True

    running = FakeProcess(True, 100)
    finished = FakeProcess(False, 101)
    unstarted = FakeProcess(False, None)
    queue = FakeQueue()
    manager = FakeManager()

    module._cleanup_process_resources([running, finished, unstarted], [queue], manager)

    assert running.terminated
    assert running.joined and finished.joined and not unstarted.joined
    assert queue.cancelled and queue.closed
    assert manager.stopped


def test_tracking_progress_cancellation_is_thread_safe():
    progress = Tracking_method_selection.TrackingProgress()
    assert not progress.is_cancelled()
    progress.cancel()
    assert progress.is_cancelled()


def test_cancelled_parallel_failure_does_not_start_fallback(monkeypatch):
    monkeypatch.setattr(Tracking_method_selection, "_allocated_cpus", lambda: 10)
    progress = Tracking_method_selection.TrackingProgress()

    def cancel_then_fail(**_kwargs):
        progress.cancel()
        raise RuntimeError("cancelled during teardown")

    monkeypatch.setattr(Tracking_method_selection.Do_the_track_multi, "Do_tracking", cancel_then_fail)
    monkeypatch.setattr(
        Tracking_method_selection.Do_the_track, "Do_tracking",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("fallback started after cancellation")),
    )

    result = Tracking_method_selection.Choose_method(
        SimpleNamespace(timer=0, show_load=lambda: None), _video(), "project", "fixed", False,
        update_ui=False, progress=progress,
    )

    assert result is False
