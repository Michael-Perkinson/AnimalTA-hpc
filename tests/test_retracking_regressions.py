"""Regression tests for re-tracking worker failures and cleanup."""

import csv
import threading
from pathlib import Path

import pytest
import numpy as np

from AnimalTA.D_Tracking_process import Do_the_track, security_settings_track
from AnimalTA.E_Post_tracking import Coos_loader_saver
from AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check import make_portion_video
from AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check import Lecteur


def _write_output(path, frame_count):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["Frame", "Time"])
        for frame in range(frame_count):
            writer.writerow([frame, frame / 30])


def test_worker_exception_is_reported_and_stops_peer_instead_of_disappearing(monkeypatch):
    state = Do_the_track._WorkerState(expected_frames=10000)
    monkeypatch.setattr(security_settings_track, "stop_threads", False)

    def failing_worker():
        raise ValueError("contour assignment failed")

    worker = threading.Thread(
        target=Do_the_track._thread_entry,
        args=("coordinate assignment", failing_worker, (), state),
    )
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    error = state.first_error()
    assert isinstance(error, Do_the_track.TrackingWorkerError)
    assert error.worker_name == "coordinate assignment"
    assert "contour assignment failed" in str(error)
    assert security_settings_track.stop_threads is True


def test_partial_output_for_a_large_selection_is_rejected():
    output = Path.cwd() / ".retracking-regression-output.csv"
    try:
        _write_output(output, 999)
        state = Do_the_track._WorkerState(expected_frames=10000)
        state.output_opened = True
        state.produced_frames = 10000
        state.assigned_frames = 999

        with pytest.raises(Do_the_track.TrackingWorkerError, match="assigned 999 of 10000"):
            Do_the_track._validate_tracking_output(str(output), state, "fixed")
    finally:
        output.unlink(missing_ok=True)


def test_complete_output_for_a_large_selection_is_accepted():
    output = Path.cwd() / ".retracking-regression-output.csv"
    try:
        _write_output(output, 10000)
        state = Do_the_track._WorkerState(expected_frames=10000)
        state.output_opened = True
        state.produced_frames = 10000
        state.assigned_frames = 10000

        Do_the_track._validate_tracking_output(str(output), state, "fixed")
    finally:
        output.unlink(missing_ok=True)


def test_partial_worker_output_never_replaces_previous_result():
    worker = Path.cwd() / ".tracking-worker-output.csv"
    final = Path.cwd() / ".tracking-final-output.csv"
    try:
        final.write_text("previous-good-result", encoding="utf-8")
        _write_output(worker, 2)
        state = Do_the_track._WorkerState(expected_frames=3)
        state.output_opened = True
        state.produced_frames = 3
        state.assigned_frames = 2

        with pytest.raises(Do_the_track.TrackingWorkerError):
            Do_the_track._commit_tracking_output(str(worker), str(final), state, "fixed")

        assert final.read_text(encoding="utf-8") == "previous-good-result"
        assert not worker.exists()
    finally:
        worker.unlink(missing_ok=True)
        final.unlink(missing_ok=True)


def test_tracking_entrypoint_releases_capture_after_success(monkeypatch):
    class Capture:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(security_settings_track, "capture", capture)
    monkeypatch.setattr(Do_the_track, "_do_tracking", lambda *args, **kwargs: True)

    assert Do_the_track.Do_tracking(None, None, None, "fixed") is True
    assert capture.released is True
    assert security_settings_track.capture is None


def test_tracking_entrypoint_releases_capture_after_worker_failure(monkeypatch):
    class Capture:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(security_settings_track, "capture", capture)

    def fail(*args, **kwargs):
        raise RuntimeError("reader failed")

    monkeypatch.setattr(Do_the_track, "_do_tracking", fail)
    with pytest.raises(RuntimeError, match="reader failed"):
        Do_the_track.Do_tracking(None, None, None, "fixed")

    assert capture.released is True
    assert security_settings_track.capture is None


def test_portion_retracking_disables_worker_ui_updates():
    # Keep this as a structural guard: the worker must not call Tk methods;
    # the main thread polls progress through _poll_tracking instead.
    source = open(
        "AnimalTA/E_Post_tracking/a_Tracking_verification/Interface_portion.py",
        encoding="utf-8",
    ).read()
    assert "update_ui=False" in source
    assert "self._poll_tracking(tracking_type)" in source
    assert "self.load_frame.show_load(self.timer, process_events=False)" in source


def test_portion_video_clone_does_not_deep_copy_frame_sized_data():
    class Video:
        pass

    source = Video()
    source.Cropped = [False, [0, 99]]
    source.Identities = [[0, "A", (1, 2, 3)]]
    source.Track = [False, [50, 0, 0, [0, 5000], [0, 1], 500, [1], False, True]]
    source.img_list = np.zeros((200, 128, 128, 3), dtype=np.uint8)
    source.Back = [1, np.zeros((128, 128), dtype=np.uint8)]
    source.Mask = [True, [[[0, 0]]]]
    source.Stab = [True, [[1, 2]]]

    portion = make_portion_video(source)

    assert portion is not source
    assert portion.img_list is source.img_list
    assert portion.Back is not source.Back
    assert portion.Mask is not source.Mask
    assert portion.Stab is not source.Stab
    assert portion.Cropped is not source.Cropped
    assert portion.Track is not source.Track
    assert portion.Identities is not source.Identities
    portion.Cropped[1][0] = 20
    portion.Track[1][0] = 60
    portion.Back[1][0, 0] = 99
    assert source.Cropped[1][0] == 0
    assert source.Track[1][0] == 50
    assert source.Back[1][0, 0] == 0


def test_fixed_csv_row_generation_is_streaming_friendly():
    class Video:
        Frame_rate = [30, 30]
        Identities = [[0, "A", (1, 2, 3)]]

    coos = np.array(
        [[[1.0, 2.0], [-1000.0, -1000.0], [3.0, 4.0]]], dtype=float
    )
    rows = list(Coos_loader_saver.iter_fixed_rows(Video(), coos))

    assert rows[0] == ["Frame", "Time", "X_Arena0_IndA", "Y_Arena0_IndA"]
    assert rows[1] == [0, 0.0, 1.0, 2.0]
    assert rows[2] == [1, 0.03, "NA", "NA"]
    assert rows[3] == [2, 0.07, 3.0, 4.0]


def test_variable_csv_save_does_not_materialize_coordinate_tables():
    class Video:
        Frame_rate = [30, 30]
        Cropped = [True, [100, 102]]
        Identities = [[0, "A", (1, 2, 3)]]

    output = Path.cwd() / ".variable-regression-output.csv"
    coos = np.array(
        [[[1.0, 2.0], [-1000.0, -1000.0], [3.0, 4.0]]], dtype=float
    )
    try:
        Coos_loader_saver.save_variable(Video(), coos, str(output))
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=";"))
    finally:
        output.unlink(missing_ok=True)

    assert rows[0] == ["Frame", "Time", "Arena", "Ind", "X", "Y"]
    assert [row[0] for row in rows[1:]] == ["100", "102"]


def test_fixed_csv_loader_streams_rows_without_object_table(monkeypatch):
    class Window:
        def destroy(self):
            pass

    class Loading:
        def __init__(self, _parent):
            pass

        def grid(self):
            pass

        def show_load(self, _value):
            pass

        def grab_set(self):
            pass

        def grab_release(self):
            pass

        def destroy(self):
            pass

    class Video:
        Frame_rate = [30, 30]
        Identities = [[0, "A", (1, 2, 3)]]

    output = Path.cwd() / ".fixed-loader-regression.csv"
    try:
        output.write_text(
            "Frame;Time;X_Arena0_IndA;Y_Arena0_IndA\n"
            "0;0;1;2\n"
            "1;0.03;NA;NA\n"
            "2;0.07;3;4\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(Coos_loader_saver, "Toplevel", Window)
        monkeypatch.setattr(Coos_loader_saver.Class_loading_Frame, "Loading", Loading)

        coos, who_is_here = Coos_loader_saver.load_fixed(Video(), str(output))

        assert coos.shape == (1, 3, 2)
        np.testing.assert_allclose(coos[0, 0], (1, 2))
        assert np.all(coos[0, 1] == -1000)
        np.testing.assert_allclose(coos[0, 2], (3, 4))
        # Fixed tracking keeps every identity present in the membership map;
        # the verification view recalculates missing values separately.
        assert who_is_here == [[0], [0], [0]]
    finally:
        output.unlink(missing_ok=True)


def test_undo_snapshot_copies_each_identity_without_advanced_indexing(monkeypatch):
    class Video:
        Folder = str(Path.cwd())

    reader = object.__new__(Lecteur)
    reader.Vid = Video()
    reader.Coos = np.arange(2 * 100 * 2, dtype=float).reshape(2, 100, 2)

    import AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check as check
    monkeypatch.setattr(
        check.UserMessages,
        "tmp_portion_dir_path",
        lambda _folder, create=True: str(Path.cwd()),
    )
    snapshot = reader._make_undo_snapshot_from_coos([0, 1], 10, 20)
    try:
        np.testing.assert_array_equal(snapshot, reader.Coos[:, 10:20, :])
        source_text = Path(
            "AnimalTA/E_Post_tracking/a_Tracking_verification/Interface_Check.py"
        ).read_text(encoding="utf-8")
        assert "self.Coos[self.inds_portion" not in source_text
    finally:
        check._discard_undo_snapshot(snapshot)


def test_undo_snapshot_cleanup_closes_mapping_before_removing_file():
    import AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check as check

    path = Path.cwd() / ".undo-cleanup-regression.dat"
    mapping = np.memmap(path, dtype=float, mode="w+", shape=(4, 2))
    mapping[:] = 1
    mapping.flush()
    check._discard_undo_snapshot(mapping)
    assert not path.exists()
