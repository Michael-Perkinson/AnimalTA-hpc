from AnimalTA.D_Tracking_process import Do_the_track_multi, Do_the_track
import multiprocessing
import os
import sys
import datetime as _dt
from AnimalTA.A_General_tools import UserMessages
import pickle
import builtins
import threading

def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[track {ts}] {msg}", file=sys.stderr, flush=True)


def _allocated_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return multiprocessing.cpu_count()


class TrackingProgress:
    """Thread-safe progress state shared by tracking and the Tk polling loop."""

    def __init__(self, value=0.0):
        self._value = value
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def get(self):
        with self._lock:
            return self._value

    def set(self, value):
        with self._lock:
            self._value = value

    def cancel(self):
        self._cancelled.set()

    def is_cancelled(self):
        return self._cancelled.is_set()


def Choose_method(parent, Vid, folder, type, head_tail, vid_num=0, vid_total=1,
                  update_ui=True, progress=None):
    if progress is None:
        parent.timer = 0
    else:
        progress.set(0)
    if update_ui:
        parent.show_load()

    Param_file = UserMessages.settings_file_path()
    with open(Param_file, 'rb') as fp:
        Params = pickle.load(fp)
        Low_Priority = Params["Low_priority"]

    one_every = Vid.Frame_rate[0] / Vid.Frame_rate[1]
    total_frames = int((Vid.Cropped[1][1] - Vid.Cropped[1][0]) / one_every)
    duration_s = total_frames / Vid.Frame_rate[1]
    allocated = _allocated_cpus()

    # The parallel pipeline needs a reader, an assignment process, at least one
    # preparation worker, and capacity for the GUI/orchestrator.
    if Low_Priority or Vid.Back[0] == 2 or total_frames < 2000 or allocated < 4:
        method=0
    else:
        method=1

    if Low_Priority:
        method=0

    _tlog("tracking: method={} (vid={}/{}, allocated_cpus={}, segments={}, frames={}, duration={:.0f}s, first_vid={})".format(
        "parallel" if method else "single", vid_num, vid_total, allocated, len(Vid.Fusion), total_frames, duration_s, Vid.User_Name))
    if method==0:
        succeed = Do_the_track.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, test=False, head_tail=head_tail, update_ui=update_ui, progress=progress)
        return succeed
    else:
        try:
            succeed = Do_the_track_multi.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, head_tail=head_tail, update_ui=update_ui, progress=progress)
            return succeed
        except Exception as exc:
            # A close request must never turn a cancelled parallel run into a
            # fresh single-process run against the same output file.
            if progress is not None and progress.is_cancelled():
                return False if type == "fixed" else (False, 0)
            _tlog("multiprocess tracking failed; retrying single-process tracker: {}: {}".format(builtins.type(exc).__name__, exc))
            succeed = Do_the_track.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, test=False, head_tail=head_tail, update_ui=update_ui, progress=progress)
            return succeed

