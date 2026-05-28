from AnimalTA.D_Tracking_process import Do_the_track_multi, Do_the_track
import multiprocessing
import os
import sys
import datetime as _dt
from AnimalTA.A_General_tools import UserMessages
import pickle

def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[track {ts}] {msg}", file=sys.stderr, flush=True)


def _allocated_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return multiprocessing.cpu_count()


def Choose_method(parent, Vid, folder, type, head_tail):
    parent.timer = 0
    parent.show_load()

    Param_file = UserMessages.settings_file_path()
    with open(Param_file, 'rb') as fp:
        Params = pickle.load(fp)
        Low_Priority = Params["Low_priority"]

    one_every = Vid.Frame_rate[0] / Vid.Frame_rate[1]
    total_frames = int((Vid.Cropped[1][1] - Vid.Cropped[1][0]) / one_every)
    duration_s = total_frames / Vid.Frame_rate[1]
    allocated = _allocated_cpus()

    if Low_Priority or Vid.Back[0] == 2 or total_frames < 2000 or allocated < 2:
        method=0
    else:
        method=1

    if Low_Priority:
        method=0

    _tlog("tracking: method={} (allocated_cpus={}, frames={}, duration={:.0f}s, files={}, vid={})".format(
        "parallel" if method else "single", allocated, total_frames, duration_s, len(Vid.Fusion), Vid.User_Name))
    if method==0:
        succeed = Do_the_track.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, test=False, head_tail=head_tail)
        return succeed
    else:
        try:
            succeed = Do_the_track_multi.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, head_tail=head_tail)
            return succeed
        except Exception as exc:
            _tlog("multiprocess tracking failed; retrying single-process tracker: {}: {}".format(type(exc).__name__, exc))
            succeed = Do_the_track.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, test=False, head_tail=head_tail)
            return succeed

