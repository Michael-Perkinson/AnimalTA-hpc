import cv2
import decord
from AnimalTA.A_General_tools import UserMessages, Video_loader as VL
from AnimalTA.D_Tracking_process import Function_prepare_images, Function_assign_cnts, security_settings_track, Treat_simgle_image
import numpy as np
import os
import threading
import time
import queue
import pickle
import sys
import re
import datetime as _dt
import csv
import traceback

def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[track {ts}] {msg}", file=sys.stderr, flush=True)

'''
To improve the speed of the tracking, we will separate the work in 2 threads.
1. Image loading, and modifications (stabilization, light correction, greyscale...) until contours are get
2. Target assignment and data recording
'''


security_settings_track.stop_threads=False
WORKER_CANCEL_TIMEOUT = 30.0
_tracking_gate = threading.Lock()


class _TrackingTimeout(RuntimeError):
    """A timeout that retains ownership of workers and the video capture."""

    def __init__(self, error, workers):
        self.error = error
        self.workers = workers
        super().__init__(str(error))


class _ProgressValue:
    """Store numeric tracking progress without accessing Tk from worker threads."""

    def __init__(self, value=0.0):
        self._value = value
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._value

    def set(self, value):
        with self._lock:
            self._value = value


class TrackingWorkerError(RuntimeError):
    """An exception raised by one of the single-process tracking workers."""

    def __init__(self, worker_name, error, formatted_traceback=None):
        self.worker_name = worker_name
        self.original_error = error
        self.formatted_traceback = formatted_traceback or ""
        super().__init__(
            "{} worker failed: {}: {}".format(
                worker_name, type(error).__name__, error
            )
        )


class _WorkerState:
    """Thread-safe state shared by the producer and assignment workers."""

    def __init__(self, expected_frames):
        self.expected_frames = expected_frames
        self.errors = queue.Queue()
        self._lock = threading.Lock()
        self.produced_frames = 0
        self.assigned_frames = 0
        self.output_opened = False

    def record_error(self, worker_name, error):
        self.errors.put(
            TrackingWorkerError(worker_name, error, traceback.format_exc())
        )

    def first_error(self):
        try:
            return self.errors.get_nowait()
        except queue.Empty:
            return None

    def increment(self, name):
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def mark_output_opened(self):
        with self._lock:
            self.output_opened = True


def _thread_entry(worker_name, function, args, worker_state):
    """Run a worker and turn exceptions into a shutdown signal for its peer."""
    try:
        function(*args)
    except BaseException as error:
        worker_state.record_error(worker_name, error)
        # The peer may be blocked on a full/empty queue.  The queue operations
        # use short timeouts and check this flag, so this wakes both workers.
        security_settings_track.stop_threads = True


def _close_tracking_capture():
    """Release the process-wide capture without assuming a backend API."""
    capture = getattr(security_settings_track, "capture", None)
    security_settings_track.capture = None
    if capture is None:
        return
    try:
        release = getattr(capture, "release", None)
        if callable(release):
            release()
    except Exception:
        # Capture cleanup must not hide the original tracking exception.
        pass
    del capture


def _finish_orphaned_tracking(workers):
    """Keep the run gate and capture owned until timed-out workers finish."""
    try:
        for worker in workers:
            worker.join()
    finally:
        _close_tracking_capture()
        _tracking_gate.release()


def _discard_tracking_output(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        # A cleanup failure must not hide the worker failure that triggered it.
        pass


def _expected_frames(start, end, one_every):
    return len({round(frame) for frame in np.arange(start, end + one_every, one_every)})


def _validate_tracking_output(path, worker_state, tracking_type):
    """Reject missing, unopened, malformed, or partial worker output."""
    if not worker_state.output_opened:
        raise TrackingWorkerError(
            "output", RuntimeError("the coordinate writer was never opened")
        )
    if worker_state.produced_frames != worker_state.expected_frames:
        raise TrackingWorkerError(
            "output",
            RuntimeError(
                "prepared {} of {} expected frames".format(
                    worker_state.produced_frames, worker_state.expected_frames
                )
            ),
        )
    if worker_state.assigned_frames != worker_state.expected_frames:
        raise TrackingWorkerError(
            "output",
            RuntimeError(
                "assigned {} of {} expected frames".format(
                    worker_state.assigned_frames, worker_state.expected_frames
                )
            ),
        )
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise TrackingWorkerError("output", RuntimeError("coordinate output is empty"))

    with open(path, newline="", encoding="utf-8") as file:
        rows = csv.reader(file, delimiter=";")
        try:
            header = next(rows)
        except StopIteration:
            raise TrackingWorkerError("output", RuntimeError("coordinate output has no header"))
        if len(header) < 2 or header[0] != "Frame":
            raise TrackingWorkerError("output", RuntimeError("coordinate output header is invalid"))
        data_rows = sum(1 for _row in rows)

    # Fixed tracking emits one row per prepared frame.  Variable tracking can
    # legitimately emit only rows for targets that enter an arena, so the
    # worker counters above are the completeness check for that format.
    if tracking_type == "fixed" and data_rows != worker_state.expected_frames:
        raise TrackingWorkerError(
            "output",
            RuntimeError(
                "coordinate output contains {} of {} expected rows".format(
                    data_rows, worker_state.expected_frames
                )
            ),
        )


def _commit_tracking_output(worker_path, final_path, worker_state, tracking_type):
    """Validate a run and atomically publish it as the user-visible result."""
    try:
        _validate_tracking_output(worker_path, worker_state, tracking_type)
        os.replace(worker_path, final_path)
    except BaseException:
        _discard_tracking_output(worker_path)
        raise


def Do_tracking(parent, Vid, folder, type, portion=False, prev_row=None, arena_interest=None, test=False, head_tail=False, ref_frame=None, update_ui=True, progress=None):
    """Run single-process tracking while always releasing its video reader."""
    if not _tracking_gate.acquire(blocking=False):
        raise TrackingWorkerError(
            "coordinator", RuntimeError("another tracking run is still shutting down")
        )
    orphaned = False
    try:
        return _do_tracking(
            parent, Vid, folder, type, portion, prev_row, arena_interest, test,
            head_tail, ref_frame, update_ui, progress
        )
    except _TrackingTimeout as timeout:
        orphaned = True
        threading.Thread(
            target=_finish_orphaned_tracking,
            args=(timeout.workers,),
            daemon=True,
        ).start()
        raise timeout.error
    finally:
        if not orphaned:
            _close_tracking_capture()
            _tracking_gate.release()


def _do_tracking(parent, Vid, folder, type, portion=False, prev_row=None, arena_interest=None, test=False, head_tail=False, ref_frame=None, update_ui=True, progress=None):
    '''This is the main tracking function of the program.
    parent=container (main window)
    Vid=current video
    portion= True if it is a rerun of the tracking over a short part of the video (for corrections)
    prev_row=If portion is True, this correspond to the last known coordinates of the targets.
    '''
    security_settings_track.stop_threads=False
    Param_file = UserMessages.settings_file_path()
    with open(Param_file, 'rb') as fp:
        Params = pickle.load(fp)
        use_Kalman=Params["Use_Kalman"]

    # Where coordinates will be saved, if the folder did not exists, it is created.
    if Vid.User_Name == Vid.Name:
        file_name = Vid.Name
        point_pos = file_name.rfind(".")
        if file_name[point_pos:].lower()!=".avi":
            file_name = Vid.User_Name
        else:
            file_name = file_name[:point_pos]
    else:
        file_name = Vid.User_Name

    if portion:
        To_save = os.path.join(UserMessages.tmp_portion_dir_path(folder, create=True), file_name + "_TMP_portion_Coordinates.csv")
    else:
        To_save = os.path.join(UserMessages.coordinates_dir_path(folder, create=True), file_name + "_Coordinates.csv")
    Worker_output = To_save + ".part"
    # Never resume or validate a stale partial file from an earlier attempt.
    _discard_tracking_output(Worker_output)

    # if the user choose to reduce the frame rate.
    one_every = Vid.Frame_rate[0] / Vid.Frame_rate[1]
    Which_part = 0

    start = Vid.Cropped[1][0]  # Video beginning (after crop)
    end = Vid.Cropped[1][1]  # Video end (after crop)

    if Vid.Cropped[0]:
        if len(Vid.Fusion) > 1:  # If the video results from concatenated videos
            Which_part = [index for index, Fu_inf in enumerate(Vid.Fusion) if Fu_inf[0] <= start][-1]

    if ref_frame is None:
        First_frame = start
    else:
        First_frame = ref_frame

    Which_part_first=0
    if Vid.Cropped[0]:
        if len(Vid.Fusion) > 1:  # If the video results from concatenated videos
            Which_part_first = [index for index, Fu_inf in enumerate(Vid.Fusion) if Fu_inf[0] <= First_frame][-1]

    security_settings_track.activate_protection=False
    security_settings_track.activate_super_protection=False

    if Vid.type=="Video":
        _tlog("opening video...")
        t0 = time.time()
        security_settings_track.capture = decord.VideoReader(Vid.Fusion[Which_part_first][1])  # Open video
        Prem_image_to_show = security_settings_track.capture[First_frame - Vid.Fusion[Which_part_first][0]].asnumpy()  # Take the first image
        security_settings_track.capture.seek(0)
        security_settings_track.capture = decord.VideoReader(Vid.Fusion[Which_part][1])  # Open video
        security_settings_track.capture.seek(0)
        _tlog("video ready ({:.1f}s)".format(time.time()-t0))

    else:
        security_settings_track.capture = VL.Video_Loader(Vid, is_crop=False, is_rotate=False, ref_frame=First_frame)
        Prem_image_to_show=security_settings_track.capture[First_frame]

    if type=="fixed":
        mask, or_bright, Arenas, Prem_image_to_show = Treat_simgle_image.Prepare_Vid(Vid, Prem_image_to_show, type, portion=portion, arena_interest=arena_interest)
    elif type=="variable":
        mask, or_bright, Arenas, Main_Arenas_image, Main_Arenas_Bimage, Prem_image_to_show = Treat_simgle_image.Prepare_Vid(Vid,
                                                                                                        Prem_image_to_show,
                                                                                                        type,
                                                                                                        portion=portion,
                                                                                                        arena_interest=arena_interest)

    Extracted_cnts = queue.Queue(maxsize=500)
    Security_break=threading.Event()

    AD = _ProgressValue()
    expected_frames = _expected_frames(start, end, one_every)
    worker_state = _WorkerState(expected_frames)

    if test:
        end=start+5
        worker_state.expected_frames = _expected_frames(start, end, one_every)
        result_container = {}
        Th_extract_cnts=threading.Thread(
            target=_thread_entry,
            args=(
                "image preparation", Function_prepare_images.Image_modif,
                (Security_break, Vid, start, end, one_every, Which_part,
                 Prem_image_to_show, mask, or_bright, Extracted_cnts, AD,
                 result_container, worker_state), worker_state,
            ),
            daemon=True,
        )
        Th_extract_cnts.start()
        while Th_extract_cnts.is_alive():
            if update_ui:
                parent.show_load()
            time.sleep(0.1)
        Th_extract_cnts.join()
        error = worker_state.first_error()
        if error is not None:
            raise error
        result = result_container.get('result', None)
        return(result)

    else:
        Th_extract_cnts = threading.Thread(
            target=_thread_entry,
            args=(
                "image preparation", Function_prepare_images.Image_modif,
                (Security_break, Vid, start, end, one_every, Which_part,
                 Prem_image_to_show, mask, or_bright, Extracted_cnts, AD,
                 {}, worker_state), worker_state,
            ),
            daemon=True,
        )

        if type=="fixed":
            associate_function = Function_assign_cnts.Treat_cnts_fixed
            associate_args = (Vid, Arenas, start, end, prev_row,
                              Extracted_cnts, Th_extract_cnts, Worker_output,
                              portion, one_every, AD, use_Kalman, head_tail,
                              worker_state)
        elif type=="variable":
            keep_entrance=Params["Keep_entrance"]
            associate_function = Function_assign_cnts.Treat_cnts_variable
            associate_args = (Vid, Arenas, Main_Arenas_image,
                              Main_Arenas_Bimage, start, end, prev_row,
                              Extracted_cnts, Th_extract_cnts, Worker_output,
                              portion, one_every, AD, not keep_entrance,
                              use_Kalman, head_tail, worker_state)
        else:
            raise ValueError("Unknown tracking type: {}".format(type))

        Th_associate_cnts=threading.Thread(
            target=_thread_entry,
            args=("coordinate assignment", associate_function,
                  associate_args, worker_state),
            daemon=True,
        )

        Th_extract_cnts.start()
        Th_associate_cnts.start()

        stop_requested_at = None
        while Th_extract_cnts.is_alive() or Th_associate_cnts.is_alive():
            value = (AD.get()-start)/(end + one_every - start)
            if progress is None:
                parent.timer = value
            else:
                progress.set(value)
            time.sleep(0.05)

            cancellation_requested = (
                progress is not None
                and getattr(progress, "is_cancelled", lambda: False)()
            )
            worker_failed = not worker_state.errors.empty()
            if cancellation_requested or worker_failed:
                security_settings_track.stop_threads = True
                Security_break.set()

            if security_settings_track.stop_threads and stop_requested_at is None:
                stop_requested_at = time.monotonic()
            if (
                security_settings_track.stop_threads
                and time.monotonic() - stop_requested_at >= WORKER_CANCEL_TIMEOUT
            ):
                _discard_tracking_output(Worker_output)
                raise _TrackingTimeout(
                    TrackingWorkerError(
                        "coordinator",
                        TimeoutError(
                            "tracking workers did not stop within {} seconds".format(
                                WORKER_CANCEL_TIMEOUT
                            )
                        ),
                    ),
                    (Th_extract_cnts, Th_associate_cnts),
                )

            if worker_failed:
                security_settings_track.stop_threads = True
                Security_break.set()

            overload = security_settings_track.check_memory_overload()#Avoid memory leak problems
            if overload==1:
                security_settings_track.activate_super_protection=True

            elif overload==0:
                security_settings_track.activate_protection=True

            elif overload==-1:
                security_settings_track.activate_protection=False
                security_settings_track.activate_super_protection = False
                Security_break.set()

        Th_extract_cnts.join(timeout=1)
        Th_associate_cnts.join(timeout=1)
        if Th_extract_cnts.is_alive() or Th_associate_cnts.is_alive():
            _discard_tracking_output(Worker_output)
            raise _TrackingTimeout(
                TrackingWorkerError(
                    "coordinator", TimeoutError("a tracking worker did not terminate")
                ),
                (Th_extract_cnts, Th_associate_cnts),
            )

        error = worker_state.first_error()
        if error is not None:
            _discard_tracking_output(Worker_output)
            raise error
        if security_settings_track.stop_threads:
            _discard_tracking_output(Worker_output)
            if type=="fixed":
                return (False)
            elif type=="variable":
                return (False,0)
        else:
            # Preserve an existing good result until the new result has
            # passed completeness validation.
            _commit_tracking_output(Worker_output, To_save, worker_state, type)
            if progress is None:
                parent.timer = 1
            else:
                progress.set(1)
            if update_ui:
                parent.show_load()
            if type == "fixed":
                return (True)
            elif type=="variable":
                return (True,security_settings_track.ID_kepts)


def urgent_close(Vid):
    security_settings_track.stop_threads = True

