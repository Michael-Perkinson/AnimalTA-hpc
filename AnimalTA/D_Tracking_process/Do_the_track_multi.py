import multiprocessing
import cv2
import decord
from AnimalTA.A_General_tools import Function_draw_arenas as Dr, UserMessages, Message_simple_question as MsgBox
from AnimalTA.D_Tracking_process import Function_prepare_images_multi, Function_assign_cnts_multi, security_settings_track, Treat_simgle_image
import numpy as np
import os
from tkinter import *
import threading
import queue
import pickle
import sys
import time
import datetime as _dt
import psutil

def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[track {ts}] {msg}", file=sys.stderr, flush=True)


def _job_memory_limit_bytes():
    """Return the memory limit imposed by the SLURM cgroup, or fall back to node available RAM."""
    # cgroup v2
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            with open(path) as f:
                val = f.read().strip()
            if val != "max":
                limit = int(val)
                # cgroup v1 reports a sentinel near 2^63 when unlimited
                if limit < (1 << 62):
                    return limit
        except Exception:
            pass
    # SLURM_MEM_PER_NODE is set in MB
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem:
        try:
            return int(slurm_mem) * 1024 * 1024
        except ValueError:
            pass
    return psutil.virtual_memory().available


def _safe_worker_count(allocated_cpus, vid_shape):
    h, w = vid_shape[0], vid_shape[1]
    # Reader model: Queue_raw holds ~2 raw frames per worker + ~2 frames of processing state per worker.
    # Raw BGR decoded frame = h*w*3 bytes.
    bytes_per_worker = h * w * 3 * 4
    job_mem = _job_memory_limit_bytes()
    usable = job_mem * 0.80
    # Reserve 1 core for the reader process and 1 for the assignment worker.
    max_from_cpus = max(1, allocated_cpus - 2)
    mem_based = max(1, int(usable / bytes_per_worker))
    count = min(max_from_cpus, mem_based)
    _tlog(
        "worker cap: cpu_limit={} mem_limit={} job_mem={:.1f}GB per_worker_est={:.0f}MB -> using {}".format(
            max_from_cpus,
            mem_based,
            job_mem / 1e9,
            bytes_per_worker / 1e6,
            count,
        )
    )
    return count


def _frame_reader(Queue_raw, Vid, start, end, one_every, num_workers):
    """Opens the video once and feeds (frame_number, raw_image) pairs to Queue_raw sequentially.
    Sends num_workers None sentinels after the last frame to signal workers to stop."""
    all_frames = np.arange(start, end + one_every, one_every)
    total_frames = len(all_frames)

    Which_part = 0
    if len(Vid.Fusion) > 1:
        Which_part = [i for i, fu in enumerate(Vid.Fusion) if fu[0] <= round(all_frames[0])][-1]

    cap_pos = 0
    capture = None

    _t_grab = _t_decode = _t_queue_wait = 0.0
    _n = 0
    _REPORT_EVERY = 200
    _reader_start = time.perf_counter()

    _tlog("reader starting: total_frames={} num_workers={} queue_maxsize={}".format(
        total_frames, num_workers, Queue_raw._maxsize))

    try:
        if Vid.type == "Video":
            capture = cv2.VideoCapture(Vid.Fusion[Which_part][1])
            if not capture.isOpened():
                raise RuntimeError("Reader could not open {}".format(Vid.Fusion[Which_part][1]))

        for frame_f in all_frames:
            frame = round(frame_f)

            if len(Vid.Fusion) > 1 and Which_part < len(Vid.Fusion) - 1 and frame >= Vid.Fusion[Which_part + 1][0]:
                while len(Vid.Fusion) > 1 and Which_part < len(Vid.Fusion) - 1 and frame >= Vid.Fusion[Which_part + 1][0]:
                    Which_part += 1
                if Vid.type == "Video":
                    if capture is not None:
                        capture.release()
                    capture = cv2.VideoCapture(Vid.Fusion[Which_part][1])
                    cap_pos = 0

            if Vid.type == "Video":
                local_frame = frame - Vid.Fusion[Which_part][0]
                _t0 = time.perf_counter()
                while cap_pos <= local_frame:
                    cap_pos += 1
                    if not capture.grab():
                        raise RuntimeError("Reader could not grab frame {} from {}".format(frame, Vid.Fusion[Which_part][1]))
                _t_grab += time.perf_counter() - _t0

                _t0 = time.perf_counter()
                ret, image = capture.retrieve()
                _t_decode += time.perf_counter() - _t0
                if not ret or image is None:
                    raise RuntimeError("Reader could not retrieve frame {} from {}".format(frame, Vid.Fusion[Which_part][1]))
            else:
                _t0 = time.perf_counter()
                image = cv2.imread(os.path.join(Vid.Fusion[Which_part][1], Vid.img_list[frame - Vid.Fusion[Which_part][0]]))
                _t_decode += time.perf_counter() - _t0
                if image is None:
                    raise RuntimeError("Reader could not read image frame {}".format(frame))

            _t0 = time.perf_counter()
            Queue_raw.put((frame, image))
            _t_queue_wait += time.perf_counter() - _t0

            _n += 1
            if _n % _REPORT_EVERY == 0:
                n = _REPORT_EVERY
                elapsed = time.perf_counter() - _reader_start
                overall_fps = _n / elapsed
                _tlog(
                    "reader frame={} ({}/{}) avg/{}: "
                    "grab={:.2f}ms decode={:.2f}ms queue_wait={:.2f}ms "
                    "overall_fps={:.0f} queue_size={}".format(
                        frame, _n, total_frames, n,
                        _t_grab / n * 1000,
                        _t_decode / n * 1000,
                        _t_queue_wait / n * 1000,
                        overall_fps,
                        Queue_raw.qsize(),
                    )
                )
                _t_grab = _t_decode = _t_queue_wait = 0.0

        elapsed = time.perf_counter() - _reader_start
        _tlog("reader done: {} frames in {:.1f}s ({:.0f} fps avg); sending {} sentinels".format(
            total_frames, elapsed, total_frames / elapsed if elapsed > 0 else 0, num_workers))

        for _ in range(num_workers):
            Queue_raw.put(None)

    finally:
        if capture is not None:
            capture.release()

'''
To improve the speed of the tracking, we will separate the work in 2 threads.
1. Image loading, and modifications (stabilization, light correction, greyscale...) until contours are get
2. Target assignment and data recording
'''

def Do_tracking(parent, Vid, folder, type, portion=False, prev_row=None, arena_interest=None, head_tail=False, ref_frame=None):
    '''This is the main tracking function of the program.
    parent=container (main window)
    Vid=current video
    portion= True if it is a rerun of the tracking over a short part of the video (for corrections)
    prev_row=If portion is True, this correspond to the last known coordinates of the targets.
    '''
    # Language importation
    Language = StringVar()
    f = open(UserMessages.resource_path("AnimalTA/Files/Language"), "r", encoding="utf-8")
    Language.set(f.read())
    f.close()
    Messages = UserMessages.Mess[Language.get()]

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

    # if the user choose to reduce the frame rate.
    one_every = Vid.Frame_rate[0] / Vid.Frame_rate[1]

    start = Vid.Cropped[1][0]  # Video beginning (after crop)
    end = Vid.Cropped[1][1]  # Video end (after crop)

    security_settings_track.init()
    security_settings_track.activate_protection=False
    security_settings_track.activate_super_protection=False

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
        _tlog("opening video (multi)...")
        t0 = time.time()
        capture = decord.VideoReader(Vid.Fusion[Which_part_first][1])  # Open video
        capture.seek(0)
        Prem_image_to_show = capture[First_frame - Vid.Fusion[Which_part_first][0]].asnumpy()  # Take the first image
        del capture
        _tlog("video ready ({:.1f}s)".format(time.time()-t0))
    else:
        Prem_image_to_show = cv2.imread(os.path.join(Vid.Fusion[Which_part_first][1], Vid.img_list[First_frame - Vid.Fusion[Which_part_first][0]]))

    if type=="fixed":
        mask, or_bright, Arenas, Prem_image_to_show = Treat_simgle_image.Prepare_Vid(Vid, Prem_image_to_show, type, portion=portion, arena_interest=arena_interest)
    elif type=="variable":
        mask, or_bright, Arenas, Main_Arenas_image, Main_Arenas_Bimage, Prem_image_to_show = Treat_simgle_image.Prepare_Vid(Vid,
                                                                                                        Prem_image_to_show,
                                                                                                        type,
                                                                                                        portion=portion,
                                                                                                        arena_interest=arena_interest)

    try:
        allocated_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        allocated_cpus = multiprocessing.cpu_count()
    nb_cpu_extract_treat = _safe_worker_count(allocated_cpus, Vid.shape)
    _tlog("CPUs allocated: {} | total on node: {} | worker processes to spawn: {}".format(allocated_cpus, multiprocessing.cpu_count(), nb_cpu_extract_treat))
    Nb_images_processed=multiprocessing.Value("i",0)


    manager = multiprocessing.Manager()

    Processes = []

    # Bounded raw-frame queue: reader pushes decoded frames, workers consume.
    # maxsize keeps memory bounded; backpressure throttles the reader when workers are busy.
    Queue_raw = multiprocessing.Queue(maxsize=nb_cpu_extract_treat * 2)

    #We create one queue per cpu (-1 as one cpu will be in charge of the tracking itself)
    Queues_cnt=multiprocessing.Queue(maxsize=100)

    if type=="fixed":
        Processes.append(multiprocessing.Process(target=Function_assign_cnts_multi.Treat_cnts_fixed, args=(Queues_cnt, Nb_images_processed, Vid, Arenas, start, end, prev_row, To_save, portion, one_every, use_Kalman, head_tail)))
    elif type == "variable":
        keep_entrance = Params["Keep_entrance"]
        ID_kepts = manager.list([manager.list(sublist) for sublist in [[] for _ in Arenas]])
        Processes.append(multiprocessing.Process(target=Function_assign_cnts_multi.Treat_cnts_variable, args=(Queues_cnt, Nb_images_processed,Vid, Arenas, Main_Arenas_image, Main_Arenas_Bimage, start, end, ID_kepts, prev_row, To_save, portion, one_every, not keep_entrance, use_Kalman, head_tail)))


    # Single reader process: opens video once, feeds raw frames to Queue_raw.
    Processes.append(multiprocessing.Process(target=_frame_reader, args=(Queue_raw, Vid, start, end, one_every, nb_cpu_extract_treat)))

    # Worker processes: pull raw frames, run processing pipeline, push contour batches.
    for process_ID in range(nb_cpu_extract_treat):
        Processes.append(multiprocessing.Process(target=Function_prepare_images_multi.Image_modif, args=(Queues_cnt, Queue_raw, Vid, Prem_image_to_show, mask, or_bright, process_ID)))

    for process in Processes:
        process.start()

    while len([p for p in Processes if p.is_alive()])>0:
        time.sleep(0.25)
        failed_processes = [p for p in Processes if p.exitcode not in (None, 0)]
        if failed_processes:
            for process in Processes:
                if process.is_alive():
                    process.terminate()
            for process in Processes:
                process.join(timeout=1)
            failed_details = ", ".join(
                "pid={} exitcode={}".format(process.pid, process.exitcode)
                for process in failed_processes
            )
            raise RuntimeError("Multiprocess tracking worker failed ({})".format(failed_details))
        with Nb_images_processed.get_lock():
            parent.timer=(Nb_images_processed.value)/(Vid.Cropped[1][1]/one_every-Vid.Cropped[1][0]/one_every)

    failed_processes = [p for p in Processes if p.exitcode not in (None, 0)]
    if failed_processes:
        failed_details = ", ".join(
            "pid={} exitcode={}".format(process.pid, process.exitcode)
            for process in failed_processes
        )
        raise RuntimeError("Multiprocess tracking worker failed ({})".format(failed_details))

    parent.timer = 1
    parent.show_load()



    if security_settings_track.stop_threads:
        if type=="fixed":
            return (False)
        elif type=="variable":
            return (False,0)
    else:
        if type == "fixed":
            return (True)
        elif type=="variable":
            ID_kepts_toret = [list(sublist) for sublist in ID_kepts]
            return (True,list(ID_kepts_toret))


def urgent_close(Vid):
    security_settings_track.stop_threads = True

