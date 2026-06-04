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
from multiprocessing.shared_memory import SharedMemory

def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[track {ts}] {msg}", file=sys.stderr, flush=True)


def _reader_backend_request():
    backend = os.environ.get("ANIMALTA_READER_BACKEND", "opencv").strip().lower()
    aliases = {
        "cv2": "opencv",
        "opencv": "opencv",
        "decord": "decord",
        "decord-cpu": "decord",
        "decord_cpu": "decord",
        "decord-gpu": "decord-gpu",
        "decord_gpu": "decord-gpu",
        "nvdec": "decord-gpu",
        "auto": "auto",
    }
    if backend not in aliases:
        _tlog("unknown ANIMALTA_READER_BACKEND={!r}; using opencv".format(backend))
        return "opencv"
    return aliases[backend]


def _decord_batch_size(num_slots):
    raw = os.environ.get("ANIMALTA_DECORD_BATCH", "32")
    try:
        size = int(raw)
    except ValueError:
        _tlog("invalid ANIMALTA_DECORD_BATCH={!r}; using 32".format(raw))
        size = 32
    return max(1, min(size, max(1, num_slots)))


def _decord_thread_count():
    raw = os.environ.get("ANIMALTA_DECORD_THREADS", "0")
    try:
        threads = int(raw)
    except ValueError:
        _tlog("invalid ANIMALTA_DECORD_THREADS={!r}; using decord default".format(raw))
        return 0
    return max(0, threads)


def _auto_decord_threads(allocated_cpus):
    if allocated_cpus >= 16:
        return 4
    if allocated_cpus >= 8:
        return 2
    return 1


def _configure_reader_defaults(allocated_cpus):
    backend = _reader_backend_request()
    if backend in ("auto", "decord", "decord-gpu") and "ANIMALTA_DECORD_THREADS" not in os.environ:
        threads = _auto_decord_threads(allocated_cpus)
        os.environ["ANIMALTA_DECORD_THREADS"] = str(threads)
        _tlog("reader default: ANIMALTA_DECORD_THREADS={} based on allocated_cpus={}".format(
            threads, allocated_cpus))


def _reader_process_count(allocated_cpus, total_frames):
    raw = os.environ.get("ANIMALTA_READER_PROCESSES")
    if raw is not None:
        try:
            requested = int(raw)
        except ValueError:
            _tlog("invalid ANIMALTA_READER_PROCESSES={!r}; using 1".format(raw))
            return 1
        return max(1, min(requested, max(1, allocated_cpus - 2), 4))

    backend = _reader_backend_request()
    if backend == "opencv" and allocated_cpus >= 16 and total_frames >= 10000:
        return 2
    return 1


def _reader_window_size():
    raw = os.environ.get("ANIMALTA_READER_WINDOW", "5000")
    try:
        size = int(raw)
    except ValueError:
        _tlog("invalid ANIMALTA_READER_WINDOW={!r}; using 5000".format(raw))
        size = 5000
    return max(100, size)


def _opencv_seek_gap():
    raw = os.environ.get("ANIMALTA_OPENCV_SEEK_GAP", "100")
    try:
        gap = int(raw)
    except ValueError:
        _tlog("invalid ANIMALTA_OPENCV_SEEK_GAP={!r}; using 100".format(raw))
        gap = 100
    return max(1, gap)


def _split_frame_values(start, end, one_every, reader_count):
    all_frames = np.arange(start, end + one_every, one_every)
    if reader_count <= 1:
        return [all_frames]

    window_size = _reader_window_size()
    reader_parts = [[] for _ in range(reader_count)]
    for window_idx, window_start in enumerate(range(0, len(all_frames), window_size)):
        window = all_frames[window_start:window_start + window_size]
        reader_parts[window_idx % reader_count].append(window)

    return [
        np.concatenate(parts)
        for parts in reader_parts
        if parts
    ]


def _open_opencv_capture(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError("Reader could not open {}".format(path))
    return capture


def _open_decord_capture(path, backend):
    if backend == "decord-gpu":
        gpu_id = int(os.environ.get("ANIMALTA_DECORD_GPU_ID", "0"))
        ctx = decord.gpu(gpu_id)
    else:
        ctx = decord.cpu(0)

    threads = _decord_thread_count()
    kwargs = {"ctx": ctx}
    if threads > 0:
        kwargs["num_threads"] = threads
    return decord.VideoReader(path, **kwargs)


def _open_video_capture(path, requested_backend):
    candidates = ["decord", "opencv"] if requested_backend == "auto" else [requested_backend]
    last_error = None
    for backend in candidates:
        try:
            if backend == "opencv":
                return backend, _open_opencv_capture(path)
            if backend in ("decord", "decord-gpu"):
                return backend, _open_decord_capture(path, backend)
        except Exception as exc:
            last_error = exc
            if requested_backend == "auto":
                _tlog("reader backend {} unavailable for {}; trying fallback ({})".format(
                    backend, os.path.basename(path), exc))
                continue
            raise
    raise RuntimeError("Reader could not open {} ({})".format(path, last_error))


def _close_video_capture(backend, capture):
    if capture is None:
        return
    if backend == "opencv":
        capture.release()


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


def _safe_worker_count(allocated_cpus, vid_shape, reader_count):
    h, w = vid_shape[0], vid_shape[1]
    # Reader model: Queue_raw holds ~2 raw frames per worker + ~2 frames of processing state per worker.
    # Raw BGR decoded frame = h*w*3 bytes.
    bytes_per_worker = h * w * 3 * 4
    job_mem = _job_memory_limit_bytes()
    usable = job_mem * 0.80
    reader_backend = _reader_backend_request()
    decord_threads = _decord_thread_count()
    reader_cores = reader_count
    if reader_backend in ("auto", "decord", "decord-gpu") and decord_threads > 1:
        reader_cores = max(reader_count, reader_count * decord_threads)
    # Reserve cores for the reader/decode pipeline and one assignment/output process.
    reserved_cores = reader_cores + 1
    max_from_cpus = max(1, allocated_cpus - reserved_cores)
    mem_based = max(1, int(usable / bytes_per_worker))
    count = min(max_from_cpus, mem_based)
    _tlog(
        "worker cap: cpu_limit={} mem_limit={} job_mem={:.1f}GB per_worker_est={:.0f}MB "
        "reader_backend={} reader_cores={} reserved_cores={} -> using {}".format(
            max_from_cpus,
            mem_based,
            job_mem / 1e9,
            bytes_per_worker / 1e6,
            reader_backend,
            reader_cores,
            reserved_cores,
            count,
        )
    )
    return count


def _send_worker_sentinels(Queue_raw, num_workers):
    for _ in range(num_workers):
        Queue_raw.put(None)


def _reader_finished(Queue_raw, num_workers, reader_name, readers_remaining=None, readers_lock=None):
    if readers_remaining is None:
        _send_worker_sentinels(Queue_raw, num_workers)
        return

    with readers_lock:
        readers_remaining.value -= 1
        remaining = readers_remaining.value

    if remaining == 0:
        _tlog("{} done; all readers complete; sending {} sentinels".format(reader_name, num_workers))
        _send_worker_sentinels(Queue_raw, num_workers)
    else:
        _tlog("{} done; waiting for {} reader(s)".format(reader_name, remaining))


def _frame_reader(Queue_raw, free_slots, shm_names, frame_shape, Vid, frame_values, num_workers, reader_name="reader", readers_remaining=None, readers_lock=None):
    """Opens the video once, writes decoded frames into shared memory slots,
    and feeds (frame_number, slot_index) pairs to Queue_raw.
    Sends num_workers None sentinels after the last reader finishes."""
    os.environ.pop('LD_PRELOAD', None)

    shm_blocks = [SharedMemory(name=n, create=False) for n in shm_names]

    all_frames = np.asarray(frame_values)
    total_frames = len(all_frames)

    Which_part = 0
    if len(Vid.Fusion) > 1:
        Which_part = [i for i, fu in enumerate(Vid.Fusion) if fu[0] <= round(all_frames[0])][-1]

    requested_backend = _reader_backend_request()
    decord_batch = _decord_batch_size(len(shm_names))
    opencv_seek_gap = _opencv_seek_gap()
    cap_pos = 0
    capture_backend = "image" if Vid.type != "Video" else None
    capture = None

    _t_grab = _t_decode = _t_slot_wait = _t_shm_write = 0.0
    _n = 0
    _REPORT_EVERY = 200
    _reader_start = time.perf_counter()

    _tlog("{} starting: total_frames={} num_workers={} n_slots={} backend_request={} decord_batch={}".format(
        reader_name, total_frames, num_workers, len(shm_names), requested_backend, decord_batch))

    def _open_part(part):
        if Vid.type != "Video":
            return "image", None
        backend, cap = _open_video_capture(Vid.Fusion[part][1], requested_backend)
        _tlog("{} opened segment={} backend={} file={}".format(
            reader_name, part + 1, backend, os.path.basename(Vid.Fusion[part][1])))
        return backend, cap

    def _write_frame(frame, image, color_space):
        nonlocal _t_slot_wait, _t_shm_write, _n
        nonlocal _t_grab, _t_decode

        _t0 = time.perf_counter()
        slot_idx = free_slots.get()
        _t_slot_wait += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        np.copyto(np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_blocks[slot_idx].buf), image)
        _t_shm_write += time.perf_counter() - _t0

        Queue_raw.put((frame, slot_idx, color_space))

        _n += 1
        if _n % _REPORT_EVERY == 0:
            n = _REPORT_EVERY
            elapsed = time.perf_counter() - _reader_start
            overall_fps = _n / elapsed
            _tlog(
                "{} frame={} ({}/{}) avg/{}: "
                "grab={:.2f}ms decode={:.2f}ms slot_wait={:.2f}ms shm_write={:.2f}ms "
                "overall_fps={:.0f} queue_size={} backend={}".format(
                    reader_name, frame, _n, total_frames, n,
                    _t_grab / n * 1000,
                    _t_decode / n * 1000,
                    _t_slot_wait / n * 1000,
                    _t_shm_write / n * 1000,
                    overall_fps,
                    Queue_raw.qsize(),
                    capture_backend,
                )
            )
            _t_grab = _t_decode = _t_slot_wait = _t_shm_write = 0.0

    try:
        if Vid.type == "Video":
            capture_backend, capture = _open_part(Which_part)

        frame_index = 0
        while frame_index < total_frames:
            frame_f = all_frames[frame_index]
            frame = round(frame_f)

            if len(Vid.Fusion) > 1 and Which_part < len(Vid.Fusion) - 1 and frame >= Vid.Fusion[Which_part + 1][0]:
                while len(Vid.Fusion) > 1 and Which_part < len(Vid.Fusion) - 1 and frame >= Vid.Fusion[Which_part + 1][0]:
                    Which_part += 1
                if Vid.type == "Video":
                    _close_video_capture(capture_backend, capture)
                    capture_backend, capture = _open_part(Which_part)
                    cap_pos = 0

            if Vid.type == "Video" and capture_backend in ("decord", "decord-gpu"):
                batch_frames = []
                batch_local_frames = []
                batch_start_index = frame_index
                while frame_index < total_frames and len(batch_frames) < decord_batch:
                    batch_frame = round(all_frames[frame_index])
                    if (
                        len(Vid.Fusion) > 1
                        and Which_part < len(Vid.Fusion) - 1
                        and batch_frame >= Vid.Fusion[Which_part + 1][0]
                    ):
                        break
                    batch_frames.append(batch_frame)
                    batch_local_frames.append(batch_frame - Vid.Fusion[Which_part][0])
                    frame_index += 1

                if not batch_frames:
                    continue

                _t0 = time.perf_counter()
                try:
                    images = capture.get_batch(batch_local_frames).asnumpy()
                    _t_decode += time.perf_counter() - _t0
                except Exception as exc:
                    if requested_backend == "auto":
                        _tlog("reader backend {} failed during decode at frame {}; switching to opencv ({})".format(
                            capture_backend, batch_frames[0], exc))
                        frame_index = batch_start_index
                        requested_backend = "opencv"
                        _close_video_capture(capture_backend, capture)
                        capture_backend, capture = _open_part(Which_part)
                        cap_pos = 0
                        continue
                    raise

                for batch_frame, image in zip(batch_frames, images):
                    _write_frame(batch_frame, image, "RGB")
                continue

            if Vid.type == "Video":
                local_frame = frame - Vid.Fusion[Which_part][0]
                if (
                    capture_backend == "opencv"
                    and local_frame > 0
                    and (cap_pos == 0 or local_frame - cap_pos > opencv_seek_gap)
                ):
                    _t0 = time.perf_counter()
                    capture.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                    _t_grab += time.perf_counter() - _t0
                    cap_pos = local_frame
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
                color_space = "BGR"
            else:
                _t0 = time.perf_counter()
                image = cv2.imread(os.path.join(Vid.Fusion[Which_part][1], Vid.img_list[frame - Vid.Fusion[Which_part][0]]))
                _t_decode += time.perf_counter() - _t0
                if image is None:
                    raise RuntimeError("Reader could not read image frame {}".format(frame))
                color_space = "BGR"

            _write_frame(frame, image, color_space)
            frame_index += 1

        elapsed = time.perf_counter() - _reader_start
        _tlog("{} done: {} frames in {:.1f}s ({:.0f} fps avg)".format(
            reader_name, total_frames, elapsed, total_frames / elapsed if elapsed > 0 else 0))
        _reader_finished(Queue_raw, num_workers, reader_name, readers_remaining, readers_lock)

    finally:
        _close_video_capture(capture_backend, capture)
        for shm in shm_blocks:
            shm.close()

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
    _configure_reader_defaults(allocated_cpus)
    total_frame_count = len(np.arange(start, end + one_every, one_every))
    requested_readers = _reader_process_count(allocated_cpus, total_frame_count)
    frame_chunks = _split_frame_values(start, end, one_every, requested_readers)
    reader_count = len(frame_chunks)
    nb_cpu_extract_treat = _safe_worker_count(allocated_cpus, Vid.shape, reader_count)
    _tlog("tracking worker config: color_mode={} stabilization={} direct_gray={}".format(
        Vid.Track[1][10][0], Vid.Stab[0], Vid.Track[1][10][0] == 0 and not Vid.Stab[0]))
    _tlog("CPUs allocated: {} | total on node: {} | reader processes: {} | worker processes to spawn: {}".format(
        allocated_cpus, multiprocessing.cpu_count(), reader_count, nb_cpu_extract_treat))
    if reader_count > 1:
        _tlog("reader split: window={} frames counts={}".format(
            _reader_window_size(), ",".join(str(len(chunk)) for chunk in frame_chunks)))
    Nb_images_processed=multiprocessing.Value("i",0)

    _shm_H, _shm_W = Vid.shape[0], Vid.shape[1]
    _frame_shape = (_shm_H, _shm_W, 3)
    _frame_nbytes = _shm_H * _shm_W * 3
    _N_SLOTS = nb_cpu_extract_treat * 3 + 4
    _shm_blocks = []
    _shm_names = []
    try:
        for _ in range(_N_SLOTS):
            _shm = SharedMemory(create=True, size=_frame_nbytes)
            _shm_blocks.append(_shm)
            _shm_names.append(_shm.name)
        _free_slots = multiprocessing.Queue()
        for i in range(_N_SLOTS):
            _free_slots.put(i)
        _tlog("shm pool: {} slots x {:.1f}MB = {:.0f}MB in /dev/shm".format(
            _N_SLOTS, _frame_nbytes / 1e6, _N_SLOTS * _frame_nbytes / 1e6))

        manager = multiprocessing.Manager()

        Processes = []

        Queue_raw = multiprocessing.Queue(maxsize=_N_SLOTS)

        #We create one queue per cpu (-1 as one cpu will be in charge of the tracking itself)
        Queues_cnt=multiprocessing.Queue(maxsize=100)

        if type=="fixed":
            Processes.append(multiprocessing.Process(name="assign-fixed", target=Function_assign_cnts_multi.Treat_cnts_fixed, args=(Queues_cnt, Nb_images_processed, Vid, Arenas, start, end, prev_row, To_save, portion, one_every, use_Kalman, head_tail)))
        elif type == "variable":
            keep_entrance = Params["Keep_entrance"]
            ID_kepts = manager.list([manager.list(sublist) for sublist in [[] for _ in Arenas]])
            Processes.append(multiprocessing.Process(name="assign-variable", target=Function_assign_cnts_multi.Treat_cnts_variable, args=(Queues_cnt, Nb_images_processed,Vid, Arenas, Main_Arenas_image, Main_Arenas_Bimage, start, end, ID_kepts, prev_row, To_save, portion, one_every, not keep_entrance, use_Kalman, head_tail)))

        if reader_count > 1:
            readers_remaining = multiprocessing.Value("i", reader_count)
            readers_lock = multiprocessing.Lock()
        else:
            readers_remaining = None
            readers_lock = None

        # Reader process(es): open video segments, feed raw frames to Queue_raw.
        for reader_idx, frame_chunk in enumerate(frame_chunks):
            reader_label = "reader-{}/{}".format(reader_idx + 1, reader_count) if reader_count > 1 else "reader"
            Processes.append(multiprocessing.Process(
                name=reader_label,
                target=_frame_reader,
                args=(
                    Queue_raw, _free_slots, _shm_names, _frame_shape, Vid,
                    frame_chunk, nb_cpu_extract_treat, reader_label,
                    readers_remaining, readers_lock,
                ),
            ))

        # Worker processes: pull raw frames, run processing pipeline, push contour batches.
        for process_ID in range(nb_cpu_extract_treat):
            Processes.append(multiprocessing.Process(name="prep-{}".format(process_ID), target=Function_prepare_images_multi.Image_modif, args=(Queues_cnt, Queue_raw, _free_slots, _shm_names, _frame_shape, Vid, Prem_image_to_show, mask, or_bright, process_ID)))

        os.environ.pop('LD_PRELOAD', None)
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
                    "{} pid={} exitcode={}".format(process.name, process.pid, process.exitcode)
                    for process in failed_processes
                )
                raise RuntimeError("Multiprocess tracking process failed ({})".format(failed_details))
            with Nb_images_processed.get_lock():
                parent.timer=(Nb_images_processed.value)/(Vid.Cropped[1][1]/one_every-Vid.Cropped[1][0]/one_every)

        failed_processes = [p for p in Processes if p.exitcode not in (None, 0)]
        if failed_processes:
            failed_details = ", ".join(
                "{} pid={} exitcode={}".format(process.name, process.pid, process.exitcode)
                for process in failed_processes
            )
            raise RuntimeError("Multiprocess tracking process failed ({})".format(failed_details))
    finally:
        for _shm in _shm_blocks:
            try:
                _shm.close()
                _shm.unlink()
            except Exception:
                pass

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

