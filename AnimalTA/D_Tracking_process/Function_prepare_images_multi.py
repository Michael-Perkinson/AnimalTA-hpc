#Function_prepare_images_multi.py
import os
import time
import datetime as _dt
import sys
import cv2
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from AnimalTA.A_General_tools import Class_stabilise, UserMessages, image_utils


def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[prep_multi {ts}] {msg}", file=sys.stderr, flush=True)


def _process_frame_cpu(img, TMP_back, Vid, mask, kernel):
    """CPU fallback for background subtraction, threshold, mask, and morphology."""
    if Vid.Back[0] == 1 or Vid.Back[0] == 2:
        sub_mode = Vid.Track[1][10][1]
        if sub_mode == 0:
            img = cv2.absdiff(TMP_back, img)
        elif sub_mode == 1:
            img = cv2.subtract(TMP_back, img)
        else:
            img = cv2.subtract(img, TMP_back)

        if Vid.Track[1][10][2] == 1:
            img = image_utils.apply_relative_background(img, TMP_back)

        if Vid.Track[1][10][0] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, img = cv2.threshold(img, Vid.Track[1][0], 255, cv2.THRESH_BINARY)

    elif Vid.Back[0] == 0:
        if Vid.Track[1][10][0] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if Vid.Track[1][10][1] == 2:
            img = cv2.bitwise_not(img)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, Vid.Track[1][0], Vid.Track[1][11])

    if Vid.Mask[0]:
        img = cv2.bitwise_and(img, img, mask=mask)

    if Vid.Track[1][1] > 0:
        img = cv2.erode(img, kernel, iterations=Vid.Track[1][1])
    if Vid.Track[1][2] > 0:
        img = cv2.dilate(img, kernel, iterations=Vid.Track[1][2])

    return img


def Image_modif(Queue_cnts, Queue_raw, free_slots, shm_names, frame_shape, Vid, Prem_image_to_show, mask, or_bright, ID):
    os.environ.pop('LD_PRELOAD', None)

    shm_blocks = [SharedMemory(name=n, create=False) for n in shm_names]

    if Vid.Stab[0]:
        prev_pts = Vid.Stab[1]

    if Vid.Back[0] == 2:  # Dynamic background -- single-process only, guarded in Tracking_method_selection
        progressive_back = cv2.createBackgroundSubtractorMOG2(history=int(Vid.Track[1][10][3] * Vid.Frame_rate[1]),
                                                              varThreshold=Vid.Track[1][0], detectShadows=False)

    if Vid.Track[1][10][0] == 0:
        try:
            TMP_back = cv2.cvtColor(Vid.Back[1].copy(), cv2.COLOR_BGR2GRAY)
        except:
            TMP_back = Vid.Back[1].copy()
    else:
        try:
            TMP_back = cv2.cvtColor(Vid.Back[1].copy(), cv2.COLOR_GRAY2BGR)
        except:
            TMP_back = Vid.Back[1].copy()

    kernel = np.ones((3, 3), np.uint8)

    BATCH_SIZE = 10
    batch = []

    _t_preproc = _t_proc = _t_contours = 0.0
    _n_frames = 0
    _REPORT_EVERY = 500
    direct_gray = Vid.Track[1][10][0] == 0 and not Vid.Stab[0]
    if ID == 0 and direct_gray:
        _tlog("direct grayscale worker path enabled")

    try:
        while True:
            item = Queue_raw.get()
            if item is None:
                if batch:
                    Queue_cnts.put(batch)
                break

            if len(item) == 2:
                frame, slot_idx = item
                color_space = "BGR"
            else:
                frame, slot_idx, color_space = item
            # Copy frame out of shared memory and immediately return the slot to the reader.
            image = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_blocks[slot_idx].buf).copy()
            free_slots.put(slot_idx)

            _t0 = time.perf_counter()
            if color_space not in ("BGR", "RGB"):
                raise RuntimeError("Unsupported reader color space: {}".format(color_space))

            if color_space == "BGR" and not direct_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            Timg = image

            if Vid.Rotation == 1:
                Timg = cv2.rotate(Timg, cv2.ROTATE_90_CLOCKWISE)
            elif Vid.Rotation == 2:
                Timg = cv2.rotate(Timg, cv2.ROTATE_180)
            if Vid.Rotation == 3:
                Timg = cv2.rotate(Timg, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if Vid.Cropped_sp[0]:
                Timg = Timg[Vid.Cropped_sp[1][0]:Vid.Cropped_sp[1][2], Vid.Cropped_sp[1][1]:Vid.Cropped_sp[1][3]]

            if Vid.Stab[0]:
                Timg = Class_stabilise.find_best_position(Vid=Vid, Prem_Im=Prem_image_to_show, frame=Timg, show=False, prev_pts=prev_pts)

            if Vid.Track[1][10][0] == 0:
                if color_space == "BGR" and direct_gray:
                    Timg = cv2.cvtColor(Timg, cv2.COLOR_BGR2GRAY)
                else:
                    Timg = cv2.cvtColor(Timg, cv2.COLOR_RGB2GRAY)

            if Vid.Track[1][7]:
                Timg = image_utils.apply_brightness_correction(Timg, mask, or_bright, Vid.Mask[0])

            img = Timg
            _t_preproc += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            if Vid.Back[0] == 2:  # Dynamic background
                TMP_back = progressive_back.getBackgroundImage()
                if TMP_back is None:
                    TMP_back = img.copy()
                progressive_back.apply(img)

            img = _process_frame_cpu(img, TMP_back, Vid, mask, kernel)
            _t_proc += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            kept_cnts = filter_cnts(cnts, Vid)
            _t_contours += time.perf_counter() - _t0

            batch.append([frame, kept_cnts])
            _n_frames += 1

            if len(batch) >= BATCH_SIZE:
                Queue_cnts.put(batch)
                batch = []

            if _n_frames % _REPORT_EVERY == 0:
                n = _REPORT_EVERY
                _tlog(
                    f"worker={ID} frame={frame} avg/{n}f: "
                    f"preproc={_t_preproc/n*1000:.1f}ms "
                    f"proc={_t_proc/n*1000:.1f}ms "
                    f"contours={_t_contours/n*1000:.1f}ms "
                    f"total={(_t_preproc+_t_proc+_t_contours)/n*1000:.1f}ms"
                )
                _t_preproc = _t_proc = _t_contours = 0.0
    finally:
        for shm in shm_blocks:
            shm.close()



def filter_cnts(cnts, Vid):
    kept_cnts = []  # We make a list of the contours that fit in the limitations defined by user
    cnts_areas=[]
    kept_cnts2=[]
    for cnt in cnts:
        cnt_area = cv2.contourArea(cnt)
        if float(Vid.Scale[0]) > 0:  # We convert the area in units
            cnt_area = cnt_area * (1 / float(Vid.Scale[0])) ** 2

        # Filter the contours by size
        if cnt_area >= Vid.Track[1][3][0] and cnt_area <= Vid.Track[1][3][1]:
            kept_cnts.append(cnt)
            cnts_areas.append(cnt_area)

        # Contours are sorted by area
        kept_cnts2= [kept_cnts[idx] for idx in np.argsort(cnts_areas)[::-1]]
    return(kept_cnts2)
