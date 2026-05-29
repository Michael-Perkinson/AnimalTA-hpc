#Function_prepare_images_multi.py
import time
import datetime as _dt
import sys
import multiprocessing
import cv2
import numpy as np
from AnimalTA.A_General_tools import Class_stabilise, UserMessages, image_utils
from multiprocessing import Lock
import os


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


def send_cap_to(capture, capture_pos, frame):
    while capture_pos<=frame:
        capture_pos+=1
        if not capture.grab():
            return None
    return(capture_pos)




def Image_modif(Queue_cnts, Queue_frames, Vid, Prem_image_to_show, mask, or_bright, ID, lock_cnt, ):

    if Vid.Stab[0]:
        prev_pts = Vid.Stab[1]

    if Vid.Back[0] == 2:  # Dynamic background
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

    #We load the tasks
    first = True
    cap_pos = 0
    _t_load = _t_preproc = _t_gpu = _t_contours = 0.0
    _n_frames = 0
    _REPORT_EVERY = 500
    while True:
        all_cnts_kept=[]

        try:
            chunck_id, Images_to_treat = Queue_frames.get_nowait()
        except:
            break

        # simulate variable processing time

        if first:
            Which_part=0
            while len(Vid.Fusion) > 1 and Which_part < (len(Vid.Fusion) - 1) and Images_to_treat[0] >= (Vid.Fusion[Which_part + 1][0]):
                Which_part += 1

            try:
                if Vid.type=="Video":
                    capture = cv2.VideoCapture(Vid.Fusion[Which_part][1])
                    if not capture.isOpened():
                        raise RuntimeError("OpenCV could not open {}".format(Vid.Fusion[Which_part][1]))
            except Exception as e:
                raise RuntimeError("Error initializing video reader: {}".format(e))
            first=False

            # We first look at what is the best strategy (doing all intermediate frames or jumping from one to the other)

        for frame in Images_to_treat:  # Process frames respecting the defined frame rate
            frame = round(frame)

            # If we are at the limit between two videos:
            if len(Vid.Fusion) > 1 and Which_part < (len(Vid.Fusion) - 1) and frame >= (Vid.Fusion[Which_part + 1][0]):
                while len(Vid.Fusion) > 1 and Which_part < (len(Vid.Fusion) - 1) and frame >= (
                Vid.Fusion[Which_part + 1][0]):
                    Which_part += 1

                if Vid.type=="Video":
                    capture.release()
                    capture = cv2.VideoCapture(Vid.Fusion[Which_part][1])
                    cap_pos = 0

            _t0 = time.perf_counter()
            if Vid.type == "Video":
                cap_pos = send_cap_to(capture, cap_pos, frame - Vid.Fusion[Which_part][0])  # Set starting frame
                if cap_pos is None:
                    raise RuntimeError("OpenCV could not grab frame {} from {}".format(frame, Vid.Fusion[Which_part][1]))
                ret, image = capture.retrieve()
                if not ret or image is None:
                    raise RuntimeError("OpenCV could not retrieve frame {} from {}".format(frame, Vid.Fusion[Which_part][1]))
            else:
                image=cv2.imread(os.path.join(Vid.Fusion[Which_part][1], Vid.img_list[frame - Vid.Fusion[Which_part][0]]))
                if image is None:
                    raise RuntimeError("OpenCV could not read image frame {}".format(frame))
            _t_load += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            Timg = image

            if Vid.Rotation == 1:
                Timg = cv2.rotate(Timg, cv2.ROTATE_90_CLOCKWISE)
            elif Vid.Rotation == 2:
                Timg = cv2.rotate(Timg, cv2.ROTATE_180)
            if Vid.Rotation == 3:
                Timg = cv2.rotate(Timg, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if Vid.Cropped_sp[0]:
                Timg = Timg[Vid.Cropped_sp[1][0]:Vid.Cropped_sp[1][2],Vid.Cropped_sp[1][1]:Vid.Cropped_sp[1][3]]

            kernel = np.ones((3, 3), np.uint8)

            if Vid.Stab[0]:
                Timg = Class_stabilise.find_best_position(Vid=Vid, Prem_Im=Prem_image_to_show, frame=Timg, show=False, prev_pts=prev_pts)

            if Vid.Track[1][10][0]==0:
                Timg = cv2.cvtColor(Timg, cv2.COLOR_BGR2GRAY)

            if Vid.Track[1][7]:
                Timg = image_utils.apply_brightness_correction(Timg, mask, or_bright, Vid.Mask[0])

            img=Timg
            _t_preproc += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            # Backgroud and threshold
            if Vid.Back[0] == 2:  # Dynamic background
                TMP_back = progressive_back.getBackgroundImage()
                if TMP_back is None:
                    TMP_back=img.copy()
                progressive_back.apply(img)

            img = _process_frame_cpu(img, TMP_back, Vid, mask, kernel)
            _t_gpu += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            # Find contours:
            cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            kept_cnts=filter_cnts(cnts, Vid)
            _t_contours += time.perf_counter() - _t0

            _n_frames += 1
            if _n_frames % _REPORT_EVERY == 0:
                n = _REPORT_EVERY
                _tlog(
                    f"worker={ID} frame={frame} avg/{n}f: "
                    f"load={_t_load/n*1000:.1f}ms "
                    f"preproc={_t_preproc/n*1000:.1f}ms "
                    f"gpu={_t_gpu/n*1000:.1f}ms "
                    f"contours={_t_contours/n*1000:.1f}ms "
                    f"total={(_t_load+_t_preproc+_t_gpu+_t_contours)/n*1000:.1f}ms"
                )
                _t_load = _t_preproc = _t_gpu = _t_contours = 0.0

            all_cnts_kept.append([frame, kept_cnts])
        with lock_cnt:
            Queue_cnts.put(all_cnts_kept)



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
