import time
from AnimalTA.D_Tracking_process import Do_the_track_multi, Do_the_track
import multiprocessing
import cv2
import os
from AnimalTA.A_General_tools import UserMessages
from AnimalTA import compat
import pickle


#We determine whether it is better to use multiprocessing method or not:
def _estimate_opencv_capture_time(video_path, one_every):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None

    start_time = time.time()
    grabbed_index = -1
    for frame_index in range(0, int(one_every * 5) + 1, int(one_every)):
        while grabbed_index < frame_index:
            if not capture.grab():
                capture.release()
                return None
            grabbed_index += 1
        ret, frame = capture.retrieve()
        if not ret or frame is None:
            capture.release()
            return None

    elapsed = time.time() - start_time
    capture.release()
    return elapsed


def Choose_method(parent, Vid, folder, type, head_tail):
    parent.timer = 0
    parent.show_load()



    Param_file = UserMessages.settings_file_path()
    with open(Param_file, 'rb') as fp:
        Params = pickle.load(fp)
        Low_Priority = Params["Low_priority"]

    duration=(Vid.Cropped[1][1]-Vid.Cropped[1][0])/(Vid.Frame_rate[0] / Vid.Frame_rate[1])
    if duration < 2000 or Vid.Back[0] == 2 or Low_Priority:  # Video beginning (after crop)
        method=0
    else:
        #We run the analysis of 5 frames with no multiprocess and decord reader
        res_normal=Do_the_track.Do_tracking(parent, Vid, folder, type, portion=False, prev_row=None, arena_interest=None, test=True)

        parent.timer = 0.0001
        parent.show_load()

        one_every = Vid.Frame_rate[0] / Vid.Frame_rate[1]

        # We look at the time needed to load five frames using opencv
        if Vid.type!="Video":
            method=1 #If we have an image sequence, then the best strategy is to do multithreading (in case of video not always beneficial as decord does not work with multiprocessing)
        else:
            res_multi = _estimate_opencv_capture_time(Vid.Fusion[0][1], one_every)

            parent.timer = 0.0002
            parent.show_load()

            #Using multiprocess is interesting only if the time win is enought to compensate slower oppening of the images + about 10 sec lost due to multithreading.
            if res_multi is not None and (res_normal/5)*duration > (res_multi/5)*duration+20:
                method=1
            else:
                method=0

    if Low_Priority:  # Video beginning (after crop)
        method=0


    if method==0:
        succeed = Do_the_track.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, test=False, head_tail=head_tail)
        return succeed
    else:
        try:
            succeed = Do_the_track_multi.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, head_tail=head_tail)
            return succeed
        except Exception as exc:
            compat.startup_debug("multiprocess tracking failed; retrying single-process tracker: {}: {}".format(type(exc).__name__, exc))
            succeed = Do_the_track.Do_tracking(parent=parent, Vid=Vid, type=type, folder=folder, test=False, head_tail=head_tail)
            return succeed

