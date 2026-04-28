import cv2
import decord
from AnimalTA.A_General_tools import UserMessages, Video_loader as VL
from AnimalTA.D_Tracking_process import Function_prepare_images, Function_assign_cnts, security_settings_track, Treat_simgle_image
import numpy as np
import os
from tkinter import *
import threading
import time
import queue
import pickle
import sys
import re
import datetime as _dt

def _tlog(msg):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[track {ts}] {msg}", file=sys.stderr, flush=True)

'''
To improve the speed of the tracking, we will separate the work in 2 threads.
1. Image loading, and modifications (stabilization, light correction, greyscale...) until contours are get
2. Target assignment and data recording
'''


security_settings_track.stop_threads=False



def Do_tracking(parent, Vid, folder, type, portion=False, prev_row=None, arena_interest=None, test=False, head_tail=False, ref_frame=None):
    '''This is the main tracking function of the program.
    parent=container (main window)
    Vid=current video
    portion= True if it is a rerun of the tracking over a short part of the video (for corrections)
    prev_row=If portion is True, this correspond to the last known coordinates of the targets.
    '''
    security_settings_track.stop_threads=False
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

    AD=DoubleVar()

    if test:
        end=start+5
        result_container = {}
        Th_extract_cnts=threading.Thread(target=Function_prepare_images.Image_modif, args=(Security_break, Vid, start, end, one_every, Which_part, Prem_image_to_show, mask, or_bright, Extracted_cnts, AD, result_container, ))
        Th_extract_cnts.start()
        while Th_extract_cnts.is_alive():
            parent.show_load()
            time.sleep(0.1)
        Th_extract_cnts.join()
        result = result_container.get('result', None)
        return(result)

    else:
        Th_extract_cnts = threading.Thread(target=Function_prepare_images.Image_modif, args=(Security_break, Vid, start, end, one_every, Which_part, Prem_image_to_show, mask, or_bright, Extracted_cnts,AD))

        if type=="fixed":
            Th_associate_cnts=threading.Thread(target=Function_assign_cnts.Treat_cnts_fixed, args=(Vid, Arenas, start, end, prev_row, Extracted_cnts, Th_extract_cnts, To_save, portion, one_every, AD, use_Kalman, head_tail))
        elif type=="variable":
            keep_entrance=Params["Keep_entrance"]
            Th_associate_cnts=threading.Thread(target=Function_assign_cnts.Treat_cnts_variable, args=(Vid, Arenas, Main_Arenas_image, Main_Arenas_Bimage, start, end, prev_row, Extracted_cnts, Th_extract_cnts, To_save, portion, one_every, AD, not keep_entrance, use_Kalman, head_tail))

        Th_extract_cnts.start()
        Th_associate_cnts.start()


        while Th_extract_cnts.is_alive() or Th_associate_cnts.is_alive():
            parent.timer=(AD.get()-start)/(end + one_every - start)
            parent.show_load()
            time.sleep(0.05)

            overload = security_settings_track.check_memory_overload()#Avoid memory leak problems
            if overload==1:
                security_settings_track.activate_super_protection=True

            elif overload==0:
                security_settings_track.activate_protection=True

            elif overload==-1:
                security_settings_track.activate_protection=False
                security_settings_track.activate_super_protection = False
                Security_break.set()

        parent.timer = 1
        parent.show_load()

        Th_extract_cnts.join()
        del security_settings_track.capture
        Th_associate_cnts.join()


        if security_settings_track.stop_threads:
            if type=="fixed":
                return (False)
            elif type=="variable":
                return (False,0)
        else:
            if type == "fixed":
                return (True)
            elif type=="variable":
                return (True,security_settings_track.ID_kepts)


def urgent_close(Vid):
    security_settings_track.stop_threads = True

