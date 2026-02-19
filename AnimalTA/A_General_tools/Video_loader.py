import threading
import decord
import cv2
import os


class Video_Loader():
    def __init__(self, Vid, File=None, is_crop=True, is_rotate=True, **kwargs):
        self.Vid=Vid#Which video of the project (can be different from the previous one in case of concatenated videos)
        self.is_crop=is_crop#Should we crop the video
        self.is_rotate=is_rotate
        self.File=File
        self.load_video(File)#Which video to load

    def __len__(self):
        return self.calculate_len()

    def __del__(self):
        if self.Vid.type=="Video":
            del self.capture
        del self

    def __getitem__(self, i):
        if self.Vid.type == "Video":
            if self.which_reader=="decord":
                frame=self.capture[i].asnumpy()
                self.capture.seek(0)

            else:
                try:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                    res, frame = self.capture.read()
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                except: #If the reader changed in the middle
                    return self[i]

        else:#If we are working with image sequence
            frame=cv2.imread(os.path.join(self.Vid.File_name, self.Vid.img_list[i]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.is_rotate and self.Vid.Rotation == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.is_rotate and self.Vid.Rotation == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.is_rotate and self.Vid.Rotation == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.is_crop and self.Vid.Cropped_sp[0]:
            frame = frame[self.Vid.Cropped_sp[1][0]:self.Vid.Cropped_sp[1][2],
                    self.Vid.Cropped_sp[1][1]:self.Vid.Cropped_sp[1][3]]

        return(frame)

    def seek(self):
        if self.which_reader=="decord":
            self.capture.seek(0)

    def calculate_len(self):
        if self.Vid.type == "Video":
            if self.which_reader=="decord":
                L=len(self.capture)
                self.capture.seek(0)
                return L
            else:
                return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return(len(self.Vid.img_list))


    def load_video(self, File):
        if self.Vid.type == "Video":
            self.which_reader = "cv2"
            self.capture = cv2.VideoCapture(File)
            Thread_load_vid =threading.Thread(target=self.load_decord_thread ,args=[File])
            Thread_load_vid.start()

    def load_decord_thread(self ,File):
        self.tmp_capture = decord.VideoReader(File)
        del self.capture
        self.which_reader ="decord"
        self.capture = self.tmp_capture
        del self.tmp_capture

        # if the video was not concatenated, we recalculate its real length (opencv is not precise in this task, so we correct this value using the decord library)
        # For concatenated videos, this step has been done at the moment of the concatenation
        if len(self.Vid.Fusion) < 2 :
            self.Vid.Frame_nb[0] = len(self.capture)
            self.capture.seek(0)
            self.Vid.Frame_nb[1] = self.Vid.Frame_nb[0] /  (self.Vid.Frame_rate[0] / self.Vid.Frame_rate[1])

            if not self.Vid.Cropped[0]:
                one_every = self.Vid.Frame_rate[0] / self.Vid.Frame_rate[1]
                self.Vid.Cropped[1][1] = round(round(self.Vid.Cropped[1][1] / one_every) * one_every)
