from tkinter import *
import os
import cv2
import PIL.Image, PIL.ImageTk
from AnimalTA.A_General_tools import Video_loader as VL, UserMessages, User_help, Color_settings, Class_change_vid_menu
import numpy as np
import pickle

class Inter_image(Frame):
    """This frame will appear when the user is not satisfied with the automatic background and wants to change it.
    It basically allow the user to select a color in the fram eand then to paint with this color.
    """
    def __init__(self, parent, img, **kwargs):
        Canvas.__init__(self, parent, **kwargs)
        self.config(**Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        self.config(bg="red")
        self.update()
        self.img=img

        self.Size = self.img.shape
        self.final_width = self.winfo_width()
        self.ratio = self.Size[1] / self.final_width
        
        self.zoom_sq = [0, 0, self.img.shape[1], self.img.shape[0]]  # If not, we show the cropped frames
        self.ZinSQ = [-1, ["NA", "NA"]]  # used to zoom in a particular area

        self.bind("<Configure>", self.afficher)

        self.zooming=False
        self.zoom_strength=1.25
        self.bind("<Control-B1-Motion>", self.Sq_Zoom_mov)
        self.bind("<B1-ButtonRelease>", lambda x: self.Zoom(event=x,Zin=True))
        self.bind("<Control-B3-ButtonRelease>", lambda x: self.Zoom(event=x,Zin=False))
        self.bind("<Button-1>", self.point)
        
        self.update()



    def Leave_zoom(self, event):
        #Stop sooming
        self.zooming = False
        if self.painting:
            self.painting=False


    def Sq_Zoom_beg(self, event):
        event_t_x = int( self.ratio * (event.x - (self.winfo_width()-self.shape[1])/2)) + self.zoom_sq[0]
        event_t_y = int( self.ratio * (event.y - (self.winfo_height()-self.shape[0])/2)) + self.zoom_sq[1]
        self.ZinSQ=[0,[event_t_x,event_t_y],[event.x,event.y]]
        self.delete("Rect")

    def Sq_Zoom_mov(self,event):
        self.delete("Rect")
        event_t_x = int( self.ratio * (event.x - (self.winfo_width()-self.shape[1])/2)) + self.zoom_sq[0]
        event_t_y = int( self.ratio * (event.y - (self.winfo_height()-self.shape[0])/2)) + self.zoom_sq[1]
        zoom_sq = [min(self.ZinSQ[1][0], event_t_x), min(self.ZinSQ[1][1], event_t_y), max(self.ZinSQ[1][0], event_t_x),max(self.ZinSQ[1][1], event_t_y)]
        if (zoom_sq[2] - zoom_sq[0]) > 50 and (zoom_sq[3] - zoom_sq[1])>50 and event_t_x>=0 and event_t_x<=self.Size[1] and event_t_y>=0 and event_t_y<=self.Size[0] and self.ZinSQ[1][0]>=0 and self.ZinSQ[1][0]<=self.Size[1] and self.ZinSQ[1][1]>=0 and self.ZinSQ[1][1]<=self.Size[0]:
            self.create_rectangle(self.ZinSQ[2][0], self.ZinSQ[2][1], event.x, event.y, outline="white", tags="Rect")
        else:
            self.create_rectangle(self.ZinSQ[2][0], self.ZinSQ[2][1], event.x, event.y, outline="red", tags="Rect")
        self.create_rectangle(self.ZinSQ[2][0],self.ZinSQ[2][1],event.x,event.y, dash=(3,3), outline="black", tags="Rect")

        if self.ZinSQ[0]>=0:
            self.ZinSQ[0]+=1


    def Zoom(self, event, Zin=True):
        '''When the user hold <Ctrl> and click on the frame, zoom on the image.
        If <Ctrl> and right click, zoom out'''
        self.painting=False
        if not bool(event.state & 0x1) and bool(event.state & 0x4):
            self.new_zoom_sq = [0, 0, self.Size[1], self.Size[0]]
            event.x = int( self.ratio * (event.x - (self.winfo_width()-self.shape[1])/2)) + self.zoom_sq[0]
            event.y = int( self.ratio * (event.y - (self.winfo_height()-self.shape[0])/2)) + self.zoom_sq[1]
            PX = event.x / self.Size[1]
            PY = event.y / self.Size[0]

            if self.ZinSQ[0]<3:
                if Zin:
                    new_total_width = self.Size[1] / self.ratio * self.zoom_strength
                    new_total_height = self.Size[0] / self.ratio * self.zoom_strength
                else:
                    new_total_width = self.Size[1] / self.ratio / self.zoom_strength
                    new_total_height = self.Size[0] / self.ratio / self.zoom_strength


                if new_total_width>self.winfo_width():
                    missing_px=new_total_width - (self.winfo_width()-5)
                    ratio_old_new=self.Size[1]/new_total_width
                    self.new_zoom_sq[0] = int(PX * missing_px*ratio_old_new)
                    self.new_zoom_sq[2] = int(self.Size[1] - ((1 - PX) * missing_px*ratio_old_new))

                if new_total_height>self.winfo_height():
                    missing_px=new_total_height - (self.winfo_height()-5)
                    ratio_old_new=self.Size[0]/new_total_height
                    self.new_zoom_sq[1] = int(PY * missing_px*ratio_old_new)
                    self.new_zoom_sq[3] = int(self.Size[0] - ((1 - PY) * missing_px*ratio_old_new))

                if self.new_zoom_sq[3]-self.new_zoom_sq[1] > 50 and self.new_zoom_sq[2]-self.new_zoom_sq[0]>50:
                    self.zoom_sq = self.new_zoom_sq
                    self.update_ratio()
                    self.afficher()

            elif event.x>=0 and event.x<=self.Size[1] and event.y>=0 and event.y<=self.Size[0] and self.ZinSQ[1][0]>=0 and self.ZinSQ[1][0]<=self.Size[1] and self.ZinSQ[1][1]>=0 and self.ZinSQ[1][1]<=self.Size[0]:
                zoom_sq = [min(self.ZinSQ[1][0], event.x), min(self.ZinSQ[1][1], event.y) , max(self.ZinSQ[1][0], event.x), max(self.ZinSQ[1][1], event.y)]
                if (zoom_sq[2] - zoom_sq[0]) > 50 and (zoom_sq[3] - zoom_sq[1])>50:
                    self.zoom_sq=zoom_sq
                    self.update_ratio()
                    self.afficher()
                self.ZinSQ = [-1, ["NA", "NA"]]

            self.zooming = False
            if self.painting:
                self.painting = False
            self.delete("Rect")


    def update_ratio(self, *args):
        '''Calculate the ratio between the original size of the video and the displayed image'''
        self.ratio=max((self.zoom_sq[2]-self.zoom_sq[0])/self.winfo_width(),(self.zoom_sq[3]-self.zoom_sq[1])/self.winfo_height())


    def point(self, event):
        #When the image is clicked, we begin to paint over it
        if not bool(event.state & 0x1) and bool(event.state & 0x4):
            self.Sq_Zoom_beg(event)
        

    def afficher(self, *arg):
        #Change the displayed image
        self.update_ratio()
        self.final_width=int(self.Size[1]/self.ratio)

        empty_back=np.copy(self.img)
        self.shape = empty_back.shape
        if not self.Size==empty_back.shape:
            self.Size = empty_back.shape
            self.zoom_sq = [0, 0, self.Size[1], self.Size[0]]  # If not, we show the cropped frames

        self.image_to_show=empty_back[self.zoom_sq[1]:self.zoom_sq[3],self.zoom_sq[0]:self.zoom_sq[2]]

        width=int((self.zoom_sq[2]-self.zoom_sq[0])/self.ratio)
        height=int((self.zoom_sq[3]-self.zoom_sq[1])/self.ratio)


        TMP_image_to_show2 = cv2.resize(self.image_to_show,(width, height))
        self.shape= TMP_image_to_show2.shape

        self.image_to_show3 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(TMP_image_to_show2))
        self.can_import = self.create_image((self.winfo_width() - self.shape[1]) / 2,
                                                         (self.winfo_height() - self.shape[0]) / 2,
                                                         image=self.image_to_show3, anchor=NW)

        self.config(height=self.shape[1], width=self.shape[0])
        self.itemconfig(self.can_import, image=self.image_to_show3)



