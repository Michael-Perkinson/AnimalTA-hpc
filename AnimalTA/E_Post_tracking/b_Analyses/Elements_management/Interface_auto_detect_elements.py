from tkinter import *
from tkinter import ttk
from AnimalTA.E_Post_tracking.b_Analyses.Elements_management import Class_list_arenas
from AnimalTA.A_General_tools import Function_draw_arenas, UserMessages, Class_loading_Frame, Color_settings
import cv2
import numpy as np
import PIL
import copy
import os

class Auto_detect(Frame):
    """ This Frame displays a list of options to select the element of interest"""
    def __init__(self, parent, boss, **kwargs):
        Frame.__init__(self, parent, bd=5, **kwargs)
        self.possible_cols=[[],[],[]]#min and max hues/sat/val


    def update_vals(self,new_vals):
        self.possible_cols[0].append(new_vals[:, 0])
        self.possible_cols[1].append(new_vals[:, 1])
        self.possible_cols[2].append(new_vals[:, 2])

        print(np.min(self.possible_cols[0]))
        print(np.max(self.possible_cols[0]))