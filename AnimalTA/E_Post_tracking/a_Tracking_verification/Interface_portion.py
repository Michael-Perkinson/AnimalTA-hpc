from tkinter import *
from tkinter import messagebox
import threading
import cv2
import numpy as np
from AnimalTA.D_Tracking_process import Do_the_track
from AnimalTA.E_Post_tracking import Coos_loader_saver as CoosLS
from AnimalTA.A_General_tools import Class_Lecteur, UserMessages, Class_stabilise, Color_settings, Class_loading_Frame
from AnimalTA.C_Pretracking import Interface_back, Interface_arenas
from AnimalTA.C_Pretracking.a_Parameters_track import Interface_parameters_track


class _PortionProgress:
    """Progress/cancellation state shared with the background tracker."""

    def __init__(self):
        self._value = 0.0
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def get(self):
        with self._lock:
            return self._value

    def set(self, value):
        with self._lock:
            self._value = value

    def cancel(self):
        self._cancelled.set()

    def is_cancelled(self):
        return self._cancelled.is_set()


class Show(Frame):
    """This Frame is used to rerun the tracking over a portion of the video, to correct potential tracking mistakes.
    To this aim, the user can define temporary tracking parameters that will be used for this portion only."""
    def __init__(self, parent, boss, Vid, Video_liste, prev_row=None, Arena=None, First_frame=None, **kwargs):
        Frame.__init__(self, parent, bd=5, **kwargs)
        self.config(**Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        self.parent = parent
        self.boss = boss
        self.grid(sticky="nsew")
        self.parent.geometry("1050x620")
        self.Video_liste = Video_liste
        self.Vid=Vid
        self.boss.PortionWin.grab_set()
        self.prev_row=prev_row
        self.Arena=Arena

        if First_frame is None:
            self.First_frame = self.Vid.Cropped[1][0]
        else:
            self.First_frame=First_frame

        #Import language
        self.Language = StringVar()
        f = open(UserMessages.resource_path("AnimalTA/Files/Language"), "r", encoding="utf-8")
        self.Language.set(f.read())
        self.LanguageO = self.Language.get()
        f.close()
        self.tail_size=5

        Grid.columnconfigure(self.parent, 0, weight=1)  ########NEW
        Grid.rowconfigure(self.parent, 0, weight=1)  ########NEW
        Grid.columnconfigure(self, 0, weight=1)  ########NEW
        Grid.rowconfigure(self, 0, weight=1)  ########NEW

        self.Folder=self.Vid.Folder

        self.CheckVar = IntVar()
        self.ecart=10#Esthetical point, we add some frames out of the portion before and after so the user can have a context

        self.Messages = UserMessages.Mess[self.Language.get()]
        self.winfo_toplevel().title(self.Messages["Portion0"])

        #Where the options are displayed
        Right_part=Frame(self, **Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        Right_part.grid(row=0, column=1)

        self.User_help = Frame(Right_part, **Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        self.User_help.grid(row=0, column=0, sticky="new")
        self.Lab_help=Label(self.User_help, text=self.Messages["Portion11"], wraplength=300, **Color_settings.My_colors.Label_Base)
        self.Lab_help.grid()

        self.User_buttons = Frame(Right_part, **Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        self.User_buttons.grid(row=1, rowspan=3, column=0, sticky="sew")

        self.text_stab=StringVar()
        if self.Vid.Stab[0]:
            self.text_stab.set(self.Messages["Portion2"])
        else:
            self.text_stab.set(self.Messages["Portion3"])
        #Stabilisation
        self.B_change_stab=Button(self.User_buttons, textvariable=self.text_stab, command=self.change_stab, **Color_settings.My_colors.Button_Base)
        self.B_change_stab.grid(row=1,column=0, columnspan=2, sticky="ew")

        #Arenas definition
        self.B_change_mask=Button(self.User_buttons, text=self.Messages["Portion4"], command=self.change_mask, **Color_settings.My_colors.Button_Base)
        self.B_change_mask.grid(row=2,column=0, columnspan=2, sticky="ew")

        #Background
        self.B_change_back=Button(self.User_buttons, text=self.Messages["Portion5"], command=self.change_back, **Color_settings.My_colors.Button_Base)
        if self.Vid.Back[0]==1:
            self.B_change_back.grid(row=3,column=0, columnspan=2, sticky="ew")

        #Tracking parameters
        self.B_change_params = Button(self.User_buttons, text=self.Messages["Portion6"], command=self.change_params, **Color_settings.My_colors.Button_Base)
        self.B_change_params.grid(row=4,column=0, columnspan=2, sticky="ew")

        #Ruen the track
        self.B_redo_track = Button(self.User_buttons, text=self.Messages["Portion0"], command=self.redo_track, **Color_settings.My_colors.Button_Base)
        self.B_redo_track.config(background=Color_settings.My_colors.list_colors["Button_ready"], fg=Color_settings.My_colors.list_colors["Fg_Button_ready"])
        self.B_redo_track.grid(row=5,column=0, columnspan=2, sticky="ew")

        #Show the progression of the tracking
        self.load_frame = Class_loading_Frame.Loading(self.User_buttons, text=self.Messages["Loading"], grab=False)
        self.load_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=6, pady=6)

        self.B_validate_track = Button(self.User_buttons, text=self.Messages["Portion8"], command=self.validate_correction, **Color_settings.My_colors.Button_Base)
        self.B_validate_track.config(state="disable")
        self.B_validate_track.grid(row=7, column=0, sticky="nsew")

        self.B_cancel = Button(self.User_buttons, text=self.Messages["Cancel"], command=self.End_of_window, **Color_settings.My_colors.Button_Base)
        self.B_cancel.config(background=Color_settings.My_colors.list_colors["Cancel"],fg=Color_settings.My_colors.list_colors["Fg_Cancel"])
        self.B_cancel.grid(row=7, column=1, sticky="nsew")


        self.Coos, _ = CoosLS.load_coos(self.Vid, TMP=True, location=self)
        self.NB_ind = len(self.Vid.Identities)

        self.Vid_Lecteur = Class_Lecteur.Lecteur(self, self.Vid, ecart=5, First_frame=self.First_frame)

        self.Vid_Lecteur.grid(row=0, column=0, sticky="nsew")
        self.Scrollbar = self.Vid_Lecteur.Scrollbar
        self.Vid_Lecteur.canvas_video.update()
        self.Vid_Lecteur.update_image(self.Vid_Lecteur.to_sub+1)
        self.Vid_Lecteur.bindings()
        self.Scrollbar.refresh()

        self.parent.protocol("WM_DELETE_WINDOW", self.leave)

    def leave(self):
        #Close without applying any modifications to the original trackings
        if self._tracking_is_running():
            self._cancel_tracking()
            return
        self.Vid_Lecteur.proper_close()
        self.boss.PortionWin.grab_release()
        self.boss.PortionWin.destroy()
        self.boss.redo_Lecteur()
        self.destroy()

    def End_of_window(self):
        if self._tracking_is_running():
            self._cancel_tracking()
            return
        self.leave()

    def validate_correction(self):
        #Apply the new tracking results and close this Frame
        self.Vid_Lecteur.proper_close()
        self.boss.PortionWin.grab_release()
        self.boss.PortionWin.destroy()
        self.boss.change_for_corrected()
        self.destroy()

    def redo_track(self):
        #Re-run the tracking
        if self._tracking_is_running():
            return
        self.B_change_stab.config(state="disable")
        self.B_change_back.config(state="disable")
        self.B_change_mask.config(state="disable")
        self.B_change_params.config(state="disable")
        self.B_redo_track.config(state="disable")
        self.B_validate_track.config(state="disable")
        # Keep Cancel available while workers are running.  It requests a
        # cooperative stop and the window is only destroyed after both
        # workers have terminated.
        self.B_cancel.config(state="normal")
        self.Vid_Lecteur.proper_close()#We ensure the last decord is well closed
        self._tracking_progress = _PortionProgress()
        self._tracking_result = None
        self._tracking_error = None
        self._tracking_cancelled = False
        self._close_after_tracking = False
        tracking_type = "fixed" if self.Vid.Track[1][8] else "variable"

        def run_tracking():
            try:
                self._tracking_result = Do_the_track.Do_tracking(
                    self, self.Vid, self.Folder, type=tracking_type,
                    portion=True, prev_row=self.prev_row,
                    arena_interest=self.Arena, ref_frame=self.First_frame,
                    update_ui=False, progress=self._tracking_progress,
                )
            except BaseException as error:
                self._tracking_error = error

        self._tracking_thread = threading.Thread(target=run_tracking, daemon=True)
        self._tracking_thread.start()
        self._poll_tracking(tracking_type)

    def _tracking_is_running(self):
        thread = getattr(self, "_tracking_thread", None)
        return thread is not None and thread.is_alive()

    def _cancel_tracking(self):
        self._tracking_cancelled = True
        self._close_after_tracking = True
        progress = getattr(self, "_tracking_progress", None)
        if progress is not None:
            progress.cancel()
        Do_the_track.urgent_close(self.Vid)
        self.B_cancel.config(state="disable")

    def _poll_tracking(self, tracking_type):
        progress = self._tracking_progress
        self.timer = progress.get()
        self.load_frame.show_load(self.timer, process_events=False)
        if self._tracking_is_running():
            self.after(75, lambda: self._poll_tracking(tracking_type))
            return

        self._tracking_thread.join()
        error = self._tracking_error
        result = self._tracking_result
        cancelled = self._tracking_cancelled or progress.is_cancelled()
        self._tracking_thread = None
        self._tracking_progress = None

        succeeded = (
            not cancelled and error is None and
            ((tracking_type == "fixed" and result is True) or
             (tracking_type == "variable" and isinstance(result, tuple) and result[0] is True))
        )
        if succeeded:
            try:
                self.Coos, _ = CoosLS.load_coos(self.Vid, TMP=True, location=self)
            except BaseException as load_error:
                error = load_error
                succeeded = False

        if cancelled and getattr(self, "_close_after_tracking", False):
            self.leave()
            return

        try:
            self._restore_portion_reader()
        except BaseException as reader_error:
            if error is None:
                error = reader_error
            succeeded = False

        self._restore_portion_controls(succeeded)
        if error is not None and not cancelled:
            messagebox.showerror(
                self.Messages["Do_trackWarnT1"],
                self.Messages["Do_trackWarn1"].format(self.Vid.User_Name, error),
                parent=self,
            )

    def _restore_portion_controls(self, succeeded):
        for button in (self.B_change_stab, self.B_change_back,
                       self.B_change_mask, self.B_change_params,
                       self.B_redo_track, self.B_cancel):
            button.config(state="normal")
        self.B_validate_track.config(state="normal" if succeeded else "disable")
        if succeeded:
            self.B_validate_track.config(
                background=Color_settings.My_colors.list_colors["Validate"],
                fg=Color_settings.My_colors.list_colors["Fg_Validate"],
            )

    def _restore_portion_reader(self):
        self.Vid_Lecteur = Class_Lecteur.Lecteur(
            self, self.Vid, ecart=5, First_frame=self.First_frame
        )
        self.Vid_Lecteur.grid(row=0, column=0, sticky="nsew")
        self.Scrollbar = self.Vid_Lecteur.Scrollbar
        self.Vid_Lecteur.canvas_video.update()
        self.Vid_Lecteur.update_image(self.Vid_Lecteur.to_sub + 1)
        self.Vid_Lecteur.bindings()
        self.Scrollbar.refresh()

    def show_load(self):
        #Show the progress of the tracking process
        self.load_frame.loading_state.config(text=self.Messages["Loading"])
        self.load_frame.show_load(self.timer)

    def modif_image(self, img=[], aff=False, move=True, actual_pos=None, *args):
        #draw the target's potition and trajectories on the image
        if len(img)==0:
            new_img=np.copy(self.last_empty)
        else:
            self.last_empty = img
            new_img = np.copy(img)

        self.Vid_Lecteur.update_ratio()

        if self.Vid.Cropped[0]:
            to_remove = round(round((self.Vid.Cropped[1][0])/self.Vid_Lecteur.one_every))
        else:
            to_remove=0

        if self.Vid.Stab[0]:
            new_img = (Class_stabilise.find_best_position(Vid=self.Vid, Prem_Im=self.Vid_Lecteur.Prem_image_to_show, frame=new_img, show=False))

        for ind in range(self.NB_ind):
            color=self.Vid.Identities[ind][2]
            for prev in range(min(int(self.tail_size*self.Vid.Frame_rate[1]), int(self.Scrollbar.active_pos - to_remove))):
                if int(self.Scrollbar.active_pos - prev) > round(((self.Vid.Cropped[1][0])/self.Vid_Lecteur.one_every)) and int(self.Scrollbar.active_pos) <= round(self.Vid.Cropped[1][1]/self.Vid_Lecteur.one_every):
                    if self.Coos[ind,int(self.Scrollbar.active_pos - 1 - prev - to_remove),0] != -1000 and self.Coos[ind,int(self.Scrollbar.active_pos - prev - to_remove),0] != -1000 :
                        TMP_tail_1 = (int(self.Coos[ind,int(self.Scrollbar.active_pos - 1 - prev - to_remove),0]),
                                      int(self.Coos[ind,int(self.Scrollbar.active_pos - 1 - prev - to_remove),1]))

                        TMP_tail_2 = (int(self.Coos[ind,int(self.Scrollbar.active_pos - prev - to_remove),0]),
                                      int(self.Coos[ind,int(self.Scrollbar.active_pos - prev - to_remove),1]))

                        new_img = cv2.line(new_img, TMP_tail_1, TMP_tail_2, color, max(int(3*self.Vid_Lecteur.ratio),1))

            if self.Scrollbar.active_pos > round(((self.Vid.Cropped[1][0]-1)/self.Vid_Lecteur.one_every)) and self.Scrollbar.active_pos <= round(((self.Vid.Cropped[1][1]-1)/self.Vid_Lecteur.one_every)+1):
                center=self.Coos[ind,self.Scrollbar.active_pos - to_remove]
                if center[0]!=-1000:
                    if self.CheckVar.get()==int(ind):
                        new_img = cv2.circle(new_img, (int(center[0]), int(center[1])), radius=max(int(5*self.Vid_Lecteur.ratio),5), color=(255,255,255),thickness=-1)
                        new_img = cv2.circle(new_img, (int(center[0]), int(center[1])), radius=max(int(6*self.Vid_Lecteur.ratio),3), color=(0,0,0), thickness=-1)
                    new_img=cv2.circle(new_img,(int(center[0]),int(center[1])),radius=max(int(4*self.Vid_Lecteur.ratio),1),color=color,thickness=-1)

        self.Vid_Lecteur.afficher_img(new_img)

    def pressed_can(self, Pt, *args):
        pass

    def moved_can(self, Pt, Shift):
        pass

    def released_can(self, Pt):
        pass

    ##All the functions bellow open the Frame corresponding to the parameters to change
    #stab=stabilisation
    #mask=arena definition
    #back=background modification
    #params=tracking parameters
    def change_stab(self):
        self.Vid.Stab[0]=1-self.Vid.Stab[0]
        if self.Vid.Stab[0]:
            self.text_stab.set(self.Messages["Portion2"])
        else:
            self.text_stab.set(self.Messages["Portion3"])

    def change_mask(self):
        self.boss.PortionWin.grab_release()
        newWindow = Toplevel(self.parent.master)
        interface = Interface_arenas.Mask(parent=newWindow, boss=self.boss, main_frame=self, proj_pos=0, Video_file=self.Vid, portion=True)

    def change_back(self):
        self.boss.PortionWin.grab_release()
        newWindow = Toplevel(self.parent.master)
        interface = Interface_back.Background(parent=newWindow, boss=self.boss, main_frame=self, Video_file=self.Vid, portion=True, ref_frame=self.First_frame)

    def change_params(self):
        self.boss.PortionWin.grab_release()
        newWindow = Toplevel(self.parent.master)
        interface= Interface_parameters_track.Param_definer(parent=newWindow, boss=self.boss, main_frame=self, Video_file=self.Vid, portion=True, ref_frame=self.First_frame)

"""
root = Tk()
root.geometry("+100+100")
file_to_open="D:/Post-doc/Experiments/Group_composition/Shoaling/Videos_conv_cut/Track_by_mark/To_Roi/Tracked/14_12_01.btr"
with open(file_to_open, 'rb') as fp:
    print(file_to_open)
    Video_liste = pickle.load(fp)
interface = Stabilise(parent=root, boss="none", Video_liste=Video_liste)
root.mainloop()
"""
