from tkinter import *
import os
from AnimalTA.A_General_tools import UserMessages, Color_settings
import time

class Loading(Frame):
    #A loading bar
    def __init__(self, parent, text=None, grab=True):
        Frame.__init__(self, parent, bd=5)
        self.config(
            **Color_settings.My_colors.Frame_Base,
            bd=2,
            highlightthickness=1,
            relief="ridge",
        )
        self.parent=parent
        self.use_grab = grab
        self._has_grab = False
        self._using_place = False

        #Import messsages
        self.Language = StringVar()
        f = open(UserMessages.resource_path(os.path.join("AnimalTA","Files","Language")), "r", encoding="utf-8")
        self.Language.set(f.read())
        self.LanguageO = self.Language.get()
        f.close()

        self.Messages = UserMessages.Mess[self.Language.get()]

        self.loading_canvas = Frame(self, **Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        self.loading_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        Grid.columnconfigure(self, 0, weight=1)
        Grid.rowconfigure(self, 0, weight=1)
        Grid.columnconfigure(self.loading_canvas, 0, weight=0)
        Grid.columnconfigure(self.loading_canvas, 1, weight=1)

        if text is None:
            text=self.Messages["Loading"]

        self.loading_state = Label(self.loading_canvas, text=text,**Color_settings.My_colors.Label_Base)
        self.loading_state.grid(row=0, column=0, padx=(0, 12), sticky="w")

        self.loading_bar = Canvas(self.loading_canvas, height=10, **Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        self.loading_bar.create_rectangle(0, 0, 400, self.loading_bar.cget("height"), fill=Color_settings.My_colors.list_colors["Loading_before"])
        self.loading_bar.grid(row=0, column=1, sticky="ew")
        self._set_initial_visibility()

    def _set_initial_visibility(self):
        self.update_idletasks()
        self.lift()
        try:
            self.winfo_toplevel().lift()
        except Exception:
            pass

        if self.use_grab:
            try:
                self.grab_set()
                self._has_grab = True
            except Exception:
                self._has_grab = False

    def grid(self, *args, **kwargs):
        if self.use_grab and not args and not kwargs:
            self._using_place = True
            self.place(relx=0.5, rely=0.5, anchor=CENTER)
            self._set_initial_visibility()
            return None

        if not self.use_grab and not args and not kwargs:
            kwargs = {"sticky": "ew", "padx": 10, "pady": 4}

        self._using_place = False
        result = super().grid(*args, **kwargs)
        self._set_initial_visibility()
        return result

    def destroy(self):
        if self._has_grab:
            try:
                self.grab_release()
            except Exception:
                pass
            self._has_grab = False
        super().destroy()

    def show_loading_while(self, thread, fps=30):
        cur_rot = 0
        while thread.is_alive():
            self.loading_bar.delete('all')
            heigh = self.loading_bar.cget("height")
            self.loading_bar.create_rectangle(0, 0, 400, heigh,
                                              fill=Color_settings.My_colors.list_colors["Loading_before"])
            self.loading_bar.create_rectangle(max(0, cur_rot * 400 - 50), 0, min(cur_rot * 400, 450), heigh,
                                              fill=Color_settings.My_colors.list_colors["Loading_after"])
            self.loading_bar.update_idletasks()
            self.update()

            cur_rot = 0 if cur_rot > 1.25 else cur_rot + 0.025
            time.sleep(1 / fps)


    def show_load(self, prop):
        if prop>=0:
            #Show the progress of the conversion process
            self.loading_bar.delete('all')
            heigh=self.loading_bar.cget("height")
            self.loading_bar.create_rectangle(0, 0, 400, heigh, fill=Color_settings.My_colors.list_colors["Loading_before"])
            self.loading_bar.create_rectangle(0, 0, prop*400, heigh, fill=Color_settings.My_colors.list_colors["Loading_after"])
            self.loading_bar.update_idletasks()
            self.update()
