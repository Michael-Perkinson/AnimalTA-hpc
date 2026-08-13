from tkinter import *
from AnimalTA.A_General_tools import Color_settings

class Pers_Scroll(Canvas):
    '''
    This Class is a personalized scrollbar/timeline. It is part of the Video Reader and allow to view a specific frame of the video, to move between frames etc.
    '''
    def __init__(self, parent, container, width=800, ecart=0, show_cropped=True, **kw):
            Canvas.__init__(self, parent, kw)
            self.list_colors = Color_settings.My_colors.list_colors
            self.Top=container #Here the Video reader
            self.Video=self.Top.Vid#The video we are interested in
            self.parent=parent#The frame widget of the Video reader within which this scrollbar will appear
            self.decalage=25#Esthetical point
            self.size_hide=40#Esthetical point
            self.config(width=width, height=50, borderwidth=0,**Color_settings.My_colors.Frame_Base, highlightthickness=0)

            self.hide_crop=False

            self.fr_rate=self.Video.Frame_rate[1]
            self.one_every=self.Video.Frame_rate[0]/ self.Video.Frame_rate[1]#Related to the frame rate, if it has been modified by the user: 1 frame evry self.one_every frames will be displayed. If not modified by user, this equals one.
            self.ecart=ecart*self.one_every #If the video has been cropped, how much supplementary frames do we show in the timebar? (same value for before/after cropped frames)

            self.crop_beg = round(self.Video.Cropped[1][0] / self.one_every)
            self.crop_end = round(self.Video.Cropped[1][1] / self.one_every)

            if self.ecart!=0 or show_cropped:
                self.debut=max(0,self.crop_beg-ecart)
                self.fin = min(self.Video.Frame_nb[1] - 1, self.crop_end + ecart)
            else:
                self.debut=0
                self.fin = self.Video.Frame_nb[1] - 1

            if show_cropped:
                self.to_show_sub=self.crop_beg
            else:
                self.to_show_sub=0

            # Keep the mapping usable for a one-frame timeline as well.
            self.video_length = max(self.fin - self.debut, 1)
            self.active_pos=self.crop_beg#the current position of the frame reader (implemented at the first frame of the video, after cropping)
            self._drag_after_id = None
            self._pending_drag_pos = None
            self._drag_debounce_ms = 16
            self.refresh()

            self.bind("<Motion>", self.afficher_frame)#Display a little square/info to tell the user what is the frame number under the mouse cursor
            self.bind("<Button-1>", self.activate_position)#Change the current frame
            self.bind("<B1-Motion>", self.move_position)#Change the current frame
            self.bind("<ButtonRelease-1>", self._finish_drag)
            self.bind("<MouseWheel>", self.on_mousewheel)
            self.bind("<Button-4>", self.on_mousewheel)
            self.bind("<Button-5>", self.on_mousewheel)

    def _widget_exists(self, widget):
        if widget is None:
            return False
        try:
            return bool(widget.winfo_exists())
        except (AttributeError, TclError):
            return False

    def _reader_is_ready(self):
        return (
            self._widget_exists(self)
            and self._widget_exists(self.parent)
            and not getattr(self, "_closed", False)
            and getattr(self.Top, "closed", False) is False
        )

    def _clamp_position(self, position):
        """Return a valid displayed-frame position for every timeline action."""
        return min(int(self.fin), max(int(self.debut), int(position)))

    def _cancel_drag_callback(self):
        callback_id = getattr(self, "_drag_after_id", None)
        if callback_id is not None:
            try:
                self.after_cancel(callback_id)
            except (AttributeError, TclError, ValueError):
                pass
        self._drag_after_id = None
        self._pending_drag_pos = None

    def _schedule_drag_update(self):
        """Coalesce rapid drag events into one Tk callback."""
        if not self._reader_is_ready():
            return
        if getattr(self, "_drag_after_id", None) is not None:
            return
        try:
            self._drag_after_id = self.after(self._drag_debounce_ms, self._flush_drag_update)
        except (AttributeError, TclError):
            # A teardown race can remove the Tk command between the readiness
            # check and after().  The reader will be closed by its owner.
            self._drag_after_id = None

    def _flush_drag_update(self):
        """Apply the latest drag position, if the reader is still alive."""
        self._drag_after_id = None
        position = getattr(self, "_pending_drag_pos", None)
        self._pending_drag_pos = None
        if position is None or not self._reader_is_ready():
            return

        self.active_pos = self._clamp_position(position)
        self.refresh()
        if self._reader_is_ready():
            self.Top.update_image(self.active_pos)

    def _finish_drag(self, _event=None):
        """Flush the last coalesced drag event when the button is released."""
        callback_id = getattr(self, "_drag_after_id", None)
        if callback_id is not None:
            try:
                self.after_cancel(callback_id)
            except (AttributeError, TclError, ValueError):
                pass
            self._drag_after_id = None
        if getattr(self, "_pending_drag_pos", None) is not None:
            self._flush_drag_update()

    def close_N_destroy(self):
        '''
        Destroy the Scrollbar, this is called when the Video Reader is destroyed
        '''
        self._closed = True
        self._cancel_drag_callback()
        if not self._widget_exists(self):
            return
        self.delete("all")
        self.unbind("<Motion>")
        self.unbind("<Button-1>")
        self.unbind("<B1-Motion>")
        self.unbind("<ButtonRelease-1>")
        self.unbind("<MouseWheel>")
        self.unbind("<Button-4>")
        self.unbind("<Button-5>")

    def on_mousewheel(self, event):
        """Move exactly one displayed frame for Windows, macOS, and X11 wheels."""
        if not self._reader_is_ready():
            return "break"

        cancel_drag_callback = getattr(self, "_cancel_drag_callback", None)
        if cancel_drag_callback is not None:
            cancel_drag_callback()

        button = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if button == 4 or (button not in (4, 5) and delta > 0):
            step = -1
        elif button == 5 or (button not in (4, 5) and delta < 0):
            step = 1
        else:
            return "break"

        clamp_position = getattr(self, "_clamp_position", None)
        if clamp_position is None:
            new_pos = min(self.fin, max(self.debut, int(self.active_pos) + step))
        else:
            new_pos = clamp_position(int(self.active_pos) + step)
        if new_pos != self.active_pos:
            self.active_pos = new_pos
            self.refresh()
            if self._reader_is_ready():
                self.Top.update_image(self.active_pos)

        # Prevent Treeview/Scale class bindings from adding their own,
        # platform-dependent scroll after this precise frame step.
        return "break"

    def refresh(self, *args):
        #Draw/Redraw the timeline, each time something is modified or that the containing widget size changes, the timeline is redraw.
        if not self._reader_is_ready():
            return
        self.active_pos = self._clamp_position(self.active_pos)
        width=self.parent.winfo_width()
        largscroll = max(width - 60, 1)
        largscan = max(width - 10, 1)
        self.delete("all")
        self.create_rectangle(0, 0, self.decalage, 20, fill=self.list_colors["Timeline_back"])
        self.create_rectangle(self.decalage-1, 0, largscroll + self.decalage+1, 20, fill=self.list_colors["Timeline_out"])
        self.create_rectangle(0, 20, largscan, 50, fill=self.list_colors["Timeline_back"], outline=self.list_colors["Timeline_back"])
        self.create_text(self.decalage, 27, fill=self.list_colors["Fg_Timeline"], font="Times 10 bold", text=self.debut-self.to_show_sub)
        self.create_text(largscroll + self.decalage, 27, fill=self.list_colors["Fg_Timeline"], font="Times 10 bold",text=self.fin-self.to_show_sub)
        self.create_text(largscroll + self.decalage, 40, fill=self.list_colors["Fg_Timeline"], font="Times 10 bold",text=str(round((self.video_length-self.to_show_sub)/self.fr_rate,2))+" s")

        if not self.hide_crop:
            self.create_rectangle((self.crop_beg-self.debut) / self.video_length * largscroll + self.decalage - 2, 0,(self.crop_beg-self.debut) / self.video_length * largscroll + self.decalage + 2, 20, fill=self.list_colors["Timeline_deb"],outline="", width=2)
            self.create_rectangle((self.crop_beg-self.debut) / self.video_length * largscroll + self.decalage - self.size_hide, 20,(self.crop_beg-self.debut) / self.video_length * largscroll + self.decalage + self.size_hide, 50, fill=self.list_colors["Timeline_back"],outline="", width=2)
            self.create_text((self.crop_beg-self.debut) / self.video_length * largscroll + self.decalage, 27, fill=self.list_colors["Timeline_deb"],font="Times 10 bold", text=self.crop_beg-self.to_show_sub)
            self.create_text((self.crop_beg-self.debut) / self.video_length * largscroll + self.decalage, 40, fill=self.list_colors["Timeline_deb"],font="Times 10 bold", text=str(round((self.crop_beg-self.to_show_sub)/self.fr_rate,2))+ " s")

            self.create_rectangle((self.crop_end-self.debut)  / self.video_length * largscroll + self.decalage - 2, 0,(self.crop_end-self.debut)  / self.video_length * largscroll + self.decalage + 2, 20, fill=self.list_colors["Timeline_end"],outline="", width=2)
            self.create_rectangle((self.crop_end-self.debut) / self.video_length * largscroll + self.decalage - self.size_hide, 20,(self.crop_end-self.debut)  / self.video_length * largscroll + self.decalage + self.size_hide, 50, fill=self.list_colors["Timeline_back"],outline="", width=2)
            self.create_text((self.crop_end-self.debut)  / self.video_length * largscroll + self.decalage, 27, fill=self.list_colors["Timeline_end"],font="Times 10 bold", text=self.crop_end - self.to_show_sub)
            self.create_text((self.crop_end-self.debut)  / self.video_length * largscroll + self.decalage, 40, fill=self.list_colors["Timeline_end"],font="Times 10 bold", text=str(round((self.crop_end - self.to_show_sub)/self.fr_rate,2))+ " s")
            self.create_rectangle((self.crop_beg-self.debut) / self.video_length * largscroll + self.decalage + 2, 0,(self.crop_end-self.debut) / self.video_length * largscroll + self.decalage - 2, 20, fill=self.list_colors["Timeline_in"],outline="", width=2)

        self.create_rectangle((self.active_pos-self.debut) / self.video_length * largscroll + self.decalage - 2, 0,(self.active_pos-self.debut) / self.video_length * largscroll + self.decalage + 2, 20,outline=self.list_colors["Fg_Timeline"], width=2)
        self.create_rectangle((self.active_pos-self.debut) / self.video_length * largscroll + self.decalage - self.size_hide, 20,
                              (self.active_pos-self.debut) / self.video_length * largscroll + self.decalage + self.size_hide, 50,
                              fill=self.list_colors["Timeline_back"], outline=self.list_colors["Timeline_back"])
        self.create_text((self.active_pos-self.debut) / self.video_length * largscroll + self.decalage, 27, fill=self.list_colors["Fg_Timeline"],font="Times 10 bold", text=self.active_pos - self.to_show_sub)
        self.create_text((self.active_pos-self.debut) / self.video_length * largscroll + self.decalage, 40, fill=self.list_colors["Fg_Timeline"],font="Times 10 bold", text=str(round((self.active_pos - self.to_show_sub)/self.fr_rate,2)) +" s")

    def afficher_frame(self,event):
        '''
        Draw a little rectangle and show text under the mouse cursor to indicates which of the frame would be selected if the user click.
        '''
        if not self._reader_is_ready():
            return
        width=self.parent.winfo_width()
        largscroll = max(width - 60, 1)
        if event.x>self.decalage and event.x<largscroll+self.decalage and event.y>0 and event.y<20:
            self.refresh()
            self.create_rectangle(event.x-self.size_hide,20,event.x+self.size_hide,50, fill=self.list_colors["Timeline_back"],outline=self.list_colors["Timeline_back"])
            self.create_text(event.x,27, fill=self.list_colors["Fg_Timeline"], font="Times 10 bold", text=(int((event.x-self.decalage) * self.video_length / largscroll)+ self.debut - self.to_show_sub))
            self.create_text(event.x,40, fill=self.list_colors["Fg_Timeline"], font="Times 10 bold", text=str(round(int(self.debut- self.to_show_sub+((event.x-self.decalage) * self.video_length / largscroll))/self.fr_rate,2))+" s")

    def activate_position(self,event):
        '''
        When timeline is clicked
        Change the current frame displayed on the Video Reader.
        '''
        if not self._reader_is_ready():
            return
        width=self.parent.winfo_width()
        largscroll = max(width - 60, 1)
        if event.x>self.decalage and event.x<largscroll+self.decalage and event.y>0 and event.y<20:
            self._cancel_drag_callback()
            self.active_pos=self._clamp_position(self.debut + int((event.x-self.decalage) * self.video_length / max(largscroll, 1)))
            self.refresh()
            if self._reader_is_ready():
                self.Top.update_image(self.active_pos)

    def move_position(self,event):
        '''
        When the user B1-Motion on the timeline.
        Change the current frame displayed on the Video Reader.
        '''
        if not self._reader_is_ready():
            return
        width=self.parent.winfo_width()
        largscroll = max(width - 60, 1)
        if event.x>self.decalage and event.x<(largscroll+self.decalage):
            new_pos = self._clamp_position(self.debut + round((event.x-self.decalage) * self.video_length / max(largscroll, 1)))
        elif event.x<self.decalage:
            new_pos = self._clamp_position(self.debut)

        elif event.x>(largscroll+self.decalage):
            new_pos = self._clamp_position(self.debut + round((max(largscroll, 1)) * self.video_length / max(largscroll, 1)))

        else:
            return

        self.active_pos = new_pos
        self._pending_drag_pos = new_pos
        self._schedule_drag_update()
