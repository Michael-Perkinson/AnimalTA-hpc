from tkinter import *
from functools import partial
from AnimalTA.A_General_tools import Color_settings, UserMessages
from AnimalTA import compat


class Messagebox(Toplevel):
    """Display a simple modal question with one or more possible answers."""

    def __init__(self, parent, title="", message="", Possibilities=[], entry=False, hilight=[], **kwargs):
        Toplevel.__init__(self, parent)
        self.parent = parent
        self.result = None
        self.entry = entry

        self.Language = StringVar()
        f = open(UserMessages.resource_path("AnimalTA/Files/Language"), "r", encoding="utf-8")
        self.Language.set(f.read())
        f.close()
        self.Messages = UserMessages.Mess[self.Language.get()]

        if len(Possibilities) == 0:
            Possibilities = [self.Messages["Yes"], self.Messages["No"]]

        buttons_per_row = 1 if len(Possibilities) > 2 or any(len(str(poss)) > 18 for poss in Possibilities) else 2
        extra_rows = max(0, len(Possibilities) - 2) if buttons_per_row == 1 else max(0, (len(Possibilities) - 1) // 2)
        height = 200 + (30 if entry else 0) + (extra_rows * 42)
        self.geometry(f"420x{height}")
        compat.set_window_icon(self)
        compat.set_toolwindow(self)
        self.title(title)
        self.update_idletasks()
        self.config(**Color_settings.My_colors.Frame_Base, bd=0, highlightthickness=0)
        self.grab_set()

        Grid.rowconfigure(self, 0, weight=1)
        Grid.rowconfigure(self, 1, weight=1)
        Grid.rowconfigure(self, 2, weight=1, minsize=2)
        for column in range(max(1, buttons_per_row)):
            Grid.columnconfigure(self, column, weight=1)

        row = 0
        Label(
            self,
            text=message,
            height=5,
            **Color_settings.My_colors.Label_Base,
            wraplength=320,
            justify=CENTER,
        ).grid(row=row, column=0, columnspan=max(1, buttons_per_row), sticky="nsew")
        row += 1

        if self.entry:
            self.entry_val = StringVar()
            Entry(self, textvariable=self.entry_val, **Color_settings.My_colors.Entry_Base).grid(
                row=row,
                column=0,
                columnspan=max(1, buttons_per_row),
                padx=4,
                pady=4,
                sticky="ew",
            )
            row += 1

        column = 0
        for count, poss in enumerate(Possibilities):
            if count in hilight:
                if column != 0:
                    row += 1
                    column = 0
                button = Button(
                    self,
                    text=poss,
                    command=partial(self.return_val, count),
                    **Color_settings.My_colors.Button_Base,
                    padx=10,
                    pady=5,
                )
                button.grid(
                    row=row,
                    column=0,
                    columnspan=max(1, buttons_per_row),
                    padx=2,
                    pady=2,
                    sticky="ew",
                )
                row += 1
                continue

            button = Button(
                self,
                text=poss,
                command=partial(self.return_val, count),
                **Color_settings.My_colors.Button_Base,
                padx=10,
                pady=5,
            )
            button.grid(row=row, column=column, padx=2, pady=2, sticky="ew")
            column += 1
            if column >= buttons_per_row:
                row += 1
                column = 0

    def return_val(self, val):
        if self.entry:
            if self.entry_val.get() != "":
                self.result = [val, self.entry_val.get()]
            else:
                return
        else:
            self.result = val
        self.grab_release()
        self.destroy()
