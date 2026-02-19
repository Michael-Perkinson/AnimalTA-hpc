
from tkinter import *
from AnimalTA.A_General_tools import  Color_settings, UserMessages
import numpy as np
import PIL
import math
import cv2
from functools import partial
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import ttk
import os



def canvas_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def open_heatmap(master, Vid, selected, Duration, heatmap_raw, Vid_Lecteur):
    # Import messages
    Language = StringVar()
    f = open(UserMessages.resource_path(os.path.join("AnimalTA", "Files", "Language")), "r", encoding="utf-8")
    Language.set(f.read())
    LanguageO = Language.get()
    Messages = UserMessages.Mess[Language.get()]

    Vid_Lecteur.unbindings()

    def main_top(*args):
        Vid_Lecteur.bindings()

    newWindow = Toplevel(master)
    newWindow.bind("<Destroy>", main_top)
    newWindow.grab_set()

    def stay_on_top():
        # Maintain this window on top
        if stay_top_heat:
            newWindow.lift()
            newWindow.after(50, stay_on_top)

    stay_top_heat = True

    Grid.columnconfigure(newWindow, 0, weight=1)
    Grid.rowconfigure(newWindow, 0, weight=1)

    main_Frame = Frame(newWindow, **Color_settings.My_colors.Frame_Base)
    main_Frame.grid(sticky="nsew")

    Grid.rowconfigure(main_Frame, 0, weight=1)
    Grid.columnconfigure(main_Frame, 0, weight=100)
    Grid.columnconfigure(main_Frame, 1, weight=1)

    Canvas_img = Canvas(main_Frame, background=Color_settings.My_colors.Frame_Base["background"], width=500, height=500)
    Canvas_img.grid(row=0, column=0, sticky="nsew")

    Canvas_optn = Canvas(main_Frame, **Color_settings.My_colors.Frame_Base)
    Canvas_optn.grid(row=0, column=1, sticky="nsew")

    Canvas_img.update()
    last_w = 0

    list_possibilities = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens',
                          'Oranges', 'spring', 'summer', 'autumn', 'winter', 'cool']
    palette = IntVar()
    palette.set(0)

    reversed = BooleanVar()
    reversed.set(0)

    repre = IntVar()
    repre.set(0)
    last_repre = 0

    title = StringVar()
    title.set("")

    colorbar_xlab = StringVar()
    colorbar_xlab.set("X")

    colorbar_ylab = StringVar()
    colorbar_ylab.set("Y")

    colorbar_label = StringVar()
    colorbar_label.set("")

    colorbar_ymin = DoubleVar()
    colorbar_ymin.set(0)

    colorbar_ymax = DoubleVar()
    colorbar_ymax.set(heatmap_raw.shape[0] / Vid.Scale[0])

    colorbar_yrange = DoubleVar()
    val = (colorbar_ymax.get()) / 5
    rounder = math.pow(10, len(str(int(val))) - 1)
    colorbar_yrange.set(round(val / rounder) * rounder)

    colorbar_xmin = DoubleVar()
    colorbar_xmin.set(0)

    colorbar_xmax = DoubleVar()
    colorbar_xmax.set(heatmap_raw.shape[1] / Vid.Scale[0])

    colorbar_xrange = DoubleVar()
    val = (colorbar_xmax.get()) / 5
    rounder = math.pow(10, len(str(int(val))) - 1)
    colorbar_xrange.set(round(val / rounder) * rounder)

    cex = DoubleVar()
    cex.set(1)

    colorbar_colmin = DoubleVar()
    colorbar_colmin.set(0)

    colorbar_colmax = DoubleVar()
    colorbar_colrange = DoubleVar()

    max_val_col=0
    min_val_col=0

    last_repre=None
    heatmap_col=None

    def redo_vmax():
        nonlocal max_val_col, min_val_col
        if repre.get() == 0:
            max_val_col = round(np.max(heatmap_raw) / Vid.Frame_rate[1], 3)
            colorbar_colmax.set(max_val_col)
            min_val_col = round(np.min(heatmap_raw) / Vid.Frame_rate[1], 3)

        else:
            max_val_col = round(np.max(heatmap_raw) / Duration, 3)
            colorbar_colmax.set(max_val_col)
            min_val_col = round(np.min(heatmap_raw) / Duration, 3)

        val = (colorbar_colmax.get()) / 5
        rounder = math.pow(10, len(str(int(val))) - 1)
        if rounder == 0:
            colorbar_colrange.set(val)
        else:
            colorbar_colrange.set(round(val / rounder) * rounder)

    redo_vmax()

    def check_val(var, max):
        val = float(var.get())
        if int(val * Vid.Scale[0]) > max:
            var.set(max)
            raise Exception("Can't be bigger than the actual frame")
        if int(val * Vid.Scale[0]) < 0:
            var.set(0)
            raise Exception("Can't be lower than zero")

        return (val)


    def change_style(heatmap_raw):
        nonlocal last_repre
        nonlocal heatmap_col

        heatmap_raw = heatmap_raw.copy().astype(float)
        if repre.get() == 0:
            heatmap_raw = np.divide(heatmap_raw, Vid.Frame_rate[1])
        else:
            heatmap_raw = np.divide(heatmap_raw, Duration)

        if last_repre != repre.get():
            redo_vmax()

        try:
            ymin = check_val(colorbar_ymin, max=heatmap_raw.shape[0])
            xmin = check_val(colorbar_xmin, max=heatmap_raw.shape[1])
            ymax = check_val(colorbar_ymax, max=heatmap_raw.shape[0])
            xmax = check_val(colorbar_xmax, max=heatmap_raw.shape[1])

            heatmap_raw = heatmap_raw[
                          (heatmap_raw.shape[0] - int(ymax * Vid.Scale[0])):(
                                  heatmap_raw.shape[0] - int(ymin * Vid.Scale[0])),
                          int(xmin * Vid.Scale[0]):int(xmax * Vid.Scale[0])
                          ]
        except:
            pass

        ratio = heatmap_raw.shape[1] / heatmap_raw.shape[0]
        fig, ax = plt.subplots(figsize=(10 + 5, 10 / ratio))  # Increase figure size to provide more space
        if reversed.get():
            color = list_possibilities[palette.get()] + "_r"
        else:
            color = list_possibilities[palette.get()]

        # Remove the black lines
        ax.spines[['right', 'top', "left", "bottom"]].set_visible(False)

        try:
            vmin = float(colorbar_colmin.get())
            vmax = float(colorbar_colmax.get())
        except:
            vmin = 0
            if repre.get() == 0:
                vmax = round(np.max(heatmap_raw) / Vid.Frame_rate[1], 3)
            else:
                vmax = round(np.max(heatmap_raw) / Duration, 3)

        im = ax.imshow(
            heatmap_raw, cmap=color, interpolation='nearest',
            extent=[0, heatmap_raw.shape[1] / Vid.Scale[0], 0,
                    heatmap_raw.shape[0] / Vid.Scale[0]],
            vmin=vmin, vmax=vmax
        )

        # X-axis
        try:
            xrange = float(colorbar_xrange.get())
            if xrange == 0:
                ax.set_xticks([])
            else:
                xticks = list(np.arange(0, heatmap_raw.shape[1] / Vid.Scale[0], xrange))
                if len(xticks) < 50:
                    ax.set_xticks(xticks)
        except:
            pass

        # Y-axis
        try:
            yrange = float(colorbar_yrange.get())
            if yrange == 0:
                ax.set_yticks([])
            else:
                yticks = list(np.arange(0, heatmap_raw.shape[0] / Vid.Scale[0], yrange))
                if len(yticks) < 50:
                    ax.set_yticks(yticks)
        except:
            pass

        plt.title(title.get(), pad=30 * cex.get(), fontsize=35.0 * cex.get())
        ax.set_xlabel(colorbar_xlab.get(), fontsize=30.0 * cex.get(), labelpad=15)  # Increase labelpad
        ax.set_ylabel(colorbar_ylab.get(), fontsize=30.0 * cex.get(), labelpad=15)  # Increase labelpad
        ax.tick_params(labelsize=25.0 * cex.get())

        plt.tight_layout()  # Adjust layout to ensure labels and colorbar are visible
        pos = ax.get_position()  # (left, bottom, width, height)

        # Define padding and size for the colorbar
        colorbar_width = 0.03 * cex.get()
        colorbar_pad = 0.02 * cex.get()
        colorbar_ypad = 0.04 * cex.get()

        # Create colorbar axis
        cax = fig.add_axes([pos.x1 + colorbar_pad, pos.y0 + colorbar_ypad, colorbar_width,
                            pos.height - (2 * colorbar_ypad)])  # Adjust position and size

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(colorbar_label.get(), fontsize=30.0 * cex.get(), labelpad=15)
        cbar.ax.tick_params(labelsize=25.0 * cex.get())

        # Customize colorbar ticks
        try:
            vmin = float(colorbar_colmin.get())
            vmax = float(colorbar_colmax.get())
            rangecol = float(colorbar_colrange.get())
            cbar_ticks = np.arange(vmin, vmax + (rangecol / 100000), rangecol)
            cbar_tick_labels = [str(tick) for tick in cbar_ticks]  # Convert all to strings for consistency

            if vmax < max_val_col:
                cbar_tick_labels[-1] = f">{cbar_tick_labels[-1]}"

            if vmin > min_val_col:
                cbar_tick_labels[0] = f"<{cbar_tick_labels[0]}"

            if len(cbar_ticks) <= 25:
                cbar.set_ticks(cbar_ticks)
            else:
                cbar.set_ticks(25)


            cbar.set_ticklabels(cbar_tick_labels)

        except:
            pass

        plt.tight_layout()  # Adjust layout to ensure labels and colorbar are visible

        heatmap_col = canvas_to_array(fig)

        plt.close("all")

        last_repre = repre.get()

    change_style(heatmap_raw)

    def save(selected):
        nonlocal heatmap_col

        heatmap_RGB = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)
        stay_top_heat = False
        file = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=Vid.User_Name + "_Heatmap.png",
                                            filetypes=(("Image", "*.png"),))

        if file != "":
            cv2.imwrite(file, heatmap_RGB)
        stay_top_heat = True

    heatmap=np.array([])

    def update_img(*args):
        nonlocal heatmap_col
        nonlocal heatmap

        ratioW = heatmap_col.shape[1] / (Canvas_img.winfo_width() - 25)
        ratioH = heatmap_col.shape[0] / (Canvas_img.winfo_height() - 25)
        ratio = max(ratioW, ratioH)

        heatmap_col = cv2.resize(heatmap_col,
                                 (int(heatmap_col.shape[1] / ratio), int(heatmap_col.shape[0] / ratio)))

        cv2.imwrite("C:/Users/Usuario/Downloads/TH.jpeg", heatmap_col)

        heatmap = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(heatmap_col))
        Canvas_img.create_image(10, 10, anchor=NW, image=heatmap)
        Canvas_img.update()

    update_img(heatmap_col)

    def change_new_style(heatmap_raw, *args):
        change_style(heatmap_raw)
        update_img()

    Canvas_img.bind("<Configure>", partial(change_new_style, heatmap_raw))

    cur_row = 0

    # Put a title:
    Label(Canvas_optn, text=Messages["Heatmap1"], font='Helvetica 12 bold',
          **Color_settings.My_colors.Label_Base).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1
    Entry(Canvas_optn, textvariable=title, **Color_settings.My_colors.Entry_Base).grid(row=cur_row, column=0,
                                                                                       columnspan=2, sticky="ew")
    title.trace("w", partial(change_new_style, heatmap_raw))
    cur_row += 1

    ttk.Separator(Canvas_optn, orient=HORIZONTAL).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    # Choose which units we want
    Label(Canvas_optn, text=Messages["Heatmap2"], font='Helvetica 12 bold',
          **Color_settings.My_colors.Label_Base).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    Units_frame = Frame(Canvas_optn, **Color_settings.My_colors.Frame_Base)
    Units_frame.grid(row=cur_row, column=0, sticky="nsew")
    cur_row_in = 0

    Radiobutton(Units_frame, text=Messages["Heatmap3"], variable=repre, value=0,
                command=partial(change_new_style, heatmap_raw), **Color_settings.My_colors.Radiobutton_Base).grid(
        row=cur_row_in, column=0, sticky="w")
    Radiobutton(Units_frame, text=Messages["Heatmap4"], variable=repre, value=1,
                command=partial(change_new_style, heatmap_raw), **Color_settings.My_colors.Radiobutton_Base).grid(
        row=cur_row_in, column=1, sticky="w")
    cur_row_in += 1
    Label(Units_frame, text=Messages["Heatmap5"] + ":", **Color_settings.My_colors.Label_Base).grid(row=cur_row_in,
                                                                                                         column=0,
                                                                                                         sticky="ew")
    Entry(Units_frame, textvariable=colorbar_label, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                                column=1, sticky="ew")
    colorbar_label.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Units_frame, text=Messages["Heatmap10"] + ":", **Color_settings.My_colors.Label_Base).grid(
        row=cur_row_in, column=0, sticky="ew")
    Entry(Units_frame, textvariable=colorbar_xlab, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in, column=1,
                                                                                               sticky="ew")
    colorbar_xlab.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Units_frame, text=Messages["Heatmap11"] + ":", **Color_settings.My_colors.Label_Base).grid(
        row=cur_row_in, column=0, sticky="ew")
    Entry(Units_frame, textvariable=colorbar_ylab, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in, column=1,
                                                                                               sticky="ew")
    colorbar_ylab.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Range_frame = Frame(Canvas_optn, **Color_settings.My_colors.Frame_Base)
    Range_frame.grid(row=cur_row, column=1, sticky="nsew")
    cur_row_in = 0

    Label(Range_frame, text=Messages["Heatmap12"], font='Helvetica 9 bold',
          **Color_settings.My_colors.Label_Base).grid(row=cur_row_in, column=0, columnspan=2, sticky="ew")
    cur_row_in += 1

    Label(Range_frame, text=Messages["Param15"] + ":", **Color_settings.My_colors.Label_Base).grid(row=cur_row_in,
                                                                                                        column=0,
                                                                                                        sticky="ew")
    Entry(Range_frame, textvariable=colorbar_colmin, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                                 column=1, sticky="ew")
    colorbar_colmin.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Range_frame, text=Messages["Param16"] + ":", **Color_settings.My_colors.Label_Base).grid(row=cur_row_in,
                                                                                                        column=0,
                                                                                                        sticky="ew")
    Entry(Range_frame, textvariable=colorbar_colmax, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                                 column=1, sticky="ew")
    colorbar_colmax.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Range_frame, text=Messages["Heatmap13"] + ":", **Color_settings.My_colors.Label_Base).grid(
        row=cur_row_in, column=0, sticky="ew")
    Entry(Range_frame, textvariable=colorbar_colrange, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                                   column=1,
                                                                                                   sticky="ew")
    colorbar_colrange.trace("w", partial(change_new_style, heatmap_raw))
    cur_row += 1

    ttk.Separator(Canvas_optn, orient=HORIZONTAL).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    # The axis ranges
    Label(Canvas_optn, text=Messages["Heatmap9"], font='Helvetica 12 bold',
          **Color_settings.My_colors.Label_Base).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    # Xaxis
    Xaxis_frame = Frame(Canvas_optn, **Color_settings.My_colors.Frame_Base)
    Xaxis_frame.grid(row=cur_row, column=0, sticky="nsew")
    cur_row_in = 0

    Label(Xaxis_frame, text="X", font='Helvetica 10 bold', **Color_settings.My_colors.Label_Base).grid(row=cur_row_in,
                                                                                                       column=0,
                                                                                                       columnspan=2,
                                                                                                       sticky="ew")
    cur_row_in += 1

    Label(Xaxis_frame, text=Messages["Param15"] + ":", **Color_settings.My_colors.Label_Base).grid(row=cur_row_in,
                                                                                                        column=0,
                                                                                                        sticky="ew")
    Entry(Xaxis_frame, textvariable=colorbar_xmin, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in, column=1,
                                                                                               sticky="ew")
    colorbar_xmin.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Xaxis_frame, text=Messages["Param16"] + ":", **Color_settings.My_colors.Label_Base).grid(row=cur_row_in,
                                                                                                        column=0,
                                                                                                        sticky="ew")
    Entry(Xaxis_frame, textvariable=colorbar_xmax, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in, column=1,
                                                                                               sticky="ew")
    colorbar_xmax.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Xaxis_frame, text=Messages["Heatmap13"] + ":", **Color_settings.My_colors.Label_Base).grid(
        row=cur_row_in, column=0, sticky="ew")
    Entry(Xaxis_frame, textvariable=colorbar_xrange, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                                 column=1, sticky="ew")
    colorbar_xrange.trace("w", partial(change_new_style, heatmap_raw))

    # Yaxis
    Yaxis_frame = Frame(Canvas_optn, **Color_settings.My_colors.Frame_Base)
    Yaxis_frame.grid(sticky="nsew")
    Yaxis_frame.grid(row=cur_row, column=1, sticky="nsew")
    cur_row_in = 0

    Label(Yaxis_frame, text="Y", font='Helvetica 10 bold',
          **Color_settings.My_colors.Label_Base).grid(row=cur_row_in, column=0, columnspan=2, sticky="ew")
    cur_row_in += 1

    Label(Yaxis_frame, text=Messages["Param15"] + ":", **Color_settings.My_colors.Label_Base).grid(
        row=cur_row_in, column=0, sticky="ew")
    Entry(Yaxis_frame, textvariable=colorbar_ymin, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                               column=1,
                                                                                               sticky="ew")
    colorbar_ymin.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Yaxis_frame, text=Messages["Param16"] + ":", **Color_settings.My_colors.Label_Base).grid(
        row=cur_row_in, column=0, sticky="ew")
    Entry(Yaxis_frame, textvariable=colorbar_ymax, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                               column=1,
                                                                                               sticky="ew")
    colorbar_ymax.trace("w", partial(change_new_style, heatmap_raw))
    cur_row_in += 1

    Label(Yaxis_frame, text=Messages["Heatmap13"] + ":", **Color_settings.My_colors.Label_Base).grid(
        row=cur_row_in, column=0, sticky="ew")
    Entry(Yaxis_frame, textvariable=colorbar_yrange, **Color_settings.My_colors.Entry_Base).grid(row=cur_row_in,
                                                                                                 column=1, sticky="ew")
    colorbar_yrange.trace("w", partial(change_new_style, heatmap_raw))
    cur_row += 1

    ttk.Separator(Canvas_optn, orient=HORIZONTAL).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    # The heatmap color
    Label(Canvas_optn, text=Messages["Heatmap6"], font='Helvetica 12 bold',
          **Color_settings.My_colors.Label_Base).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    colors_frame = Frame(Canvas_optn, **Color_settings.My_colors.Frame_Base)
    colors_frame.grid(row=cur_row, column=0, columnspan=2, sticky="nsew")

    Grid.columnconfigure(colors_frame, 0, weight=1)
    Grid.columnconfigure(colors_frame, 1, weight=1)
    Label(colors_frame, text=Messages["Heatmap14"], font='Helvetica 9 bold',
          **Color_settings.My_colors.Label_Base).grid(row=0, column=0, sticky="nsw")
    Label(colors_frame, text=Messages["Heatmap15"], font='Helvetica 9 bold',
          **Color_settings.My_colors.Label_Base).grid(row=0, column=1, sticky="nsw")
    Label(colors_frame, text="Sequential2", font='Helvetica 9 bold', **Color_settings.My_colors.Label_Base).grid(row=0,
                                                                                                                 column=2,
                                                                                                                 sticky="nsw")  # CTXT
    cur_row += 1

    row_in = 1
    col_in = 0
    nb_row = 5
    v = 0
    for color in list_possibilities:
        Radiobutton(colors_frame, text=list_possibilities[v], variable=palette, value=v,
                    command=partial(change_new_style, heatmap_raw), **Color_settings.My_colors.Radiobutton_Base).grid(
            row=row_in, column=col_in, sticky="w")

        row_in += 1
        if row_in > nb_row:
            col_in += 1
            row_in = 1
        v += 1

    Check_rev = Checkbutton(Canvas_optn, text=Messages["Heatmap7"], variable=reversed,
                            command=partial(change_new_style, heatmap_raw), **Color_settings.My_colors.Checkbutton_Base)
    Check_rev.grid(row=cur_row, columnspan=2, column=0, sticky="ew")
    cur_row += 1

    ttk.Separator(Canvas_optn, orient=HORIZONTAL).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    Label(Canvas_optn, text=Messages["Heatmap16"], font='Helvetica 12 bold',
          **Color_settings.My_colors.Label_Base).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    Scale(Canvas_optn, variable=cex, from_=0.05, to=2, orient=HORIZONTAL, resolution=0.05, show=True,
          **Color_settings.My_colors.Scale_Base).grid(row=cur_row, columnspan=2, column=0, sticky="ew")
    cex.trace("w", partial(change_new_style, heatmap_raw))
    cur_row += 1

    ttk.Separator(Canvas_optn, orient=HORIZONTAL).grid(row=cur_row, column=0, columnspan=2, sticky="ew")
    cur_row += 1

    Butt_save = Button(Canvas_optn, text=Messages["Heatmap8"], command=partial(save, selected),
                       **Color_settings.My_colors.Button_Base)
    Butt_save.grid(row=cur_row, columnspan=2, column=0, sticky="ew")
    cur_row += 1