import os
import csv
import numpy as np
from tkinter import *
from AnimalTA.A_General_tools import Class_loading_Frame, UserMessages
import time

def load_coos(Vid, TMP=False, location=None):
    # Importation of the coordinates associated with the current video
    if Vid.User_Name == Vid.Name:
        file_name = Vid.Name
        point_pos = file_name.rfind(".")
        if file_name[point_pos:].lower()!=".avi":#Old versions of AnimalTA did not allow to rename or duplicate the videos, the name of the video was the file name without the ".avi" extension
            file_name = Vid.User_Name
        else:
            file_name = file_name[:point_pos]
    else:
        file_name = Vid.User_Name

    if not TMP:
        file_tracked_not_corr = os.path.join(
            UserMessages.coordinates_dir_path(Vid.Folder),
            file_name + "_Coordinates.csv",
        )
        file_tracked_corr = os.path.join(
            UserMessages.corrected_coordinates_dir_path(Vid.Folder),
            file_name + "_Corrected.csv",
        )
    else:
        file_tracked_corr = os.path.join(
            UserMessages.tmp_portion_dir_path(Vid.Folder),
            file_name + "_TMP_portion_Coordinates.csv",
        )

    if os.path.isfile(file_tracked_corr):
        path = file_tracked_corr
    else:
        path = file_tracked_not_corr



    if Vid.Track[1][8]:
        return load_fixed(Vid, path, location)
    else:
        return load_variable(Vid, path)





def save(Vid, Coos, TMP=False, location=None):
    # Save the coordinates associated with the current video
    if Vid.User_Name == Vid.Name:
        file_name = Vid.Name
        point_pos = file_name.rfind(".")
        if file_name[point_pos:].lower()!=".avi":#Old versions of AnimalTA did not allow to rename or duplicate the videos, the name of the video was the file name without the ".avi" extension
            file_name = Vid.User_Name
        else:
            file_name = file_name[:point_pos]
    else:
        file_name = Vid.User_Name

    if not TMP:
        path = os.path.join(
            UserMessages.corrected_coordinates_dir_path(Vid.Folder, create=True),
            file_name + "_Corrected.csv",
        )
    else:
        path = os.path.join(
            UserMessages.tmp_portion_dir_path(Vid.Folder, create=True),
            file_name + "_TMP_portion_Coordinates.csv",
        )

    if os.path.isfile(path):
        path = path

    if Vid.Track[1][8]:
        save_fixed(Vid, Coos, path, location)
    else:
        save_variable(Vid, Coos, path)


def load_variable(Vid, path):
    one_every=Vid.Frame_rate[0] / Vid.Frame_rate[1]
    newWindow = Toplevel()
    load_frame = Class_loading_Frame.Loading(newWindow)  # Progression bar
    load_frame.grid()

    frame_count = int((Vid.Cropped[1][1] - Vid.Cropped[1][0]) / one_every) + 1
    who_is_here = [[] for _ in range(frame_count)]
    has_data = False
    with open(path, encoding="utf-8", newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        next(csv_reader, None)
        has_data = next(csv_reader, None) is not None

    identity_lookup = {}
    for index, identity in enumerate(Vid.Identities):
        arena, name = str(identity[0]), str(identity[1])
        identity_lookup[(arena, name)] = index
        identity_lookup[(arena, name[3:])] = index
    identity_count = len(Vid.Identities) if has_data else 1
    Coos = np.full((identity_count, frame_count, 2), -1000, dtype=float)

    with open(path, encoding="utf-8", newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        next(csv_reader, None)
        for row_index, row in enumerate(csv_reader):
            if len(row) < 6:
                continue
            identity_index = identity_lookup.get((str(row[2]), str(row[3])))
            if identity_index is None:
                continue
            try:
                frame = int(float(row[0])) - round(Vid.Cropped[1][0] / one_every)
                x, y = float(row[4]), float(row[5])
            except (ValueError, TypeError):
                continue
            if 0 <= frame < frame_count:
                Coos[identity_index, frame, :] = (x, y)
                who_is_here[frame].append(identity_index)
            if row_index % 1000 == 0:
                load_frame.show_load(row_index / max(frame_count, 1))

    load_frame.destroy()
    newWindow.destroy()
    return(Coos, who_is_here)


def _fixed_header(Vid, coordinate_width):
    columns = []
    for ind in Vid.Identities:
        if coordinate_width == 2:
            columns.extend(["X_Arena{}_Ind{}".format(ind[0], ind[1]),
                            "Y_Arena{}_Ind{}".format(ind[0], ind[1])])
        else:
            columns.extend(
                ["Coordinate{}_Arena{}_Ind{}".format(axis, ind[0], ind[1])
                 for axis in range(coordinate_width)]
            )
    return ["Frame", "Time"] + columns


def _fixed_csv_value(value):
    return "NA" if value == -1000 else value.item() if hasattr(value, "item") else value


def iter_fixed_rows(Vid, Coos):
    """Yield fixed-tracking CSV rows without building an object-array copy."""
    coordinate_width = Coos.shape[2]
    yield _fixed_header(Vid, coordinate_width)
    for frame in range(Coos.shape[1]):
        row = [frame, round(frame / Vid.Frame_rate[1], 2)]
        for ind in range(Coos.shape[0]):
            row.extend(_fixed_csv_value(value) for value in Coos[ind, frame])
        yield row


def load_fixed(Vid, path, location=None):
    if location==None:
        frame = Toplevel()
    else:
        frame=location
    load_frame = Class_loading_Frame.Loading(frame)  # Progression bar
    load_frame.grid()
    load_frame.show_load(0)
    load_frame.grab_set()

    with open(path, encoding="utf-8", newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        header = next(csv_reader, [])
        row_count = sum(1 for _row in csv_reader)

    identity_count = len(Vid.Identities)
    if identity_count == 0:
        coordinate_width = 2
    else:
        coordinate_columns = len(header) - 2
        if coordinate_columns == 4 * identity_count:
            coordinate_width = 4
        elif coordinate_columns == 3 * identity_count:
            coordinate_width = 3
        else:
            coordinate_width = 2
    Coos = np.full(
        (identity_count, row_count, coordinate_width), -1000, dtype=float
    )

    with open(path, encoding="utf-8", newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        next(csv_reader, None)
        for row_index, row in enumerate(csv_reader):
            for ind in range(identity_count):
                start = 2 + ind * coordinate_width
                values = row[start:start + coordinate_width]
                if len(values) != coordinate_width:
                    continue
                for axis, value in enumerate(values):
                    if value != "NA":
                        try:
                            Coos[ind, row_index, axis] = float(value)
                        except ValueError:
                            pass
            if row_index % 1000 == 0:
                load_frame.show_load(1 / 3 + (row_index / max(row_count, 1)) * 2 / 3)

    load_frame.destroy()
    if location==None:
        frame.destroy()

    load_frame.grab_release()
    return (Coos, [list(range(len(Vid.Identities)))]*len(Coos[0,:,0]))

def save_fixed(Vid, Coos, path, location=None):
    if location == None:
        frame = Toplevel()
    else:
        frame = location
    load_frame = Class_loading_Frame.Loading(frame)  # Progression bar
    load_frame.grid()
    load_frame.show_load(0)
    with open(path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        for row_index, row in enumerate(iter_fixed_rows(Vid, Coos)):
            writer.writerow(row)
            if row_index % 300 == 0:
                load_frame.show_load(
                    1 / 3 + (row_index / max(Coos.shape[1], 1)) * 2 / 3
                )

    #np.savetxt(path, General_Coos, delimiter=';', encoding="utf-8", fmt='%s')
    load_frame.destroy()

    if location == None:
        frame.destroy()


def save_variable(Vid, Coos, path):
    one_every = Vid.Frame_rate[0] / Vid.Frame_rate[1]
    frame_offset = round(Vid.Cropped[1][0] / one_every)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["Frame", "Time", "Arena", "Ind", "X", "Y"])
        for frame in range(Coos.shape[1]):
            output_frame = frame + frame_offset
            for ind in range(Coos.shape[0]):
                if Coos[ind, frame, 0] == -1000:
                    continue
                writer.writerow([
                    output_frame,
                    output_frame / Vid.Frame_rate[1],
                    Vid.Identities[ind][0],
                    Vid.Identities[ind][1],
                    Coos[ind, frame, 0],
                    Coos[ind, frame, 1],
                ])
