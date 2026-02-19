import os
import cv2
from AnimalTA.A_General_tools import UserMessages
from pymediainfo import MediaInfo
import subprocess
import re


def convert_to_avi(file, new_file, frame_rate=None, quality_vid=10, progress=None):
    """Function to convert videos to .avi format. The new .avi files will be stored in the project folder with the same name as the previous one."""
    try:
        File_folder = UserMessages.resource_path(os.path.join("AnimalTA", "Files"))
        ffmpeg_path = os.path.join(File_folder, "ffmpeg", "ffmpeg.exe")

        cap = cv2.VideoCapture(file)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        if frame_rate is None:
            frame_rate = cap.get(cv2.CAP_PROP_FPS)

        nb_frame_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        max_frames = 60000
        parts = (nb_frame_tot // max_frames) + (1 if nb_frame_tot % max_frames > 0 else 0)

        Fusions=[]


        for i in range(parts):
            start_frame = i * max_frames
            if parts>1:
                output_name = new_file.replace(".avi", f"_part_{i + 1}.avi")
            else:
                output_name = new_file

            start_time = start_frame / frame_rate
            duration_sec = (max_frames / frame_rate)

            media_info = MediaInfo.parse(file)
            rotation = 0  # Default: no rotation
            dar = 1

            for track in media_info.tracks:
                if track.track_type == "Video" and not track.display_aspect_ratio is None:
                    dar = float(track.display_aspect_ratio)  # Get DAR
                    if hasattr(track, "rotation") and not track.rotation is None:
                        rotation = int(float(track.rotation))  # Convert safely

            # Determine correct width
            if rotation in [90, 270]:  # Video is in portrait mode
                frame_width, frame_height = frame_height, frame_width  # Swap width & height
                frame_width_show = int(round(frame_height / dar))  # Use division
            else:
                frame_width_show = int(round(frame_height * dar))  # Use multiplication

            # Ensure width and height are even
            even_width = frame_width_show - (frame_width_show % 2)
            even_height = frame_height - (frame_height % 2)

            scale_filter = f"scale={even_width}:{even_height}"

            current_dar = frame_width / frame_height

            base_cmd = [
                ffmpeg_path, "-stats_period", "0.25",
                "-ss", str(start_time),
                "-t", str(duration_sec),
                "-i", file
            ]

            if round(current_dar, 2) == 1.0:
                cmd = base_cmd + [
                    "-c:v", "libxvid",
                    "-q:v", str(quality_vid),
                    "-r", str(frame_rate),
                    "-vsync", "cfr",
                    "-pix_fmt", "yuv420p",
                    "-progress", "pipe:1", "-y", output_name
                ]
            else:
                cmd = base_cmd + [
                    "-vf", f"setpts=PTS-STARTPTS,{scale_filter}",
                    "-c:v", "libxvid",
                    "-q:v", str(quality_vid),
                    "-r", str(frame_rate),
                    "-vsync", "cfr",
                    "-pix_fmt", "yuv420p",
                    "-progress", "pipe:1", "-y", output_name
                ]

            Fusions.append([start_frame, output_name])


            startupinfo = None
            if hasattr(subprocess, "STARTUPINFO"):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW, text=True)
            # Read progress in real time

            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break  # Process finished

                match = re.search(r"frame=(\d+)", output)  # Extract only frame count
                if match:
                    cur_fr=match.group(1)  # Print only the frame number
                    progress_val = (int(cur_fr)+i*max_frames) / nb_frame_tot
                    progress.value=progress_val

        real_nb_fr=(int(cur_fr)+i*max_frames)


    except Exception as e:
        print(e)
        return(["Error"])


    return([Fusions, real_nb_fr, new_file])


    #
    # except Exception as e:
    #     print(e)
    #     if os.path.isfile(new_file):
    #         os.remove(new_file)
    #     return ["Error"]

