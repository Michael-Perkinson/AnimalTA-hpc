import os
import sys
import cv2
from AnimalTA.A_General_tools import UserMessages
from pymediainfo import MediaInfo
import subprocess
import re
import tempfile

_nvenc_cache = None


def _probe_nvenc(ffmpeg_path):
    """Return True if this ffmpeg build has h264_nvenc and a CUDA device is accessible."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
            tmp = f.name
        cmd = [
            ffmpeg_path, "-y",
            "-f", "lavfi", "-i", "color=black:size=16x16:rate=1:duration=1",
            "-c:v", "h264_nvenc", "-frames:v", "1", tmp,
        ]
        startupinfo = None
        creationflags = 0
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            startupinfo=startupinfo, creationflags=creationflags, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass


def _nvenc_available(ffmpeg_path):
    global _nvenc_cache
    if _nvenc_cache is None:
        _nvenc_cache = _probe_nvenc(ffmpeg_path)
    return _nvenc_cache


def _codec_args(use_nvenc, quality_vid, scale_filter=None):
    """Return the ffmpeg codec/quality/filter args for the chosen encoder.

    quality_vid is the libxvid q:v scale (1=best, 31=worst).  NVENC uses a
    constrained-quality scale (0=best, 51=worst) so we remap linearly.
    """
    vf_args = ["-vf", f"setpts=PTS-STARTPTS,{scale_filter}"] if scale_filter else []

    if use_nvenc:
        cq = int(round((quality_vid - 1) / 30 * 51))  # remap 1-31 -> 0-51
        return vf_args + [
            "-c:v", "h264_nvenc",
            "-rc", "constqp", "-qp", str(cq),
            "-pix_fmt", "yuv420p",
        ]
    else:
        return vf_args + [
            "-c:v", "libxvid",
            "-q:v", str(quality_vid),
            "-pix_fmt", "yuv420p",
            "-threads", str(max(1, (os.cpu_count() or 2) // 2)),
        ]


def convert_to_avi(file, new_file, frame_rate=None, quality_vid=10, progress=None):
    """Function to convert videos to .avi format. The new .avi files will be stored in the project folder with the same name as the previous one."""
    try:
        File_folder = UserMessages.resource_path(os.path.join("AnimalTA", "Files"))
        if sys.platform == "win32":
            ffmpeg_path = os.path.join(File_folder, "ffmpeg", "ffmpeg.exe")
        else:
            ffmpeg_path = "ffmpeg"  # use system ffmpeg on Linux/macOS

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

            use_nvenc = _nvenc_available(ffmpeg_path)
            needs_scale = round(current_dar, 2) != 1.0

            base_cmd = [
                ffmpeg_path, "-stats_period", "0.25",
                "-ss", str(start_time),
                "-t", str(duration_sec),
                "-i", file
            ]

            codec_args = _codec_args(
                use_nvenc, quality_vid,
                scale_filter=scale_filter if needs_scale else None,
            )
            cmd = base_cmd + codec_args + [
                "-r", str(frame_rate),
                "-vsync", "cfr",
                "-progress", "pipe:1", "-y", output_name
            ]

            Fusions.append([start_frame, output_name])


            startupinfo = None
            creationflags = 0
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, startupinfo=startupinfo, creationflags=creationflags, text=True)
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

