"""Cross-platform compatibility layer for AnimalTA.

Replaces Windows-only calls with portable equivalents.
All functions are best-effort and never raise.
"""
import sys
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def beep(frequency: int = 440, duration_ms: int = 200) -> None:
    """Cross-platform beep. Best-effort, never raises."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(frequency, duration_ms)
        # Audio alerts are non-critical for the tracking workflow.
        # In a VNC/OOD session audio is typically unavailable anyway.
    except Exception:
        logger.debug("Audio beep unavailable on this platform")


def play_sound(filepath: str) -> None:
    """Cross-platform WAV playback. Best-effort, never raises."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_ASYNC)
        # Non-critical in OOD/VNC context â€” fail silently on other platforms.
    except Exception:
        logger.debug("Sound playback failed", exc_info=True)


def open_file_external(filepath: str) -> None:
    """Cross-platform 'open file with default app'. Best-effort, never raises."""
    try:
        import shutil
        import subprocess
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        else:
            xdg = shutil.which("xdg-open")
            if xdg:
                subprocess.Popen([xdg, filepath])
    except Exception:
        logger.debug("open_file_external failed", exc_info=True)


def get_pointer_position(widget=None):
    """Return the current pointer position using Tk rather than pyautogui."""
    try:
        if widget is not None:
            return widget.winfo_pointerxy()
    except Exception:
        logger.debug("Pointer lookup via widget failed", exc_info=True)

    try:
        import tkinter
        if tkinter._default_root is not None:
            return tkinter._default_root.winfo_pointerxy()
    except Exception:
        logger.debug("Pointer lookup via default root failed", exc_info=True)

    return 0, 0


def set_window_icon(window) -> None:
    """Best-effort cross-platform window icon setup."""
    try:
        from tkinter import PhotoImage
        from AnimalTA.A_General_tools import UserMessages

        if sys.platform == "win32":
            icon_path = UserMessages.resource_path(os.path.join("AnimalTA", "Files", "Logo.ico"))
            if os.path.exists(icon_path):
                window.iconbitmap(icon_path)
            return

        icon_path = UserMessages.resource_path(os.path.join("AnimalTA", "Files", "Logo.png"))
        if os.path.exists(icon_path):
            icon = PhotoImage(file=icon_path)
            window._animalta_icon = icon
            window.iconphoto(True, icon)
    except Exception:
        logger.debug("set_window_icon failed", exc_info=True)



def set_toolwindow(window, enabled: bool = True) -> None:
    """Best-effort Windows toolwindow flag. Ignored on non-Windows platforms."""
    try:
        if sys.platform == "win32":
            window.attributes("-toolwindow", bool(enabled))
    except Exception:
        logger.debug("set_toolwindow failed", exc_info=True)
def find_resource_path(*relative_paths):
    """Return the first existing bundled resource path from the candidates."""
    try:
        from AnimalTA.A_General_tools import UserMessages

        for relative_path in relative_paths:
            path = UserMessages.resource_path(relative_path)
            if os.path.exists(path):
                return path
    except Exception:
        logger.debug("find_resource_path failed", exc_info=True)

    return None


def load_cv_rgb_resource(*relative_paths, fallback_shape=(1, 1, 3)):
    """Load a bundled image via OpenCV and return RGB data, or a placeholder."""
    import cv2
    import numpy as np

    path = find_resource_path(*relative_paths)
    if path is not None:
        image = cv2.imread(path)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.zeros(fallback_shape, dtype=np.uint8)


def load_tk_image_resource(*relative_paths, size=None, fallback_size=(22, 22)):
    """Load a bundled image as a Tk PhotoImage, or return a transparent placeholder."""
    from PIL import Image, ImageTk

    path = find_resource_path(*relative_paths)
    if path is not None:
        try:
            image = Image.open(path)
            if size is not None:
                image = image.resize(size)
            return ImageTk.PhotoImage(image)
        except Exception:
            logger.debug("load_tk_image_resource failed for %s", path, exc_info=True)

    image = Image.new("RGBA", size or fallback_size, (0, 0, 0, 0))
    return ImageTk.PhotoImage(image)


def startup_debug_enabled() -> bool:
    """Return True when startup tracing is enabled."""
    return os.environ.get("ANIMALTA_STARTUP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def startup_debug(message: str) -> None:
    """Emit a flushed startup trace line to stderr when enabled."""
    if not startup_debug_enabled():
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[AnimalTA startup {timestamp}] {message}", file=sys.stderr, flush=True)


def get_font_path(size: int = 20):
    """Return a (fontpath, size) tuple for a font available on this platform.

    Falls back to PIL's built-in default if no TrueType font is found.
    Returns (path_or_None, size) â€” callers should use ImageFont.load_default()
    when path is None.
    """
    import shutil

    candidates = [
        # Linux (DejaVu ships with most distros and matplotlib)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        # macOS system font
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        # Windows fallback (original was simsun.ttc)
        os.path.join(".", "simsun.ttc"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path, size

    return None, size

