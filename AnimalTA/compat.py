"""Cross-platform compatibility layer for AnimalTA.

Replaces Windows-only calls with portable equivalents.
All functions are best-effort and never raise.
"""
import sys
import os
import logging

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
        # Non-critical in OOD/VNC context — fail silently on other platforms.
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


def get_font_path(size: int = 20):
    """Return a (fontpath, size) tuple for a font available on this platform.

    Falls back to PIL's built-in default if no TrueType font is found.
    Returns (path_or_None, size) — callers should use ImageFont.load_default()
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
