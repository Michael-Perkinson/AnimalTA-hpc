import numpy as np


def apply_relative_background(image, background):
    """Scale a background-difference image while avoiding divide-by-zero artifacts."""
    safe_background = np.maximum(background.astype(np.uint32), 1)
    scaled = (image.astype(np.uint32) * 255) // safe_background
    return np.clip(scaled, 0, 255).astype(np.uint8)
