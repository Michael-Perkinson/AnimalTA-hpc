import cv2
import numpy as np


def apply_relative_background(image, background):
    """Scale a background-difference image while avoiding divide-by-zero artifacts."""
    safe_background = np.maximum(background.astype(np.uint32), 1)
    scaled = (image.astype(np.uint32) * 255) // safe_background
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_brightness_correction(Timg, mask, or_bright, mask_enabled):
    """Normalise per-frame brightness to the reference level of the first frame."""
    grey = np.copy(Timg)
    bool_mask = mask[:, :].astype(bool) if mask_enabled else np.full(grey.shape, True)
    grey2 = grey[bool_mask]
    brightness = np.sum(grey2) / (255 * grey2.size)
    ratio = brightness / or_bright
    return cv2.convertScaleAbs(grey, alpha=1.0 / ratio, beta=0)
