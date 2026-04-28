import cv2
import numpy as np

from AnimalTA.A_General_tools import gpu_utils


def apply_relative_background(image, background):
    """Scale a background-difference image while avoiding divide-by-zero artifacts."""
    if gpu_utils.CUPY_AVAILABLE:
        import cupy as cp
        img_gpu = cp.asarray(image)
        bg_gpu = cp.asarray(background)
        safe = cp.maximum(bg_gpu.astype(cp.uint32), 1)
        scaled = (img_gpu.astype(cp.uint32) * 255) // safe
        return cp.asnumpy(cp.clip(scaled, 0, 255).astype(cp.uint8))
    safe_background = np.maximum(background.astype(np.uint32), 1)
    scaled = (image.astype(np.uint32) * 255) // safe_background
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_brightness_correction(Timg, mask, or_bright, mask_enabled):
    """Normalise per-frame brightness to the reference level of the first frame."""
    if gpu_utils.CUPY_AVAILABLE:
        import cupy as cp
        grey_gpu = cp.asarray(Timg)
        if mask_enabled:
            bool_mask = cp.asarray(mask[:, :].astype(bool))
        else:
            bool_mask = cp.ones(grey_gpu.shape, dtype=bool)
        grey2 = grey_gpu[bool_mask]
        brightness = float(cp.sum(grey2)) / (255 * grey2.size)
        ratio = brightness / or_bright
        result_gpu = cp.clip(grey_gpu.astype(cp.float32) * (1.0 / ratio), 0, 255).astype(cp.uint8)
        return cp.asnumpy(result_gpu)
    grey = np.copy(Timg)
    bool_mask = mask[:, :].astype(bool) if mask_enabled else np.full(grey.shape, True)
    grey2 = grey[bool_mask]
    brightness = np.sum(grey2) / (255 * grey2.size)
    ratio = brightness / or_bright
    return cv2.convertScaleAbs(grey, alpha=1.0 / ratio, beta=0)
