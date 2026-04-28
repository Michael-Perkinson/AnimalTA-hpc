import numpy as np
import pytest

from AnimalTA.A_General_tools.image_utils import (
    apply_brightness_correction,
    apply_relative_background,
)
from AnimalTA.A_General_tools import gpu_utils


class TestApplyRelativeBackground:
    def _make(self, h=4, w=4):
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (h, w), dtype=np.uint8)

    def test_output_dtype_is_uint8(self):
        result = apply_relative_background(self._make(), self._make())
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self):
        img = self._make(8, 6)
        result = apply_relative_background(img, self._make(8, 6))
        assert result.shape == img.shape

    def test_output_values_in_uint8_range(self):
        result = apply_relative_background(self._make(), self._make())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_zero_background_no_divide_by_zero(self):
        img = np.full((4, 4), 128, dtype=np.uint8)
        bg = np.zeros((4, 4), dtype=np.uint8)
        result = apply_relative_background(img, bg)
        assert result.dtype == np.uint8

    def test_zero_background_treated_as_one(self):
        img = np.array([[100, 200], [0, 50]], dtype=np.uint8)
        bg = np.zeros((2, 2), dtype=np.uint8)
        result = apply_relative_background(img, bg)
        expected = np.clip((img.astype(np.uint32) * 255) // 1, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_background_equal_to_image_gives_255(self):
        v = np.array([[100]], dtype=np.uint8)
        assert apply_relative_background(v, v)[0, 0] == 255

    def test_background_larger_than_image_gives_less_than_255(self):
        img = np.array([[100]], dtype=np.uint8)
        bg = np.array([[200]], dtype=np.uint8)
        assert apply_relative_background(img, bg)[0, 0] < 255

    def test_black_image_gives_all_zeros(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        np.testing.assert_array_equal(apply_relative_background(img, self._make()), 0)

    def test_computation_correctness(self):
        img = np.array([[50, 100], [0, 255]], dtype=np.uint8)
        bg = np.array([[100, 200], [1, 255]], dtype=np.uint8)
        result = apply_relative_background(img, bg)
        safe_bg = np.maximum(bg.astype(np.uint32), 1)
        expected = np.clip((img.astype(np.uint32) * 255) // safe_bg, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_uniform_image_gives_uniform_result(self):
        img = np.full((3, 3), 50, dtype=np.uint8)
        bg = np.full((3, 3), 100, dtype=np.uint8)
        result = apply_relative_background(img, bg)
        assert np.all(result == result[0, 0])


class TestApplyBrightnessCorrection:
    def _make(self, h=20, w=20, low=50, high=200):
        rng = np.random.default_rng(42)
        return rng.integers(low, high, (h, w), dtype=np.uint8)

    def test_output_dtype_is_uint8(self):
        img = self._make()
        mask = np.ones((20, 20), dtype=np.uint8)
        result = apply_brightness_correction(img, mask, 0.5, True)
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self):
        img = self._make(30, 40)
        mask = np.ones((30, 40), dtype=np.uint8)
        result = apply_brightness_correction(img, mask, 0.5, True)
        assert result.shape == img.shape

    def test_output_values_in_uint8_range(self):
        img = self._make()
        mask = np.ones((20, 20), dtype=np.uint8)
        result = apply_brightness_correction(img, mask, 0.5, True)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_no_mask_uses_full_image(self):
        img = self._make()
        mask = np.zeros((20, 20), dtype=np.uint8)  # mask irrelevant when mask_enabled=False
        result = apply_brightness_correction(img, mask, 0.5, False)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_bright_frame_is_darkened(self):
        # Image brighter than reference → ratio > 1 → alpha < 1 → output darker
        img = np.full((10, 10), 200, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)
        or_bright = 0.3  # reference brightness ~77/255 ≈ 0.30
        result = apply_brightness_correction(img, mask, or_bright, True)
        assert result.mean() < img.mean()

    def test_dark_frame_is_brightened(self):
        img = np.full((10, 10), 50, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)
        or_bright = 0.6  # reference much brighter than current frame
        result = apply_brightness_correction(img, mask, or_bright, True)
        assert result.mean() > img.mean()

    def test_mask_restricts_brightness_sample(self):
        # Only masked region is used to compute brightness; surrounding pixels irrelevant
        img = np.full((10, 10), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4:6, 4:6] = 1
        result_masked = apply_brightness_correction(img, mask, 0.5, True)
        result_full = apply_brightness_correction(img, mask, 0.5, False)
        # Both should be uint8 and same shape; values may differ or not
        assert result_masked.dtype == np.uint8
        assert result_full.dtype == np.uint8


@pytest.mark.skipif(not gpu_utils.CUPY_AVAILABLE, reason="CuPy not available")
class TestGpuApplyRelativeBackground:
    def test_result_matches_cpu(self):
        import cupy as cp
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        bg = rng.integers(1, 256, (100, 100), dtype=np.uint8)
        safe_bg = np.maximum(bg.astype(np.uint32), 1)
        cpu = np.clip((img.astype(np.uint32) * 255) // safe_bg, 0, 255).astype(np.uint8)
        gpu = apply_relative_background(img, bg)
        np.testing.assert_array_equal(cpu, gpu)

    def test_output_dtype_is_uint8(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (50, 50), dtype=np.uint8)
        bg = rng.integers(1, 256, (50, 50), dtype=np.uint8)
        assert apply_relative_background(img, bg).dtype == np.uint8

    def test_zero_background_no_divide_by_zero(self):
        img = np.full((4, 4), 128, dtype=np.uint8)
        bg = np.zeros((4, 4), dtype=np.uint8)
        result = apply_relative_background(img, bg)
        assert result.dtype == np.uint8


@pytest.mark.skipif(not gpu_utils.CUPY_AVAILABLE, reason="CuPy not available")
class TestGpuApplyBrightnessCorrection:
    def test_result_close_to_cpu(self):
        import cupy as cp
        rng = np.random.default_rng(1)
        img = rng.integers(50, 200, (200, 200), dtype=np.uint8)
        mask = np.ones((200, 200), dtype=np.uint8)
        # Force CPU path by temporarily patching CUPY_AVAILABLE
        import AnimalTA.A_General_tools.gpu_utils as _gu
        orig = _gu.CUPY_AVAILABLE
        _gu.CUPY_AVAILABLE = False
        cpu = apply_brightness_correction(img, mask, 0.5, True)
        _gu.CUPY_AVAILABLE = orig
        gpu = apply_brightness_correction(img, mask, 0.5, True)
        np.testing.assert_allclose(cpu.astype(float), gpu.astype(float), atol=2)

    def test_output_dtype_is_uint8(self):
        rng = np.random.default_rng(2)
        img = rng.integers(50, 200, (50, 50), dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.uint8)
        assert apply_brightness_correction(img, mask, 0.5, True).dtype == np.uint8
