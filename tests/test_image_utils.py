import numpy as np
import pytest

from AnimalTA.A_General_tools.image_utils import apply_relative_background


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
