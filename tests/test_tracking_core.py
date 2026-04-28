"""Tests for core tracking/image-processing functions.

Run with: python -m pytest tests/test_tracking_core.py -v
"""
import types
import numpy as np
import cv2
import pytest

# ---------------------------------------------------------------------------
# image_utils.apply_relative_background
# ---------------------------------------------------------------------------

from AnimalTA.A_General_tools.image_utils import apply_relative_background


class TestApplyRelativeBackground:
    def _make(self, h=4, w=4, dtype=np.uint8):
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (h, w), dtype=dtype)

    def test_output_dtype_is_uint8(self):
        img = self._make()
        bg = self._make()
        result = apply_relative_background(img, bg)
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self):
        img = self._make(8, 6)
        bg = self._make(8, 6)
        result = apply_relative_background(img, bg)
        assert result.shape == img.shape

    def test_output_values_in_uint8_range(self):
        img = self._make()
        bg = self._make()
        result = apply_relative_background(img, bg)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_zero_background_no_divide_by_zero(self):
        img = np.full((4, 4), 128, dtype=np.uint8)
        bg = np.zeros((4, 4), dtype=np.uint8)
        # Must not raise; zero background is clamped to 1.
        result = apply_relative_background(img, bg)
        assert result.dtype == np.uint8

    def test_zero_background_treated_as_one(self):
        # bg=0 → safe_bg=1 → scaled = img*255//1 → clipped to 255
        img = np.array([[100, 200], [0, 50]], dtype=np.uint8)
        bg = np.zeros((2, 2), dtype=np.uint8)
        result = apply_relative_background(img, bg)
        expected_raw = (img.astype(np.uint32) * 255) // 1
        expected = np.clip(expected_raw, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_background_equal_to_image_gives_255(self):
        v = np.array([[100]], dtype=np.uint8)
        result = apply_relative_background(v, v)
        assert result[0, 0] == 255

    def test_background_larger_than_image_gives_less_than_255(self):
        img = np.array([[100]], dtype=np.uint8)
        bg = np.array([[200]], dtype=np.uint8)
        result = apply_relative_background(img, bg)
        assert result[0, 0] < 255

    def test_black_image_gives_all_zeros(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        bg = self._make()
        result = apply_relative_background(img, bg)
        np.testing.assert_array_equal(result, 0)

    def test_computation_correctness(self):
        img = np.array([[50, 100], [0, 255]], dtype=np.uint8)
        bg = np.array([[100, 200], [1, 255]], dtype=np.uint8)
        result = apply_relative_background(img, bg)
        safe_bg = np.maximum(bg.astype(np.uint32), 1)
        expected = np.clip((img.astype(np.uint32) * 255) // safe_bg, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_all_same_values_symmetric(self):
        # Every pixel the same → result should be uniform.
        img = np.full((3, 3), 50, dtype=np.uint8)
        bg = np.full((3, 3), 100, dtype=np.uint8)
        result = apply_relative_background(img, bg)
        assert np.all(result == result[0, 0])


# ---------------------------------------------------------------------------
# Function_prepare_images.filter_cnts
# ---------------------------------------------------------------------------

from AnimalTA.D_Tracking_process.Function_prepare_images import filter_cnts


def _make_vid(scale=0.0, min_area=0.0, max_area=1e9):
    """Build a minimal mock Vid object accepted by filter_cnts."""
    vid = types.SimpleNamespace()
    vid.Scale = [scale]
    # Track[1][3] = [min_area, max_area]
    vid.Track = [None, [None, None, None, [min_area, max_area]]]
    return vid


def _rect_contour(w, h):
    """Return an OpenCV-style contour for a w×h rectangle (area = w*h)."""
    return np.array([
        [[0, 0]], [[w, 0]], [[w, h]], [[0, h]]
    ], dtype=np.int32)


class TestFilterCnts:
    def test_empty_input_returns_empty(self):
        vid = _make_vid(min_area=0, max_area=1000)
        assert filter_cnts([], vid) == []

    def test_all_pass_when_no_scale_and_wide_bounds(self):
        cnts = [_rect_contour(10, 10), _rect_contour(5, 5)]
        vid = _make_vid(scale=0.0, min_area=0, max_area=1e9)
        result = filter_cnts(cnts, vid)
        assert len(result) == 2

    def test_too_small_rejected(self):
        cnts = [_rect_contour(2, 2)]  # area ≈ 4
        vid = _make_vid(scale=0.0, min_area=50, max_area=1e9)
        assert filter_cnts(cnts, vid) == []

    def test_too_large_rejected(self):
        cnts = [_rect_contour(100, 100)]  # area ≈ 10000
        vid = _make_vid(scale=0.0, min_area=0, max_area=100)
        assert filter_cnts(cnts, vid) == []

    def test_sorted_largest_first(self):
        small = _rect_contour(5, 5)    # area ≈ 25
        large = _rect_contour(20, 20)  # area ≈ 400
        vid = _make_vid(scale=0.0, min_area=0, max_area=1e9)
        result = filter_cnts([small, large], vid)
        area_first = cv2.contourArea(result[0])
        area_second = cv2.contourArea(result[1])
        assert area_first >= area_second

    def test_sorted_largest_first_reversed_input(self):
        small = _rect_contour(5, 5)
        large = _rect_contour(20, 20)
        vid = _make_vid(scale=0.0, min_area=0, max_area=1e9)
        result_a = filter_cnts([small, large], vid)
        result_b = filter_cnts([large, small], vid)
        np.testing.assert_array_equal(result_a[0], result_b[0])

    def test_scale_converts_area(self):
        # With scale=2 (2 px per unit), pixel area is divided by 4.
        # A 10×10 px rect has pixel area ≈ 100 → unit area = 100*(1/2)^2 = 25.
        cnt = _rect_contour(10, 10)
        vid_pass = _make_vid(scale=2.0, min_area=20, max_area=30)
        vid_fail = _make_vid(scale=2.0, min_area=200, max_area=300)
        assert len(filter_cnts([cnt], vid_pass)) == 1
        assert len(filter_cnts([cnt], vid_fail)) == 0

    def test_exact_boundary_min_included(self):
        cnt = _rect_contour(10, 10)
        pixel_area = cv2.contourArea(cnt)
        vid = _make_vid(scale=0.0, min_area=pixel_area, max_area=1e9)
        assert len(filter_cnts([cnt], vid)) == 1

    def test_exact_boundary_max_included(self):
        cnt = _rect_contour(10, 10)
        pixel_area = cv2.contourArea(cnt)
        vid = _make_vid(scale=0.0, min_area=0, max_area=pixel_area)
        assert len(filter_cnts([cnt], vid)) == 1

    def test_only_in_range_kept(self):
        tiny = _rect_contour(2, 2)    # area ≈ 4
        medium = _rect_contour(10, 10)  # area ≈ 100
        huge = _rect_contour(50, 50)  # area ≈ 2500
        vid = _make_vid(scale=0.0, min_area=50, max_area=500)
        result = filter_cnts([tiny, medium, huge], vid)
        assert len(result) == 1
        assert cv2.contourArea(result[0]) == pytest.approx(cv2.contourArea(medium), rel=0.05)
