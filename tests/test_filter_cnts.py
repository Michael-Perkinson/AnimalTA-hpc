import types

import cv2
import numpy as np
import pytest

from AnimalTA.D_Tracking_process.Function_prepare_images import filter_cnts


def _make_vid(scale=0.0, min_area=0.0, max_area=1e9):
    vid = types.SimpleNamespace()
    vid.Scale = [scale]
    vid.Track = [None, [None, None, None, [min_area, max_area]]]
    return vid


def _rect_contour(w, h):
    return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)


class TestFilterCnts:
    def test_empty_input_returns_empty(self):
        assert filter_cnts([], _make_vid()) == []

    def test_all_pass_wide_bounds(self):
        cnts = [_rect_contour(10, 10), _rect_contour(5, 5)]
        assert len(filter_cnts(cnts, _make_vid(max_area=1e9))) == 2

    def test_too_small_rejected(self):
        assert filter_cnts([_rect_contour(2, 2)], _make_vid(min_area=50)) == []

    def test_too_large_rejected(self):
        assert filter_cnts([_rect_contour(100, 100)], _make_vid(max_area=100)) == []

    def test_sorted_largest_first(self):
        small = _rect_contour(5, 5)
        large = _rect_contour(20, 20)
        result = filter_cnts([small, large], _make_vid())
        assert cv2.contourArea(result[0]) >= cv2.contourArea(result[1])

    def test_sort_order_independent_of_input_order(self):
        small = _rect_contour(5, 5)
        large = _rect_contour(20, 20)
        vid = _make_vid()
        a = filter_cnts([small, large], vid)
        b = filter_cnts([large, small], vid)
        np.testing.assert_array_equal(a[0], b[0])

    def test_scale_converts_pixel_area_to_unit_area(self):
        # scale=2 px/unit → pixel area * (1/2)^2 → 10×10 rect = 25 unit²
        cnt = _rect_contour(10, 10)
        assert len(filter_cnts([cnt], _make_vid(scale=2.0, min_area=20, max_area=30))) == 1
        assert len(filter_cnts([cnt], _make_vid(scale=2.0, min_area=200, max_area=300))) == 0

    def test_boundary_min_included(self):
        cnt = _rect_contour(10, 10)
        area = cv2.contourArea(cnt)
        assert len(filter_cnts([cnt], _make_vid(min_area=area))) == 1

    def test_boundary_max_included(self):
        cnt = _rect_contour(10, 10)
        area = cv2.contourArea(cnt)
        assert len(filter_cnts([cnt], _make_vid(max_area=area))) == 1

    def test_only_in_range_kept(self):
        tiny = _rect_contour(2, 2)
        medium = _rect_contour(10, 10)
        huge = _rect_contour(50, 50)
        result = filter_cnts([tiny, medium, huge], _make_vid(min_area=50, max_area=500))
        assert len(result) == 1
        assert cv2.contourArea(result[0]) == pytest.approx(cv2.contourArea(medium), rel=0.05)
