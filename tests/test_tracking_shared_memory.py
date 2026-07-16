"""Regression tests for raw-frame shared-memory sizing."""

import ast
from pathlib import Path

import numpy as np
import pytest


TRACKING_MODULE = (
    Path(__file__).resolve().parents[1]
    / "AnimalTA"
    / "D_Tracking_process"
    / "Do_the_track_multi.py"
)


def _tracking_tree():
    return ast.parse(
        TRACKING_MODULE.read_text(encoding="utf-8"), filename=str(TRACKING_MODULE)
    )


def _load_shape_helpers():
    tree = _tracking_tree()
    helper_names = {"_decoded_frame_shape", "_validate_reader_frame_shape"}
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(TRACKING_MODULE), "exec"),
        namespace,
    )
    return namespace


def test_raw_decoded_shape_is_not_replaced_by_spatial_crop_shape():
    helpers = _load_shape_helpers()
    raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cropped_video_shape = (650, 1735, 3)

    frame_shape = helpers["_decoded_frame_shape"](raw_frame)

    assert frame_shape == (1080, 1920, 3)
    assert frame_shape != cropped_video_shape


def test_reader_shape_validation_accepts_matching_raw_frame():
    helpers = _load_shape_helpers()
    raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    helpers["_validate_reader_frame_shape"](
        raw_frame, (1080, 1920, 3), frame=42, reader_name="reader"
    )


def test_reader_shape_validation_reports_mismatched_source_segment():
    helpers = _load_shape_helpers()
    raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="shared-memory pool expects"):
        helpers["_validate_reader_frame_shape"](
            raw_frame, (650, 1735, 3), frame=42, reader_name="reader"
        )


def test_tracking_pool_and_worker_estimate_use_decoded_raw_shape():
    tree = _tracking_tree()
    do_tracking = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "Do_tracking"
    )

    raw_shape_assignment = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "raw_frame_shape"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_decoded_frame_shape"
        for node in ast.walk(do_tracking)
    )
    worker_count_uses_raw_shape = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_safe_worker_count"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "raw_frame_shape"
        for node in ast.walk(do_tracking)
    )
    pool_uses_raw_shape = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_frame_shape"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "raw_frame_shape"
        for node in ast.walk(do_tracking)
    )

    assert raw_shape_assignment
    assert worker_count_uses_raw_shape
    assert pool_uses_raw_shape
