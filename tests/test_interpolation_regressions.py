"""Regression tests for the post-tracking interpolation workflow.

The first test deliberately implements the pre-fix condition from
``Interface_Check.Lecteur.interpolate``.  That makes the original silent
no-op observable instead of only testing that a button is wired to a method.
"""

import numpy as np
from pathlib import Path

from AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check import (
    prepare_interpolation,
)
from AnimalTA.E_Post_tracking.a_Tracking_verification import Interpolate_all


def _old_interpolate(coos, selected_ind, selected_rows):
    """The old local interpolation body, kept only as a regression oracle."""
    if len(selected_rows) > 2:
        if (
            coos[selected_ind, selected_rows[0], 0] != -1000
            or coos[selected_ind, selected_rows[-1], 0] != -1000
        ):
            first = int(selected_rows[0])
            last = int(selected_rows[-1])
            add = 0
            if coos[selected_ind, selected_rows[0], 0] == -1000:
                first = last
            elif coos[selected_ind, selected_rows[-1], 0] == -1000:
                last = first
                add = 1

            for raw in selected_rows[0 : len(selected_rows) - 1 + add]:
                raw = int(raw)
                coos[selected_ind, raw, 0] = coos[selected_ind, first, 0] + (
                    (coos[selected_ind, last, 0] - coos[selected_ind, first, 0])
                    * ((raw - first) / (len(selected_rows) - 1))
                )
                coos[selected_ind, raw, 1] = coos[selected_ind, first, 1] + (
                    (coos[selected_ind, last, 1] - coos[selected_ind, first, 1])
                    * ((raw - first) / (len(selected_rows) - 1))
                )


def _apply_plan(coos, plan, selected_ind=0):
    if plan.get("changed"):
        coos[selected_ind, plan["start"] : plan["end"]] = plan["replacement"]


def test_old_handler_silently_noops_when_selection_edges_are_missing():
    # This is a realistic gap selection: the visible trajectory has usable
    # points inside the selected interval, but both selected edges are NA.
    coos = np.full((1, 5, 3), -1000.0)
    coos[0, 1, :2] = (10, 10)
    coos[0, 3, :2] = (30, 30)
    selected = [0, 1, 2, 3, 4]

    old_result = coos.copy()
    _old_interpolate(old_result, 0, selected)
    assert np.array_equal(old_result, coos), "the historical silent no-op is reproduced"

    plan = prepare_interpolation(coos, 0, selected)
    new_result = coos.copy()
    _apply_plan(new_result, plan)

    assert plan["changed"] == 1
    np.testing.assert_allclose(new_result[0, 2, :2], (20, 20))
    assert np.all(new_result[0, 0, :2] == -1000)
    assert np.all(new_result[0, 4, :2] == -1000)


def test_local_interpolation_only_changes_missing_interior_rows():
    coos = np.array(
        [[[0, 0, 7], [10, 10, 8], [-1000, -1000, 9], [30, 30, 10], [40, 40, 11]]],
        dtype=float,
    )
    original = coos.copy()
    plan = prepare_interpolation(coos, 0, [0, 1, 2, 3, 4])
    _apply_plan(coos, plan)

    assert plan["changed"] == 1
    np.testing.assert_array_equal(coos[0, 0], original[0, 0])
    np.testing.assert_array_equal(coos[0, 1], original[0, 1])
    np.testing.assert_array_equal(coos[0, 3], original[0, 3])
    np.testing.assert_allclose(coos[0, 2, :2], (20, 20))
    assert coos[0, 2, 2] == original[0, 2, 2]


def test_selecting_only_the_missing_rows_uses_neighbouring_anchors():
    coos = np.array(
        [[[10, 10, 0], [-1000, -1000, 0], [30, 30, 0]]], dtype=float
    )
    plan = prepare_interpolation(coos, 0, [1])

    assert plan["changed"] == 1
    _apply_plan(coos, plan)
    np.testing.assert_allclose(coos[0, 1, :2], (20, 20))


def test_local_interpolation_reports_no_change_when_no_gap_exists():
    coos = np.array([[[0, 0, 1], [10, 10, 2], [20, 20, 3]]], dtype=float)
    plan = prepare_interpolation(coos, 0, [0, 1, 2])

    assert plan == {"changed": 0, "reason": "no_missing_interior"}


def test_all_video_interpolation_fills_internal_gaps_only(monkeypatch):
    coos = np.array(
        [
            [
                [-1000, -1000, 0],
                [10, 10, 1],
                [-1000, -1000, 2],
                [30, 30, 3],
                [-1000, -1000, 4],
            ]
        ],
        dtype=float,
    )
    saved = []
    monkeypatch.setattr(Interpolate_all.CoosLS, "load_coos", lambda _vid: (coos, []))
    monkeypatch.setattr(Interpolate_all.CoosLS, "save", lambda _vid, value: saved.append(value.copy()))

    changed = Interpolate_all.interpolate_all(object())

    assert changed == 1
    np.testing.assert_allclose(coos[0, 2, :2], (20, 20))
    assert np.all(coos[0, 0, :2] == -1000)
    assert np.all(coos[0, 4, :2] == -1000)
    assert len(saved) == 1


def test_interpolation_undo_restores_coordinates_and_refreshes_membership():
    # Exercise the actual Lecteur.remove_last method without constructing Tk.
    from AnimalTA.E_Post_tracking.a_Tracking_verification.Interface_Check import Lecteur

    original = np.array(
        [[[0, 0, 0], [-1000, -1000, 0], [20, 20, 0]]], dtype=float
    )
    plan = prepare_interpolation(original, 0, [0, 1, 2])
    changed = original.copy()
    _apply_plan(changed, plan)

    reader = object.__new__(Lecteur)
    reader.Coos = changed
    reader.save_changes = [["interpolate", 0, [plan["start"], plan["end"]], plan["old"]]]
    reader.copied_cells = []
    refreshed = []
    reader.redo_who_is_here = lambda: refreshed.append("membership")
    reader.afficher_table = lambda **_kwargs: refreshed.append("table")
    reader.modif_image = lambda: refreshed.append("image")
    reader.calculate_NA = lambda: refreshed.append("na")

    reader.remove_last()

    np.testing.assert_array_equal(reader.Coos, original)
    assert reader.save_changes == []
    assert refreshed[:2] == ["membership", "table"]


def test_short_selection_keeps_the_all_video_interpolation_workflow():
    source = Path(
        "AnimalTA/E_Post_tracking/a_Tracking_verification/Interface_Check.py"
    ).read_text(encoding="utf-8")

    assert 'if len(self.selected_rows) <= 2:' in source
    assert "Interpolate_all.interpolate_all(video)" in source
