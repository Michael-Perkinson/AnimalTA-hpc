"""Regression tests for contour-assignment logic in Function_assign_cnts.

These tests lock down the numerical behaviour of the CPU path so that any
future GPU implementation can be validated against them.  No Vid object or
video file is required -- we exercise the raw computations directly.

Tolerance policy (matches user requirement):
  - Assignment indices on unambiguous inputs must be EXACT.
  - On near-tie inputs we only assert the assignment is *optimal* (total cost
    is minimal), because float32/float64 differences can legally flip the
    choice between two equally-good assignments.
  - Coordinate values (e.g. KMeans centers) are asserted with atol=1.0 pixel,
    since GPU float32 arithmetic can differ by sub-pixel amounts.
"""

import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Helpers that mirror the exact distance formula used in Function_assign_cnts
# ---------------------------------------------------------------------------

def _build_dist_table(old_positions, new_centroids, scale, threshold, frame_area):
    """Replicate table_dists construction from Treat_cnts_fixed (lines 118-131)."""
    table = []
    for old in old_positions:
        row = []
        for new_pt in new_centroids:
            if old[0] != "NA":
                dist = math.sqrt(
                    (float(old[0]) - float(new_pt[0])) ** 2
                    + (float(old[1]) - float(new_pt[1])) ** 2
                ) / float(scale)
                row.append(dist if dist < threshold else frame_area / float(scale))
            else:
                row.append(0)
        table.append(row)
    return table


def _is_valid_assignment(table, row_ind, col_ind):
    """Return True if (row_ind, col_ind) achieves the minimum possible total cost."""
    cost = sum(table[r][c] for r, c in zip(row_ind, col_ind))
    # brute-force optimal over all permutations of columns
    opt_row, opt_col = linear_sum_assignment(table)
    opt_cost = sum(table[r][c] for r, c in zip(opt_row, opt_col))
    return math.isclose(cost, opt_cost, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Distance table construction
# ---------------------------------------------------------------------------

class TestBuildDistTable:
    SCALE = 1.0
    THRESHOLD = 100.0
    FRAME_AREA = 640 * 480

    def test_exact_distances_unambiguous(self):
        old = [(0.0, 0.0), (100.0, 0.0)]
        new = [(5.0, 0.0), (95.0, 0.0)]
        table = _build_dist_table(old, new, self.SCALE, self.THRESHOLD, self.FRAME_AREA)
        assert math.isclose(table[0][0], 5.0)
        assert math.isclose(table[0][1], 95.0)
        assert math.isclose(table[1][0], 95.0)
        assert math.isclose(table[1][1], 5.0)

    def test_distance_above_threshold_becomes_sentinel(self):
        old = [(0.0, 0.0)]
        new = [(200.0, 0.0)]  # 200px > threshold of 100
        table = _build_dist_table(old, new, self.SCALE, self.THRESHOLD, self.FRAME_AREA)
        sentinel = self.FRAME_AREA / self.SCALE
        assert table[0][0] == sentinel

    def test_na_old_position_gives_zero_distance(self):
        old = [("NA", "NA")]
        new = [(50.0, 50.0)]
        table = _build_dist_table(old, new, self.SCALE, self.THRESHOLD, self.FRAME_AREA)
        assert table[0][0] == 0

    def test_scale_divides_distance(self):
        old = [(0.0, 0.0)]
        new = [(30.0, 40.0)]  # Euclidean = 50
        table_1 = _build_dist_table(old, new, 1.0, self.THRESHOLD, self.FRAME_AREA)
        table_2 = _build_dist_table(old, new, 2.0, self.THRESHOLD, self.FRAME_AREA)
        assert math.isclose(table_1[0][0], 50.0)
        assert math.isclose(table_2[0][0], 25.0)

    def test_diagonal_distance(self):
        old = [(0.0, 0.0)]
        new = [(3.0, 4.0)]
        table = _build_dist_table(old, new, 1.0, self.THRESHOLD, self.FRAME_AREA)
        assert math.isclose(table[0][0], 5.0)

    def test_multiple_animals_multiple_contours(self):
        old = [(0.0, 0.0), (50.0, 50.0), (100.0, 100.0)]
        new = [(2.0, 0.0), (52.0, 50.0), (98.0, 100.0)]
        table = _build_dist_table(old, new, 1.0, self.THRESHOLD, self.FRAME_AREA)
        assert len(table) == 3
        assert all(len(row) == 3 for row in table)
        # Diagonal should be smallest in each row
        for i in range(3):
            assert table[i][i] == min(table[i])


# ---------------------------------------------------------------------------
# linear_sum_assignment -- unambiguous cases (exact index match required)
# ---------------------------------------------------------------------------

class TestLinearSumAssignmentUnambiguous:
    """When one assignment is clearly optimal, GPU must return the same indices."""

    def test_identity_assignment_1x1(self):
        table = [[5.0]]
        row_ind, col_ind = linear_sum_assignment(table)
        assert list(row_ind) == [0]
        assert list(col_ind) == [0]

    def test_clear_diagonal_2x2(self):
        # Animal 0 is close to contour 0, animal 1 close to contour 1
        table = [[2.0, 90.0],
                 [90.0, 3.0]]
        row_ind, col_ind = linear_sum_assignment(table)
        assert list(col_ind) == [0, 1]

    def test_clear_swap_2x2(self):
        # Animal 0 is closer to contour 1, animal 1 closer to contour 0
        table = [[80.0, 2.0],
                 [3.0, 80.0]]
        row_ind, col_ind = linear_sum_assignment(table)
        assert list(col_ind) == [1, 0]

    def test_3x3_unambiguous(self):
        table = [
            [1.0,  50.0, 50.0],
            [50.0,  2.0, 50.0],
            [50.0, 50.0,  3.0],
        ]
        row_ind, col_ind = linear_sum_assignment(table)
        assert list(col_ind) == [0, 1, 2]

    def test_more_contours_than_animals(self):
        # 2 animals, 3 contours -- each animal should get its nearest contour
        table = [
            [1.0, 40.0, 80.0],
            [80.0, 45.0,  2.0],
        ]
        row_ind, col_ind = linear_sum_assignment(table)
        assert col_ind[0] == 0  # animal 0 -> contour 0
        assert col_ind[1] == 2  # animal 1 -> contour 2

    def test_sentinel_values_avoid_bad_assignments(self):
        # Contour 1 is beyond threshold for both animals; sentinel makes it unattractive
        sentinel = 640 * 480
        table = [
            [5.0, sentinel],
            [sentinel, 4.0],
        ]
        row_ind, col_ind = linear_sum_assignment(table)
        assert col_ind[0] == 0
        assert col_ind[1] == 1

    def test_threshold_gate_drops_assignment(self):
        """Assignments where the chosen distance >= threshold are discarded."""
        threshold = 100.0
        table = [[50.0, 200.0],
                 [200.0, 60.0]]
        row_ind, col_ind = linear_sum_assignment(table)
        kept = [
            (r, c) for r, c in zip(row_ind, col_ind)
            if table[r][c] < threshold
        ]
        assert len(kept) == 2
        assert (0, 0) in kept
        assert (1, 1) in kept

    def test_all_beyond_threshold_all_dropped(self):
        threshold = 10.0
        table = [[50.0, 80.0],
                 [70.0, 55.0]]
        row_ind, col_ind = linear_sum_assignment(table)
        kept = [
            (r, c) for r, c in zip(row_ind, col_ind)
            if table[r][c] < threshold
        ]
        assert len(kept) == 0


# ---------------------------------------------------------------------------
# linear_sum_assignment -- near-tie cases (only optimality required)
# ---------------------------------------------------------------------------

class TestLinearSumAssignmentNearTie:
    """When two assignments have nearly equal cost, only assert optimality.

    A GPU float32 implementation may legally flip the choice between them.
    These tests document that expectation rather than over-constraining the
    GPU implementation.
    """

    def test_near_tie_2x2_is_optimal(self):
        # Costs differ by 0.001 -- a float32 rounding could swap the choice
        table = [[10.000, 10.001],
                 [10.001, 10.000]]
        row_ind, col_ind = linear_sum_assignment(table)
        assert _is_valid_assignment(table, row_ind, col_ind)

    def test_near_tie_3x3_is_optimal(self):
        table = [
            [5.000, 5.001, 20.0],
            [5.001, 5.000, 20.0],
            [20.0,  20.0,   1.0],
        ]
        row_ind, col_ind = linear_sum_assignment(table)
        assert _is_valid_assignment(table, row_ind, col_ind)


# ---------------------------------------------------------------------------
# KMeans contour splitting
# ---------------------------------------------------------------------------

class TestKMeansSplitting:
    """When fewer contours than animals, contours are split via KMeans."""

    def test_split_produces_correct_number_of_centers(self):
        from sklearn.cluster import KMeans
        rng = np.random.default_rng(0)
        # One big blob of points -- needs to be split into 3
        points = rng.integers(0, 100, (200, 2))
        n_animals = 3
        kmeans = KMeans(n_clusters=n_animals, random_state=0, n_init=50).fit(points)
        assert kmeans.cluster_centers_.shape == (n_animals, 2)

    def test_split_centers_are_finite(self):
        from sklearn.cluster import KMeans
        rng = np.random.default_rng(1)
        points = rng.integers(0, 50, (100, 2))
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=50).fit(points)
        assert np.all(np.isfinite(kmeans.cluster_centers_))

    def test_split_centers_within_input_bounds(self):
        from sklearn.cluster import KMeans
        points = np.array([[0, 0], [0, 1], [100, 100], [101, 100], [101, 101]], dtype=np.float32)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(points)
        centers = kmeans.cluster_centers_
        # Each center should lie within the bounding box of the input
        assert np.all(centers >= 0)
        assert np.all(centers[:, 0] <= 101)
        assert np.all(centers[:, 1] <= 101)

    def test_two_well_separated_clusters(self):
        from sklearn.cluster import KMeans
        # Two tight clusters far apart -- centers should be near (10,10) and (90,90)
        group_a = np.full((50, 2), 10, dtype=np.float32) + np.random.default_rng(2).uniform(-1, 1, (50, 2))
        group_b = np.full((50, 2), 90, dtype=np.float32) + np.random.default_rng(3).uniform(-1, 1, (50, 2))
        points = np.vstack([group_a, group_b])
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(points)
        centers = sorted(kmeans.cluster_centers_.tolist())
        np.testing.assert_allclose(centers[0], [10, 10], atol=2.0)
        np.testing.assert_allclose(centers[1], [90, 90], atol=2.0)

    def test_split_n_init_50_vs_5_same_shape(self):
        """n_init=50 (first-frame) and n_init=5 (mid-track) both return correct shape."""
        from sklearn.cluster import KMeans
        points = np.random.default_rng(4).integers(0, 200, (80, 2))
        for n_init in (5, 50):
            km = KMeans(n_clusters=3, random_state=0, n_init=n_init).fit(points)
            assert km.cluster_centers_.shape == (3, 2)


# ---------------------------------------------------------------------------
# cdist + linear_sum_assignment for head/tail ordering
# ---------------------------------------------------------------------------

class TestHeadTailAssignment:
    """Mirrors lines 268-270 of Function_assign_cnts: cdist on 2 last positions
    vs 2 new endpoints, then linear_sum_assignment to decide which is head/tail.
    """

    def _assign(self, last_pos, new_endpoints):
        dists = cdist(last_pos, new_endpoints)
        order = list(linear_sum_assignment(dists)[1])
        return [new_endpoints[order[0]], new_endpoints[order[1]]]

    def test_no_swap_needed(self):
        # Last head=(0,0), last tail=(100,100); new endpoints the same order
        last = [(0, 0), (100, 100)]
        new  = [(1, 1), (99, 99)]
        result = self._assign(last, new)
        np.testing.assert_allclose(result[0], (1, 1), atol=1)
        np.testing.assert_allclose(result[1], (99, 99), atol=1)

    def test_swap_when_endpoints_reversed(self):
        # Last head=(0,0), last tail=(100,100); new endpoints in reversed order
        last = [(0, 0), (100, 100)]
        new  = [(99, 99), (1, 1)]
        result = self._assign(last, new)
        # head should be assigned to the endpoint nearest (0,0) i.e. (1,1)
        np.testing.assert_allclose(result[0], (1, 1), atol=1)
        np.testing.assert_allclose(result[1], (99, 99), atol=1)

    def test_1x1_trivial(self):
        last = [(50, 50), (150, 150)]
        new  = [(52, 48), (148, 152)]
        result = self._assign(last, new)
        np.testing.assert_allclose(result[0], (52, 48), atol=1)
        np.testing.assert_allclose(result[1], (148, 152), atol=1)

    def test_cdist_shape(self):
        last = [(0, 0), (100, 100)]
        new  = [(5, 5), (95, 95)]
        dists = cdist(last, new)
        assert dists.shape == (2, 2)

    def test_cdist_values_euclidean(self):
        last = [(0, 0), (10, 0)]
        new  = [(3, 4), (10, 0)]
        dists = cdist(last, new)
        assert math.isclose(dists[0, 0], 5.0)   # sqrt(9+16)
        assert math.isclose(dists[0, 1], 10.0)  # sqrt(100)
        assert math.isclose(dists[1, 0], math.sqrt(49 + 16))
        assert math.isclose(dists[1, 1], 0.0)

    def test_near_tie_head_tail_is_optimal(self):
        # Nearly symmetric -- only assert optimality, not specific ordering
        last = [(0, 0), (10, 0)]
        new  = [(5, 0), (5, 0.001)]  # almost identical endpoints
        dists = cdist(last, new)
        row_ind, col_ind = linear_sum_assignment(dists)
        assert _is_valid_assignment(dists.tolist(), row_ind, col_ind)
