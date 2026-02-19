import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra.numpy import arrays

import AnimalTA.E_Post_tracking.b_Analyses.Functions_analyses.Functions_trajectory_summarise as FTS# replace with actual import
import pandas as pd

csv_file = "simplified_coordinates.csv"
df = pd.read_csv(csv_file, sep=";")
# Convert to numpy array (float)
simplified_coos = df.iloc[:, :2].to_numpy(dtype=float)  # shape (N,2)




# Base integers
ints = st.integers(min_value=-100, max_value=100)

# Mix in NaN explicitly
ints_or_nan = st.one_of(
    ints.map(float),            # convert integers to float, so they can coexist with NaN
    st.just(float("nan"))
)

# Strategy for coordinates: at least 1 point, each coord is float in range [-100, 100]
coos_strategy = arrays(
    dtype=float,
    shape=st.tuples(st.integers(min_value=1, max_value=20), st.just(2)),  # (N, 2)
    elements=ints_or_nan
)

# Strategy for scale: positive floats only
scale_strategy = st.lists(
    st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=1
)

# Strategy for scale: positive floats only
frame_rate_strategy = st.lists(
    st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=1
)

# Strategy for scale: positive floats only
mov_threshold_strategy = st.lists(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=1
)




@given(coos=coos_strategy, scale=scale_strategy)
@example(coos=simplified_coos, scale=[1.0])
def test_prepare_dists_properties(coos, scale):
    result = FTS.prepare_Dists(coos, scale)
    # Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: first element should always be NaN
    assert np.isnan(result[0])
    # Invariant 3: all other distances must be >= 0
    assert np.all((result[1:] >= 0) | np.isnan(result[1:]))
    # Invariant 4: scaling works — recompute raw distances manually
    diffs = np.sqrt(np.diff(coos[:, 0]) ** 2 + np.diff(coos[:, 1]) ** 2) / float(scale[0])
    expected = np.concatenate([[np.nan], diffs])
    np.testing.assert_allclose(result, expected, equal_nan=True)

    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 2].to_numpy(dtype=float), equal_nan=True)


@given(coos=coos_strategy, scale=scale_strategy, frame_rate=frame_rate_strategy)
@example(coos=simplified_coos, scale=[1.0], frame_rate=[1])
def test_prepare_Speeds(coos, scale, frame_rate):
    Dists = FTS.prepare_Dists(coos, scale)
    result = FTS.prepare_Speeds(Dists, frame_rate[0])

    #Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: first element should always be NaN
    assert np.isnan(result[0])
    # Invariant 3: all other speeds must be >= 0
    assert np.all((result >= 0) | np.isnan(result))
    # Invariant 4: scaling works — recompute raw distances manually
    speeds=Dists/(1/frame_rate[0])
    np.testing.assert_allclose(result, speeds, equal_nan=True)

    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 3].to_numpy(dtype=float), equal_nan=True)


@given(coos=coos_strategy, scale=scale_strategy, frame_rate=frame_rate_strategy, mov_threshold=mov_threshold_strategy)
@example(coos=simplified_coos, scale=[1.0], frame_rate=[1], mov_threshold=[0.5])
def test_prepare_State(coos, scale, frame_rate, mov_threshold):
    Dists = FTS.prepare_Dists(coos, scale)
    Speeds = FTS.prepare_Speeds(Dists, frame_rate[0])
    result=FTS.prepare_State(Speeds, mov_threshold)

    #Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: first element should always be NaN
    assert np.isnan(result[0])
    # Invariant 3: all other speeds must be >= 0
    assert np.all((result == 1) | (result == 0) | np.isnan(result))
    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 4].to_numpy(dtype=float), equal_nan=True)


@given(coos=coos_strategy, scale=scale_strategy, frame_rate=frame_rate_strategy)
@example(coos=simplified_coos, scale=[1.0], frame_rate=[1])
def test_prepare_Accelerations(coos, scale, frame_rate):
    Dists = FTS.prepare_Dists(coos, scale)
    Speeds = FTS.prepare_Speeds(Dists, frame_rate[0])
    result=FTS.prepare_Acceleration(Speeds, frame_rate[0])

    #Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: first element should always be NaN
    assert np.isnan(result[0])
    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 5].to_numpy(dtype=float), equal_nan=True)


@given(coos=coos_strategy)
@example(coos=simplified_coos)
def test_prepare_Orient(coos):
    result = FTS.prepare_Orient(coos)
    #Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: Angles should stay between 0 and 360
    assert np.all(((result>=-180) & (result<=180)) | np.isnan(result))
    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 6].to_numpy(dtype=float), equal_nan=True)


@given(coos=coos_strategy)
@example(coos=simplified_coos)
def test_prepare_Angles(coos):
    Orient = FTS.prepare_Orient(coos)
    result=FTS.prepare_Angles(Orient)
    #Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: Angle diff should stay between 0 and 180
    assert np.all(((result>=0) & (result<=180)) | np.isnan(result))
    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 7].to_numpy(dtype=float), equal_nan=True)


@given(coos=coos_strategy, frame_rate=frame_rate_strategy)
@example(coos=simplified_coos, frame_rate=[1])
def test_prepare_Angular_speed(coos, frame_rate):
    Orient = FTS.prepare_Orient(coos)
    Angles = FTS.prepare_Angles(Orient)
    result = FTS.prepare_Angular_Speed(Angles, frame_rate[0])
    # Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: Angle diff should never be lower than 0
    assert np.all((result >= 0) | np.isnan(result))
    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 8].to_numpy(dtype=float), equal_nan=True)

@given(coos=coos_strategy, scale=scale_strategy)
@example(coos=simplified_coos, scale=[1.0])
def test_prepare_Meander(coos, scale):
    Orient = FTS.prepare_Orient(coos)
    Dists = FTS.prepare_Dists(coos, scale)
    Angles = FTS.prepare_Angles(Orient)
    result = FTS.prepare_Meander(Angles, Dists)
    # Invariant 1: length should match number of coordinates
    assert result.shape[0] == coos.shape[0]
    # Invariant 2: Angle diff should never be lower than 0
    assert np.all((result >= 0) | np.isnan(result))
    if np.array_equal(coos, simplified_coos):
        np.testing.assert_allclose(result, df.iloc[:, 9].to_numpy(dtype=float), equal_nan=True)
