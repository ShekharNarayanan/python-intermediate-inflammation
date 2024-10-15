"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest




@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 3], [2, 6], [5, 4] ], [5, 6]),
        ([ [1, 3], [-2, 6], [0, 4] ], [1, 6]),
    ])
def test_daily_max(test, expected):
    """Test that max function works for positive/negative (?) integers"""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(test), expected)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 3], [2, 6], [5, 4] ], [1, 3]),
        ([ [1, 3], [-2, 6], [0, 4] ], [-2, 3]),
    ])
def test_daily_min(test, expected):
    """Test that max function works for positive/negative (?) integers"""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(test), expected)

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test), expected)

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None
        ),
        (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None
        ),
        (
                [[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
                [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
            None
        ),
        (
                [[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]],
                [[0.33, 0.67, 1], [0.8, 1, 0], [0.78, 0.89, 1]],
            None
        ),

        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None
        ),  # previous test cases here, with None for expect_raises, except for the next one - add ValueError as an expected exception (since it has a negative input value)
            (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
                None,
        ),
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            result = patient_normalise(np.array(test))
            npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
@pytest.mark.parametrize('data, expected_standard_deviation, expect_raises',
[
    ([0, 0, 0], 0.0, None),
    ([1.0, 1.0, 1.0], 0, None),
    ([1.0, 2.0, 3.0, float('nan')], 0.8164, None),
    ([-1.0, 2.0], 1.5, None),
])

def test_daily_standard_deviation(data, expected_standard_deviation, expect_raises):
    from inflammation.models import standard_deviation

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            result_data = standard_deviation(data)
            npt.assert_approx_equal(result_data, expected_standard_deviation, significant = 2)
    else:
        result_data = standard_deviation(data)
        npt.assert_approx_equal(result_data, expected_standard_deviation, significant = 1e-2)
