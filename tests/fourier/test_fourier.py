"""
Test module for fourier module within ../src
"""

import pytest
import numpy as np
from numpy.fft import rfft

from kosmos_signal_processing.fourier import find_thresholds, filter_signal

# signal and boundary values for thresholds
params_find_thresholds = [
    ("raw_signal_even", 50, 10, 60),
    ("raw_signal_odd", 10, 0, 50),
]


@pytest.mark.parametrize(
    "signal, peak, prev_loc_max, next_loc_max", params_find_thresholds
)
def test_find_thresholds(signal, peak, prev_loc_max, next_loc_max, request):
    """
    According to graphic representation thresholds should be located
    after the previous and before next local maxima.
    """
    thresh1, thresh2 = find_thresholds(request.getfixturevalue(signal))
    assert prev_loc_max < thresh1 < peak & peak < thresh2 < next_loc_max


# signal and expected result of find_threshold()
param_filter_signal_even = [("raw_signal_even", (30, 80)), ("raw_signal_odd", (20, 60))]


@pytest.mark.usefixtures("mock_find_threshold")
@pytest.mark.parametrize("signal, expect", param_filter_signal_even)
def test_filter_signal_even(signal, expect, request):
    """
    Test filter_signal().
    raw_signal has frequencies < upper_thresh and > lower_thresh.
    Assert that signal does not contain filtered frequencies anymore after filtering.
    """
    lower_thresh, upper_thresh = expect
    filtered_signal = filter_signal(request.getfixturevalue(signal))
    np.testing.assert_almost_equal(
        rfft(filtered_signal)[lower_thresh + 1 : upper_thresh - 1], 0
    )
