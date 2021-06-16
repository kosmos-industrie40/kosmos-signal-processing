"""
Defines fixtures for the fourier module tests.
"""
import pytest
import numpy as np


@pytest.fixture
def raw_signal_even():
    """
    Fixture to create an example signal with even number of points
    """
    signal_x = np.linspace(0, 2 * np.pi, 1000)
    signal_y = (
        np.sin(10 * signal_x)
        + np.sin(50 * signal_x)
        + np.sin(60 * signal_x)
        + np.sin(100 * signal_x)
        + 2
    )
    return signal_y


@pytest.fixture
def raw_signal_odd():
    """
    Fixture to create an example signal with odd number of points
    """
    signal_x = np.linspace(0, 2 * np.pi, 1001)
    signal_y = np.sin(10 * signal_x) + np.sin(50 * signal_x) + 2
    return signal_y


@pytest.fixture
def mock_find_threshold(mocker, expect):
    """
    for mocking find_threshold() within filter_signal()
    returns parametrized upper_thresh and lower_thresh
    raw_signal does contain these frequencies
    """
    mocker.patch(
        "kosmos_signal_processing.fourier.fft.find_thresholds", return_value=expect
    )
