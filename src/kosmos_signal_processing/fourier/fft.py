"""
This module provides functions in order to use a bandstop filter (inverse bandpass).
"""

from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from numpy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks


def find_thresholds(signal: ArrayLike) -> Tuple[float, float]:
    """
    Function enabling scipy peak detection algorithm to identify threshold window
    for the fourier transformation.

    :param signal: signal represented by sequence of numbers
    :return: lower & upper threshold
    """
    # numpy.fft.rfft - Computes the one-dimensional discrete Fourier Transform (DFT) for real input.
    fourier = rfft(signal)
    peaks, _ = find_peaks(np.abs(fourier), distance=len(fourier))
    valleys, _ = find_peaks(-np.abs(fourier))

    # since within a complex value, there lies information within both the imaginary and the real
    # part of the value. In order to account for both parts, np.abs() is used
    thresh1 = (np.abs(valleys - peaks[0])).argmin()
    if valleys[thresh1] < peaks[0]:
        thresh2 = valleys[thresh1 + 1]
        thresh1 = valleys[thresh1]
    else:
        thresh2 = valleys[thresh1 - 1]
        thresh1 = valleys[thresh1]
        thresh1, thresh2 = thresh2, thresh1
    return thresh1, thresh2


def filter_signal(signal: ArrayLike) -> np.ndarray:
    """
    Function to filter out signals which are between the upper and the lower threshold.

    :param signal: signal represented by sequence of numbers
    :return: filtered signal
    """
    # In order to account for characteristic of irfft()
    # (rounds and thus creates ValueError if len(signal) % 2 == 1)
    if len(signal) % 2 != 0:
        signal = signal[1:]

    fourier = rfft(signal)

    lower_thresh, upper_thresh = find_thresholds(signal)
    # numpy.fft.rfftfreq() - Returns the DFT sample frequencies.
    frequencies = rfftfreq(signal.size, d=20e-3 / signal.size)
    lower_thresh = frequencies[lower_thresh]
    upper_thresh = frequencies[upper_thresh]
    # Setting all frequencies to 0 which are smaller than
    fourier[(frequencies < upper_thresh) & (frequencies > lower_thresh)] = 0

    # numpy.fft.irfft() - Computes the inverse of the n-point DFT for real input.
    return irfft(fourier)
