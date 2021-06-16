"""
Module to test trig.py module from ../src
"""
import pytest
import numpy as np

from kosmos_signal_processing.interpolate.trig import (
    e_complex_pow,
    get_complex_representation,
    get_sin_cos_representation,
    eval_sin_cos_representation,
)


def test_e_complex_pow():
    """
    Tests the creation of a complex number using Euler's formula for 0 value
    """
    assert e_complex_pow(0) == 1 + 0j


@pytest.fixture
def points():
    """
    Fixture to create polynomial values
    """
    return [2, 3, 4]


@pytest.fixture
def ck_s():
    """
    Fixture to create example Fourier coefficients
    """
    return [-0.5 - 0.288675j, 3, -0.5 + 0.288675j]


@pytest.fixture
def ak_expected():
    """
    Fixture to create example cosine coefficients
    """
    return [6, -1]


@pytest.fixture
def bk_expected():
    """
    Fixture to create example sine coefficients
    """
    return [0, -0.57735]


def test_get_complex_representation(points, ck_s):
    """
    Tests if Fourier coefficients calculated by the function correspond to ones calculated by hand.
    """
    np.testing.assert_allclose(get_complex_representation(points), ck_s, atol=1e-6)


def test_get_sin_cos_representation(points, ak_expected, bk_expected):
    """
    Tests if cosine, sine coefficients calculated by the function
    correspond to ones calculated by hand.
    """
    ak_s, bk_s = get_sin_cos_representation(points)
    np.testing.assert_allclose(ak_s, ak_expected)
    np.testing.assert_allclose(bk_s, bk_expected, atol=1e-6)


def test_eval_sin_cos_representation(ak_expected, bk_expected):
    """
    Tests if Fourier series calculated by the function correspond to the number calculated by hand.
    """
    period = 2
    np.testing.assert_array_almost_equal(
        eval_sin_cos_representation(period, ak_expected, bk_expected), 2.8911, decimal=2
    )
