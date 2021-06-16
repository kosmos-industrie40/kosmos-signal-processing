"""
Module to test interpol.py module from ../src
"""

import pytest
import pandas as pd
import numpy as np

from kosmos_signal_processing.interpolate import (
    selection_mask,
    do_interpolate,
    interpol_method_selection,
)


def test_selection_mask(sel_mask_data):
    """
    Fixture to create example dataframe selection mask
    """
    data = selection_mask(sel_mask_data)
    assert len(data[0].dropna().unique()) == 100


# Parameterization to test interpolation methods individually
param_array = [
    ("polynomial", 1),
    ("polynomial", 3),
    ("polynomial", 5),
    ("spline", 1),
    ("spline", 3),
    ("spline", 5),
    ("nearest", None),
]


@pytest.mark.parametrize("method,order", param_array)
def test_do_interpolate(my_data, method, order):
    """
    Checks if calculated loss of scipy interpolation less than infinity is
    """
    loss = do_interpolate(my_data, method=method, order=order)
    assert 0 < loss < np.inf


@pytest.mark.parametrize("method,order", param_array)
def test_do_interpolate_loss(my_data, method, order):
    """
    Checks if calculated loss is constant by the same scipy interpolation method
    and the same input dataframe
    """
    loss1 = do_interpolate(my_data, method=method, order=order)
    loss2 = do_interpolate(my_data, method=method, order=order)
    assert loss1 == loss2


def func_error(df_input):
    """
    Error prone function to test do_interpolate()
    """
    raise ValueError


def test_do_interpolate_func_failes(my_data):
    """
    If custom function is error prone, the loss value is set to infinity
    """
    loss = do_interpolate(my_data, scipy=False, func=func_error)
    assert loss == np.inf


def func(df_input):
    """
    Function to test do_interpolate() returns a random filled dataframe
    """
    np.random.seed(99)
    return pd.DataFrame(np.random.randn(100, 1))


def test_do_interpolate_func(my_data):
    """
    If custom function func() is not error prone and is not None,
    the loss value should be less then infinity
    """
    loss = do_interpolate(my_data, scipy=False, func=func)
    assert 0 < loss < np.inf


def test_do_interpolate_wrong_param(my_data):
    """
    If custom function is not a function, AssertionError is expected
    """
    with pytest.raises(AssertionError):
        do_interpolate(my_data, scipy=False, func=3)


def test_do_interpolate_wrong_method(my_data):
    """
    If scipy function raises an Exception, the loss value is set to infinity
    """
    loss = do_interpolate(my_data, method="wrong")
    assert loss == np.inf


def patched_do_interpolate(df_input, scipy=True, func=None, **kwargs):
    """
    Mock do_interpolate() for further test of interpol_method_selection(),
    where the following should be proved: best loss = min(loss).
    do_interpolate() interpolates series by application of selection mask
    and calculation of loss value. interpol_method_selection() applies
    do_interpolate() in dependence on the interpolation method.
    (e.g by trigonometric interpolation trigon_int() is executed)
    Test should check whether by given multiple interpolation methods
    with different loss values the method with the smallest loss is chosen.
    """
    _ = df_input
    loss_dict = {
        "nearest": 0.034,
        "polynomial": 0.032,
        "spline": 0.033,
        "trigonometric": 0.035,
    }
    if "method" in kwargs:
        method = kwargs["method"]
    elif func is not None and func.__name__ == "_trigon_int":
        method = "trigonometric"
    return loss_dict[method]


def test_interpol_method_selection(mocker, sel_mask_data):
    """
    Mock do_interpolate() and return loss: float from parameterization
    """
    mocker.patch(
        "kosmos_signal_processing.interpolate.interpol.do_interpolate",
        patched_do_interpolate,
    )
    best_method = interpol_method_selection(sel_mask_data)
    assert best_method == ("polynomial", 1)
