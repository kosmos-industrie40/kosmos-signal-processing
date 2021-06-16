"""
Defines fixtures for the interpol module tests.
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def my_data():
    """
    Fixture to create example dataframe
    """
    np.random.seed(9999)
    data_frame = pd.DataFrame(np.random.randn(100, 1), columns=["y"])
    return data_frame


@pytest.fixture
def sel_mask_data():
    """
    Fixture to create example dataframe selection mask
    """
    data = []
    for i in range(100):
        data.append(i)
        data.append(i)
    return pd.DataFrame(data)
