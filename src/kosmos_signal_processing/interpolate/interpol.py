"""
This module provides an evaluation algorithm for interpolation methods.
This enables the developer to choose the best fitting interpolation method
for a given dataset subject to a specified array of methods.
"""

from typing import Callable, Tuple
import numpy as np
import pandas as pd

from kosmos_signal_processing.interpolate.trig import (
    get_sin_cos_representation,
    eval_sin_cos_representation,
)


def selection_mask(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Function in order to generate selection mask to prepare dataframe for interpolation evaluation.

    :param df_input: raw dataframe
    :return: dataframe prepared for evaluation
    """
    # Deleting every second existing element from df
    selection_mask_df = [bool(i % 2) for i in range(len(df_input))]
    test_df = df_input.copy()
    test_df.loc[selection_mask_df, "y"] = np.NaN
    return test_df


# Disabling broad except lint because in any case this goes wrong we'll return inf
# pylint: disable=broad-except
def do_interpolate(
    df_input: pd.DataFrame,
    scipy: bool = True,
    func: Callable[[pd.DataFrame], pd.DataFrame] = None,
    **kwargs,
) -> float:
    """
    Function for the core interpolation function. Attention: for Pandas 1d.interpolate ONLY!

    :param df_input: dataframe to perform interpolation on
    :param scipy: flag to set if a scipy interpolation function is used
    :param func: custom interpolation function to use in case scipy is not used
    :param kwargs: parameters for scipy interpolation function
    :return: loss value
    """
    # Get frames
    df_masked = selection_mask(df_input)
    # Apply selection mask
    if scipy:
        try:
            interpol_df = df_masked.interpolate(**kwargs)
            loss = np.nansum(np.square(interpol_df["y"] - df_input["y"]))
            return loss
        except Exception as exc:
            print(f"Warning: Error during interpolation:\n{exc}")
            # if interpolation can't be computed due to some reason, loss is set to infinity
            # so this method can't possibly be the best method
            return np.inf
    else:
        try:
            assert (
                func is not None
            ), "If Scipy should not be used, function func must be provided"
            assert callable(func), "Parameter func has to be a function"
            loss = np.nansum(np.square(func(df_masked) - df_input["y"]))
            return loss
        except AssertionError:
            raise
        except Exception as exc:
            print(f"Warning: Error during interpolation:\n{exc}")
            # if interpolation can't be computed due to some reason, loss is set to infinity
            # so this method can't possibly be the best method
            return np.inf


def _trigon_int(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Trigonometric Interpolation Function.

    :param df_input: dataframe to perform interpolation on
    :return: interpolated dataframe
    """
    # special case with utilization of DFT can only be applied
    #   if support positions are equidistant
    # Ensuring that points are equidistant
    if not np.isclose(df_input.index.to_series().diff().dropna().var(), 0, atol=1e-3):
        print(
            "Warning: Input is rather not equidistant. \n"
            "Equidistance is a condition for trigonometric interpolation."
        )
    # Function works only with odd number of rows
    if df_input["y"].dropna().count() % 2 == 0:
        df_input = df_input.iloc[2:]

    if np.isnan(df_input.iloc[0]["y"]):
        df_input = df_input.iloc[1:]
    if np.isnan(df_input.iloc[-1]["y"]):
        df_input = df_input.iloc[:-1]

    # Create new helper-column with #len(df equidistant points with the distance 2*Pi
    # Needed because x values are distributed between 0 and 2 Pi,
    #   as assumption that each period is of length 2*Pi
    df_input["x_loc"] = np.linspace(0, 2 * np.pi, len(df_input))

    a_coeff, b_coeff = get_sin_cos_representation(df_input.dropna()["y"])

    df_input["y"] = df_input.apply(
        lambda row: eval_sin_cos_representation(row["x_loc"], a_coeff, b_coeff)
        if np.isnan(row["y"])
        else row["y"],
        axis=1,
    )
    df_input = df_input.drop("x_loc", axis=1)
    return df_input


# Next steps could be:
# for the further modularisation - pass methods which should be evaluated as a parameter
def interpol_method_selection(df_input: pd.DataFrame) -> Tuple[str, int]:
    """
    Function to select the best performing interpolation method for the given data.

    :param df_input: dataframe to perform interpolation on
    :return: best method and best order
    """
    # Sub-List of possible interpolation methods from pd.interpolate
    # Barycentric was tossed since it is not applicable
    #   for the one-variable case due to its methodology
    interpol_list = ["nearest", "spline", "polynomial", "trigonometric"]
    # Iterate over list of interpolation methods,
    #   interpolate the missing data and compare with ground truth data
    best_loss = np.inf
    best_method = None
    best_order = None
    for method in interpol_list:
        # Both polynomial and spline require an int value for the order
        #   within the interpolate Pandas method
        if method in ["polynomial", "spline"]:
            # Order range to evaluate best order based on lowest loss.
            #
            # best_order is only updated after each iteration,
            #   if the loss is lower than the loss of previous order.
            #
            # Interval for range is based on best practices from mathematical papers
            #   and university  scripts where it is said that it is discouraged to use
            #   an order > 5, since models than tend to get too complex,
            #   swing to much between support points and thus increase the risk of overfitting.
            for order in range(1, 6):
                # The polynomial method only supports odd order degrees for now
                if method == "polynomial" and order % 2 == 0:
                    continue
                current_loss = do_interpolate(df_input, method=method, order=order)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_method = method
                    best_order = order
        elif method == "nearest":
            # Block for nearest, which doesn't require an integer value for the order
            current_loss = do_interpolate(df_input, method=method)
            if current_loss < best_loss:
                best_loss = current_loss
                best_method = method
                best_order = None
        # FYI: trigonometric interpolation is hereby a global interpolation
        #   method instead of nearest, spline or polynomial,
        #   which are local interpol. methods
        elif method == "trigonometric":
            current_loss = do_interpolate(df_input, scipy=False, func=_trigon_int)
            if current_loss < best_loss:
                best_loss = current_loss
                best_method = method
                best_order = None

    return best_method, best_order
