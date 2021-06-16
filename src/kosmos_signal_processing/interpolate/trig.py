"""
This module contains the functions needed for the trigonometric interpolation.
"""

from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


def e_complex_pow(power: float) -> complex:
    r"""
    Function which implements Euler's Formula for a real input number:
    :math:`e^{j \times power} = \cos(power) + j \times\sin(power)` .

    :param power: real number
    :return: complex number
    """
    return complex(np.cos(power), np.sin(power))


def get_complex_representation(points: ArrayLike) -> np.ndarray:
    r"""
    Function to calculate the Fourier coefficients of polynomial from its values,
    provided as an input parameter.

    Calculate following:
       - nth root of unity which is a complex number: :math:`w = e^{{2\pi i}/n}`
       - discrete Fourier transformation matrix:
       :math:`F = (w^{jk})_{0 \leq j \leq n-1, -(n-1)/2 \leq k \leq (n-1)/2}`

       - discrete Fourier coefficients :math:`c_k = (1/n) \times conjugate(F) \times y`

    :param points: polynomial values
    :return: complex Fourier coefficients
    """
    len_n = len(points)

    y_s = np.array(points, dtype=np.complex_)

    w_comp = e_complex_pow(2 * np.pi / len_n)
    # Fourier-Matrix
    f_matrix = np.array(
        [
            [w_comp ** (j * k) for j in range(len_n)]
            for k in range(-(len_n - 1) // 2, (len_n - 1) // 2 + 1)
        ]
    )

    ck_s = (1 / len_n) * (np.conj(f_matrix) @ y_s)  # Fourier-Coefficients
    return ck_s


def get_sin_cos_representation(points: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Function to calculate sine and cosine coefficients.

    .. math::
     a_0=2 \times c_0  \\  a_k = c_k + c_{-k}  \\  b_k = j \times (c_k - c_{-k})

    :param points: polynomial values
    :return: sine and cosine coefficients
    """
    ck_s = get_complex_representation(points)
    ck_zero_idx = (len(ck_s) - 1) // 2

    # cosine coefficients
    ak_s = [2 * ck_s[ck_zero_idx]] + [
        ck_s[ck_zero_idx + n] + ck_s[ck_zero_idx - n] for n in range(1, ck_zero_idx + 1)
    ]
    # sine coefficients
    bk_s = [complex(0)] + [
        complex(0, ck_s[ck_zero_idx + n] - ck_s[ck_zero_idx - n])
        for n in range(1, ck_zero_idx + 1)
    ]

    for n_len, _ in enumerate(ak_s):
        if ak_s[n_len].imag < 1e-3:
            ak_s[n_len] = ak_s[n_len].real

        if bk_s[n_len].imag < 1e-3:
            bk_s[n_len] = bk_s[n_len].real

    return np.array(ak_s), np.array(bk_s)


def eval_sin_cos_representation(
    t_period: float, a_coeff: np.ndarray, b_coeff: np.ndarray
) -> float:
    r"""
    Function which calculates Fourier series from given period and coefficients.

    :math:`f(x) = a_0/2 + \sum_{n=1}^N(a_n\cos(n \times x) + b_n\sin(n \times x))`

    :param t_period: parameter of function `f(x)` which represent the point in the period of time
    :param a_coeff: cosine coefficients
    :param b_coeff: sine coefficients
    :return: Fourier series
    """
    return a_coeff[0] / 2 + sum(
        a_coeff[n] * np.cos(n * t_period) + b_coeff[n] * np.sin(n * t_period)
        for n in range(1, len(a_coeff))
    )


def plot_sin_cos_representation(
    a_coeff: np.ndarray,
    b_coeff: np.ndarray,
    y_points: ArrayLike,
    start: float = -10,
    end: float = 10,
) -> None:
    """
    Function to plot trigonometric Fourier series.

    :param a_coeff: cosine coefficients
    :param b_coeff: sine coefficients
    :param y_points: polynomial values
    :param start: starting value of the sequence on x axis
    :param end: end value of the sequence on x axis
    """
    x_s = np.linspace(start, end, 5000)
    y_s = [eval_sin_cos_representation(t, a_coeff, b_coeff) for t in x_s]

    n_len = len(y_points)
    x_points = np.array([(2 * np.pi * i) / n_len for i in range(n_len)])

    plt.figure(figsize=(14, 7))
    plt.plot(x_s, y_s)
    plt.scatter(x_points, y_points, c="black")
    plt.show()
