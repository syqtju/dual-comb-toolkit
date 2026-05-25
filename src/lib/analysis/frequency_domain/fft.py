from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import fft, fftfreq

if TYPE_CHECKING:
    from numpy import ndarray


def compute_raw_fft(x: "ndarray", y: "ndarray") -> "tuple[ndarray, ndarray]":
    """
    Compute the Fast Fourier Transform (FFT) of a real-valued
    time series. The second half of the FFT output (negative
    frequencies) is discarded since it is redundant (complex
    conjugates) for real-valued inputs.

    Parameters
    ----------
    x : ndarray
        Array of x values of the series
    y : ndarray
        Array of y values of the series

    Returns
    -------
    tuple[ndarray, ndarray]
        Tuple containing the x (real positive frequencies) and
        y (complex amplitude) values of the FFT.
    """
    N = len(x)  # Number of sample points
    T = x[1] - x[0]  # Sample spacing

    yf = fft(y)[: N // 2]
    xf = fftfreq(N, T)[: N // 2]

    return xf, yf


def compute_complex_fft(x: "ndarray", y: "ndarray") -> "tuple[ndarray, ndarray]":
    """
    Obtain the positive Fast Fourier Transform frequency and complex values of a time series.

    Parameters
    ----------
    x : ndarray
        Array of x values of the series
    y : ndarray
        Array of y values of the series

    Returns
    -------
    tuple[ndarray, ndarray]
        Tuple containing the frequency and complex values (amplitude + phase) of the FFT.
    """
    xf, yf = compute_raw_fft(x, y)

    N = len(x)
    yf = 2.0 / N * yf
    yf[0] /= 2
    if N % 2 == 0:
        yf[-1] /= 2

    return xf, yf


def compute_real_fft(x: "ndarray", y: "ndarray") -> "tuple[ndarray, ndarray]":
    """
    Obtain the positive Fast Fourier Transform frequency and amplitude of a time series.

    Parameters
    ----------
    x : ndarray
        Array of x values of the series
    y : ndarray
        Array of y values of the series

    Returns
    -------
    tuple[ndarray, ndarray]
        Tuple containing the frequency and amplitude values of the FFT.
    """
    x_data, y_data = compute_complex_fft(x, y)
    return x_data, np.abs(y_data)
