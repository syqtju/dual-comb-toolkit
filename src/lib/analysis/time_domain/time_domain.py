from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    import matplotlib.pyplot as plt
    from numpy import ndarray


class FFTCalculator:
    """Compute the Fast Fourier Transform of a signal.

    Parameters
    ----------
    t : ndarray
        Time values.
    amplitude : ndarray
        Amplitude values.
    """

    def __init__(self, t: "ndarray", amplitude: "ndarray"):
        self.t: "ndarray" = t
        self.amplitude: "ndarray" = amplitude

        self._fft_x: "Optional[ndarray]" = None
        self._fft_y: "Optional[ndarray]" = None

    @property
    def fft_x(self) -> "ndarray":
        if self._fft_x is None:
            self.calculate_fft()
        return self._fft_x

    @property
    def fft_y(self) -> "ndarray":
        if self._fft_y is None:
            self.calculate_fft()
        return self._fft_y

    def calculate_fft(self) -> "tuple[ndarray, ndarray]":
        from lib.analysis.frequency_domain.fft import compute_fft

        self._fft_x, self._fft_y = compute_fft(self.t, self.amplitude)
        return self._fft_x, self._fft_y

    # Plot

    def generate_fft_plot(self, yscale="linear") -> "plt":
        from lib.plots import spectrum_plot

        return spectrum_plot(
            self.fft_x,
            self.fft_y,
            "Spectrum of the Signal",
            "Frequency (Hz)",
            "Amplitude",
            yscale=yscale,
        )

    def show_fft_plot(self, yscale="linear") -> None:
        self.generate_fft_plot(yscale=yscale).show()


def chop_interferograms(
    t: "ndarray",
    amp: "ndarray",
    acq_freq: float,
    rf_comb_spacing: float,
) -> "tuple[ndarray, ndarray]":
    """
    Blindly slices comb data into exactly equal interferograms, based on
    the acquisition frequency and the repetition frequency difference.
    If there are any fractional interferograms at the end of the data,
    they are discarded. Note that the acquisition frequency must be an
    integer multiple of the repetition frequency difference.

    Parameters
    ----------
    t : ndarray
        Time values of the acquired signal.
    amp : ndarray
        Amplitude values of the acquired signal.
    acq_freq : float
        Acquisition frequency in Hz.
    rf_comb_spacing : float
        Comb spacing of the radio frequency comb in Hz.
    """
    nr_points_per_period = acq_freq / rf_comb_spacing
    nr_points_in_total = len(amp)

    if not nr_points_per_period % 1 == 0:
        raise ValueError(
            f"Acquisition frequency ({acq_freq} Hz) must be an integer "
            f"multiple of the comb spacing ({rf_comb_spacing} Hz)."
        )

    nr_points_per_period = int(nr_points_per_period)
    nr_interferograms = nr_points_in_total // nr_points_per_period

    # Remove any fractional interferograms at the end of the data
    clean_t = t[: nr_interferograms * nr_points_per_period]
    clean_amp = amp[: nr_interferograms * nr_points_per_period]

    mat_amp = clean_amp.reshape((nr_interferograms, nr_points_per_period))
    mat_t = clean_t.reshape((nr_interferograms, nr_points_per_period))

    return mat_t, mat_amp
