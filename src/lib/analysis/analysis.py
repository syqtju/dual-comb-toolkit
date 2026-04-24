from typing import TYPE_CHECKING

import allantools
import numpy as np

from lib.analysis.frequency_domain import compute_complex_fft
from lib.analysis.frequency_domain.rf_domain import RadioFrequencyCombExtractor
from lib.analysis.time_domain import chop_interferograms

if TYPE_CHECKING:
    from typing import Optional

    from numpy import ndarray

__all__ = [
    "PhaseAllanAnalyser",
    "TransmissionAnalyser",
]

# Transmission


class TransmissionAnalyser:
    """Analyse the transmission of a sample and reference time-series data.

    Parameters
    ----------
    t : ndarray
        The time array of the time series data.
    sample_amplitude : ndarray
        The amplitude of the sample time series data.
    reference_amplitude : ndarray
        The amplitude of the reference time series data.

    Keyword Arguments
    -----------------
    center_freq : float, optional
        The central frequency of the rf comb in Hz.
    freq_spacing : float, optional
        The frequency spacing of the rf comb in Hz.
    number_of_teeth : int, optional
        The number of teeth to extract.
    laser_wavelength : float, optional
        The wavelength of the laser in meters.
    optical_comb_spacing : float, optional
        Spacing of the optical optical comb in Hz.
    """

    def __init__(
        self,
        t: "ndarray",
        sample_amplitude: "ndarray",
        reference_amplitude: "ndarray",
        **kwargs,
    ) -> None:
        from lib.defaults import (
            CENTER_FREQ,
            FREQ_SPACING,
            LASER_WAVELENGTH,
            NUMBER_OF_TEETH,
            OPTICAL_COMB_SPACING,
        )

        # Time series data
        self.t: "ndarray" = t
        self.sample_amplitude: "ndarray" = sample_amplitude
        self.reference_amplitude: "ndarray" = reference_amplitude

        # Transmission analysis parameters
        self.center_freq: float = kwargs.get("center_freq", CENTER_FREQ)
        self.freq_spacing: float = kwargs.get("freq_spacing", FREQ_SPACING)
        self.number_of_teeth: int = kwargs.get("number_of_teeth", NUMBER_OF_TEETH)
        self.laser_wavelength: float = kwargs.get("laser_wavelength", LASER_WAVELENGTH)
        self.optical_comb_spacing: float = kwargs.get(
            "optical_comb_spacing", OPTICAL_COMB_SPACING
        )
        self.flip: bool = kwargs.get("flip", False)

        # Transmission data
        self._transmission_freq: "Optional[ndarray]" = None
        self._transmission_amp: "Optional[ndarray]" = None

    @property
    def kwargs(self) -> dict:
        return {
            "center_freq": self.center_freq,
            "freq_spacing": self.freq_spacing,
            "number_of_teeth": self.number_of_teeth,
            "laser_wavelength": self.laser_wavelength,
            "optical_comb_spacing": self.optical_comb_spacing,
        }

    @property
    def transmission_freq(self) -> "ndarray":
        if self._transmission_freq is None:
            self.compute_transmission()
        return self._transmission_freq

    @property
    def transmission_amp(self) -> "ndarray":
        if self._transmission_amp is None:
            self.compute_transmission()
        return self._transmission_amp

    @property
    def absorption_freq(self) -> "ndarray":
        return self.transmission_freq

    @property
    def absorption_amp(self) -> "ndarray":
        return 1 - self.transmission_amp

    def compute_transmission(self) -> "tuple[ndarray, ndarray]":
        from .frequency_domain.optical_domain import (
            OpticalFrequencyCombAnalyser,
            OpticalFrequencyCombExtractor,
        )
        from .frequency_domain.rf_domain import RadioFrequencyCombExtractor
        from .time_domain import FFTCalculator

        kwargs = self.kwargs

        fftc_sample = FFTCalculator(self.t, self.sample_amplitude)
        fftc_reference = FFTCalculator(self.t, self.reference_amplitude)

        rfce_sample = RadioFrequencyCombExtractor(
            fftc_sample.fft_x, fftc_sample.fft_y, **kwargs
        )
        rfce_reference = RadioFrequencyCombExtractor(
            fftc_reference.fft_x, fftc_reference.fft_y, **kwargs
        )

        if self.flip:
            rfce_sample.flip_comb()
            rfce_reference.flip_comb()

        ofce_sample = OpticalFrequencyCombExtractor(
            rfce_sample.comb_freq, rfce_sample.comb_amp, **kwargs
        )
        ofce_reference = OpticalFrequencyCombExtractor(
            rfce_reference.comb_freq, rfce_reference.comb_amp, **kwargs
        )

        ofca = OpticalFrequencyCombAnalyser(
            ofce_sample.comb_freq,
            ofce_sample.comb_amp,
            ofce_reference.comb_freq,
            ofce_reference.comb_amp,
        )
        self._transmission_freq = ofca.transmission_freq
        self._transmission_amp = ofca.transmission_amp

        return self._transmission_freq, self._transmission_amp


# Allan deviation

ADEV = "adev"
OAVDEV = "oadev"


def get_valid_taus(
    requested_taus: "np.ndarray",
    rate: float,
    N: int,
) -> "np.ndarray":
    """
    Filters and corrects an array of requested averaging times
    (taus) to ensure they are mathematically valid and eliminate
    step-like plotting artifacts.

    Parameters
    ----------
    requested_taus : ndarray
        The initial array of taus (e.g., from np.logspace).
    rate : float
        The sampling rate of the 1D array.
    N : int
        The total number of data points (interferograms) in
        the array.

    Returns
    -------
    valid_taus : ndarray
        A cleaned array of exact, valid taus.
    """
    # Convert requested taus to the number of discrete chunks (m)
    m_raw = requested_taus * rate

    # Round to the nearest integer to find the closest physical
    # chunk size
    m_int = np.round(m_raw).astype(int)

    # Remove duplicates to eliminate the staircase effect
    m_unique = np.unique(m_int)

    # Apply physical bounds
    # Condition A: m >= 1 (Cannot average less than 1 data point)
    # Condition B: 2*m <= N (Must have at least 2 chunks to
    # subtract from each other)
    valid_m = m_unique[(m_unique >= 1) & (2 * m_unique <= N)]

    # Convert the validated integer chunks back to exact tau times
    valid_taus = valid_m / rate

    return valid_taus


class BaseAllanAnalyser:
    """
    Base class for analysing the stability of a time-series data
    using Allan deviation.

    Note: the acquisition frequency must be an integer multiple
    of the RF comb spacing.

    Parameters
    ----------
    t : ndarray
        The time array of the time series data.
    amplitude : ndarray
        The amplitude of the time series data.
    rf_comb_spacing : float
        The spacing of the RF comb in Hz.
    rf_center_freq : float
        The center frequency of the RF comb in Hz.
    nr_teeth : int
        The number of teeth to extract for the Allan deviation
        analysis.
    acq_freq : float
        The acquisition frequency in Hz.

    Keyword Arguments
    -----------------
    nr_points : int, optional
        The number of points to use for the Allan deviation analysis.
        Default is 100.
    method : str, optional
        The method to use for the Allan deviation analysis. Can be
        "adev" or "oadev". Default is "adev".
    """

    def __init__(
        self,
        time: "ndarray",
        amplitude: "ndarray",
        rf_comb_spacing: float,
        rf_center_freq: float,
        nr_teeth: int,
        acq_freq: float,
        **kwargs,
    ) -> None:
        from lib.defaults import ALLAN_METHOD, ALLAN_NR_POINTS

        # Store parameters
        self.time = time
        self.amplitude = amplitude
        self.rf_comb_spacing = rf_comb_spacing
        self.rf_center_freq = rf_center_freq
        self.nr_teeth = nr_teeth
        self.acq_freq = acq_freq

        # Initialize interferogram data
        self.chopped_time: "Optional[ndarray]" = None
        self.chopped_amp: "Optional[ndarray]" = None
        self.chopped_phase: "Optional[ndarray]" = None
        self.teeth: "Optional[ndarray]" = None
        self.freqs: "Optional[ndarray]" = None

        # Initialize Allan deviation data for each tooth
        self.taus: "Optional[ndarray]" = None
        self.devs: "Optional[ndarray]" = None
        self.deverrs: "Optional[ndarray]" = None

        # Initialize final Allan deviation data
        self.tau: "Optional[ndarray]" = None
        self.dev: "Optional[ndarray]" = None
        self.deverr: "Optional[ndarray]" = None

        # Optional parameters
        self.nr_points: int = kwargs.get("nr_points", ALLAN_NR_POINTS)
        self.method: str = kwargs.get("method", ALLAN_METHOD)

        if self.acq_freq % self.rf_comb_spacing != 0:
            raise ValueError(
                "The acquisition frequency must be an integer "
                "multiple of the RF comb spacing."
            )

    @property
    def nr_interferograms(self) -> int:
        """Return the number of interferograms in the time series data."""
        if self.chopped_amp is not None:
            return self.chopped_amp.shape[0]

        nr_points_per_period = int(self.acq_freq / self.rf_comb_spacing)
        nr_points_in_total = len(self.amplitude)

        return nr_points_in_total // nr_points_per_period

    @property
    def measurement_duration(self) -> float:
        """Return the total duration of the measurement in seconds."""
        return self.time[-1] - self.time[0]

    @property
    def interferogram_duration(self) -> float:
        """Return the duration of a single interferogram in seconds."""
        return 1 / self.rf_comb_spacing

    def chop_interferograms(self) -> "tuple[ndarray, ndarray]":
        self.chopped_time, self.chopped_amp = chop_interferograms(
            self.time,
            self.amplitude,
            self.acq_freq,
            self.rf_comb_spacing,
        )
        return self.chopped_time, self.chopped_amp

    def extract_comb_teeth(self) -> "ndarray":
        """
        Extract the comb teeth from the time-domain signal. Subclasses
        should implement this method to extract the appropriate comb teeth
        for their analysis.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    # def compute_allan_for_tooth(self, tooth_idx: int, taus: "ndarray" = None):
    #     """Compute the Allan deviation for a specific tooth index. Subclasses
    #     should implement this method to compute the appropriate Allan deviation
    #     for their analysis.
    #     """
    #     raise NotImplementedError("Subclasses should implement this method.")

    def compute_allan_for_teeth(
        self,
        taus: "ndarray" = None,
        exclude_teeth: "list[int]" = None,
    ) -> None:
        """
        Computes the fractional frequency Allan deviation (stability of phase/timing)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_allan(
        self,
        taus: "ndarray" = None,
        exclude_teeth: "list[int]" = None,
    ) -> "tuple[ndarray, ndarray, ndarray]":
        self.compute_allan_for_teeth(taus=taus, exclude_teeth=exclude_teeth)

        devs_matrix = self.devs  # (nr_teeth, nr_taus)
        deverrs_matrix = self.deverrs  # (nr_teeth, nr_taus)

        vs = devs_matrix**2  # (nr_teeth, nr_taus)
        verrs = 2 * devs_matrix * deverrs_matrix  # (nr_teeth, nr_taus)

        avgvs = np.mean(vs, axis=0)  # (nr_taus,)
        avgverrs = np.mean(verrs, axis=0)  # (nr_taus,)

        self.tau = self.taus  # (nr_taus,)
        self.dev = np.sqrt(avgvs)  # (nr_taus,)
        self.deverr = avgverrs / (2 * self.dev)  # (nr_taus,)

        return self.tau, self.dev, self.deverr

    @property
    def default_taus(self) -> "ndarray":
        """
        Return a default array of taus to use for the Allan deviation
        analysis.
        """
        # Taus must be an integer multiple of the interferogram duration,
        # in order to avoid spectra leakage in the FFT. The minimum tau
        # is the interferogram duration, and the maximum tau is half of
        # the measurement duration, in order to have at least 3 chunks to
        # compare for the Allan deviation calculation.
        possible_taus = np.arange(
            self.interferogram_duration,
            self.measurement_duration / 3,
            self.interferogram_duration,
        )
        nr_possible_taus = len(possible_taus)

        if nr_possible_taus == 0:
            return np.array([])

        # Select a logarithmic number of points from the possible taus,
        # ensuring that the selected taus are exact multiples of the
        # interferogram duration. This is done by generating logarithmically
        # spaced indices to choose from the possible_taus array.

        # Generate logarithmically spaced floats from 10^0 = 1 to
        # 10^log10(nr_possible_taus).
        log_multipliers = np.logspace(
            start=0, stop=np.log10(nr_possible_taus), num=self.nr_points
        )

        # Round to nearest integer and remove duplicates.
        unique_multipliers = np.unique(np.round(log_multipliers).astype(int))

        # Convert the 1-based integers to 0-based array indices.
        indices = unique_multipliers - 1

        return possible_taus[indices]


# class IncoherentAllanAnalyser(BaseAllanAnalyser):
#     """
#     Analyse the stability of a sample and reference time-series
#     data using Allan deviation. The data is assumed to be incoherent,
#     meaning that the averaging happens after the amplitude of the
#     comb teeth is extracted.

#     Note: the acquisition frequency must be an integer multiple
#     of the RF comb spacing.

#     Parameters
#     ----------
#     t : ndarray
#         The time array of the time series data.
#     amplitude : ndarray
#         The amplitude of the time series data.
#     rf_comb_spacing : float
#         The spacing of the RF comb in Hz.
#     rf_center_freq : float
#         The center frequency of the RF comb in Hz.
#     nr_teeth : int
#         The number of teeth to extract for the Allan deviation
#         analysis.
#     acq_freq : float
#         The acquisition frequency in Hz.

#     Keyword Arguments
#     -----------------
#     nr_points : int, optional
#         The number of points to use for the Allan deviation analysis.
#         Default is 100.
#     method : str, optional
#         The method to use for the Allan deviation analysis. Can be
#         "adev" or "oadev". Default is "adev".
#     """

#     def extract_comb_teeth(self) -> "ndarray":
#         """
#         Calculate the amplitude of a comb tooth for the given time-domain signal.
#         If `tooth_idx` is None, the central tooth will be used. Otherwise, the
#         tooth at index `tooth_idx` will be used.
#         """
#         if self.chopped_time is None or self.chopped_amp is None:
#             self.chop_interferograms()

#         comb_amps = []

#         for t, amp in zip(self.chopped_time, self.chopped_amp):
#             fftc = FFTCalculator(t - t[0], amp)
#             rfce = RadioFrequencyCombExtractor(
#                 fftc.fft_x,
#                 fftc.fft_y,
#                 center_freq=self.rf_center_freq,
#                 freq_spacing=self.rf_comb_spacing,
#                 number_of_teeth=self.nr_teeth,
#             )
#             comb_amps.append(rfce.comb_amp)

#         self.teeth = np.asarray(comb_amps)  # (nr_interferograms, nr_teeth)

#         return self.teeth

#     def compute_allan_for_tooth(self, tooth_idx: int, taus: "ndarray" = None):
#         """Compute the Allan deviation for a specific tooth index."""
#         if self.teeth is None:
#             self.extract_comb_teeth()

#         import allantools

#         if taus is None:
#             taus = np.logspace(
#                 np.log10(1 / self.rf_comb_spacing),
#                 np.log10(self.measurement_duration / 3),
#                 self.nr_points,
#             )

#         tooth_amps = self.teeth[:, tooth_idx]

#         method = allantools.adev if self.method == ADEV else allantools.oadev

#         taus, dev, deverr, _n = method(
#             tooth_amps,
#             rate=self.rf_comb_spacing,
#             data_type="freq",
#             taus=taus,
#         )

#         self.taus[tooth_idx] = taus
#         self.devs[tooth_idx] = dev
#         self.deverrs[tooth_idx] = deverr

#         return taus, dev, deverr


# class CoherentAllanAnalyser(BaseAllanAnalyser):
#     """
#     Analyse the stability of a sample and reference time-series
#     data using Allan deviation. The data is assumed to be coherent,
#     meaning that the averaging happens before the amplitude of the
#     comb teeth is extracted.

#     Note: the acquisition frequency must be an integer multiple
#     of the RF comb spacing.

#     Parameters
#     ----------
#     t : ndarray
#         The time array of the time series data.
#     amplitude : ndarray
#         The amplitude of the time series data.
#     rf_comb_spacing : float
#         The spacing of the RF comb in Hz.
#     rf_center_freq : float
#         The center frequency of the RF comb in Hz.
#     nr_teeth : int
#         The number of teeth to extract for the Allan deviation
#         analysis.
#     acq_freq : float
#         The acquisition frequency in Hz.

#     Keyword Arguments
#     -----------------
#     nr_points : int, optional
#         The number of points to use for the Allan deviation analysis.
#         Default is 100.
#     method : str, optional
#         The method to use for the Allan deviation analysis. Can be
#         "adev" or "oadev". Default is "adev".
#     """

#     def extract_comb_teeth(self) -> "np.ndarray":
#         """
#         Extract the complex amplitude of the comb teeth.
#         """
#         if self.chopped_time is None or self.chopped_amp is None:
#             self.chop_interferograms()

#         comb_amps = []

#         for t, amp in zip(self.chopped_time, self.chopped_amp):
#             fft_x, fft_y = compute_complex_fft(t - t[0], amp)
#             rfce = RadioFrequencyCombExtractor(
#                 fft_x,
#                 fft_y,
#                 center_freq=self.rf_center_freq,
#                 freq_spacing=self.rf_comb_spacing,
#                 number_of_teeth=self.nr_teeth,
#             )

#             comb_amps.append(rfce.comb_amp)

#         self.teeth = np.asarray(comb_amps)  # (nr_interferograms, nr_teeth)

#         return self.teeth

#     def _coherent_adev(
#         self,
#         complex_data: "np.ndarray",
#         rate: float,
#         taus: "np.ndarray",
#     ):
#         """Calculates standard Coherent Allan Deviation (non-overlapping)."""
#         adev, adeverr = [], []
#         nr_interferograms = len(complex_data)
#         taus = get_valid_taus(taus, rate, nr_interferograms)

#         for tau in taus:
#             nr_interferograms_in_tau = int(np.round(tau * rate))
#             if (
#                 nr_interferograms_in_tau == 0
#                 or nr_interferograms < 2 * nr_interferograms_in_tau
#             ):
#                 adev.append(np.nan)
#                 adeverr.append(np.nan)
#                 continue

#             nr_chunks = nr_interferograms // nr_interferograms_in_tau

#             complex_data_cropped = complex_data[: nr_chunks * nr_interferograms_in_tau]

#             chunked_matrix = complex_data_cropped.reshape(
#                 (nr_chunks, nr_interferograms_in_tau)
#             )

#             amps = np.abs(np.mean(chunked_matrix, axis=1))  # (nr_chunks,)

#             diffs = np.diff(amps)  # Adjacent differences, (nr_chunks - 1,)

#             variance = 0.5 * np.mean(diffs**2)  # Allan variance

#             adev.append(np.sqrt(variance))
#             adeverr.append(np.sqrt(variance) / np.sqrt(nr_chunks - 1))

#         return taus, np.asarray(adev), np.asarray(adeverr)

#     def _coherent_oadev(
#         self,
#         complex_data: "np.ndarray",
#         rate: float,
#         taus: "np.ndarray",
#     ):
#         """Calculates Coherent Overlapping Allan Deviation (sliding window)."""
#         adev, adeverr = [], []
#         N = len(complex_data)

#         taus = get_valid_taus(taus, rate, N)

#         # 1. Pre-compute the cumulative sum of the complex array.
#         # We pad with a leading zero so the math aligns perfectly for index subtraction.
#         S = np.zeros(N + 1, dtype=complex)
#         S[1:] = np.cumsum(complex_data)

#         for tau in taus:
#             m = int(np.round(tau * rate))

#             # Ensure the window size is valid and we have at least 2 adjacent chunks
#             if m == 0 or N < 2 * m:
#                 adev.append(np.nan)
#                 adeverr.append(np.nan)
#                 continue

#             # 2. Vectorized Coherent Averaging
#             # S[m:] represents the ends of all possible windows of length m
#             # S[:-m] represents the starts of all possible windows of length m
#             # Subtracting them gives the sum of every sliding window instantly.
#             window_sums = S[m:] - S[:-m]

#             # Divide by m to get the mean, then take the absolute value
#             # This array contains the coherently averaged amplitude for EVERY sliding window
#             amps = np.abs(window_sums / m)

#             # 3. Calculate adjacent differences
#             # Amps[:-m] are the amplitudes of "Chunk 1"
#             # Amps[m:] are the amplitudes of the strictly adjacent "Chunk 2"
#             diffs = amps[m:] - amps[:-m]

#             # 4. Allan Variance Math
#             # Square the differences, average them, multiply by 0.5
#             variance = 0.5 * np.mean(diffs**2)

#             # 5. Calculate error bars
#             # The number of overlapping pairs we just evaluated
#             num_pairs = N - 2 * m + 1

#             adev.append(np.sqrt(variance))
#             adeverr.append(np.sqrt(variance) / np.sqrt(num_pairs))

#         return taus, np.array(adev), np.array(adeverr)

#     def compute_allan_for_tooth(self, tooth_idx: int, taus: "np.ndarray" = None):
#         """Compute the coherent Allan deviation for a specific tooth index."""
#         if self.teeth is None:
#             self.extract_comb_teeth()

#         if taus is None:
#             taus = np.logspace(
#                 np.log10(1 / self.rf_comb_spacing),
#                 np.log10(self.measurement_duration / 3),
#                 self.nr_points,
#             )

#         tooth_amps_complex = self.teeth[:, tooth_idx]

#         if self.method == ADEV:
#             taus, dev, deverr = self._coherent_adev(
#                 tooth_amps_complex, self.rf_comb_spacing, taus
#             )
#         else:
#             taus, dev, deverr = self._coherent_oadev(
#                 tooth_amps_complex, self.rf_comb_spacing, taus
#             )

#         self.taus[tooth_idx] = taus
#         self.devs[tooth_idx] = dev
#         self.deverrs[tooth_idx] = deverr

#         return taus, dev, deverr


# class AllanAnalyser:
#     def __init__(
#         self,
#         time: "ndarray",
#         amplitude: "ndarray",
#         rf_comb_spacing: float,
#         rf_center_freq: float,
#         nr_teeth: int,
#         acq_freq: float,
#         **kwargs,
#     ) -> None:
#         from lib.defaults import ALLAN_METHOD, ALLAN_NR_POINTS

#         # Store parameters
#         self.time = time
#         self.amplitude = amplitude
#         self.rf_comb_spacing = rf_comb_spacing
#         self.rf_center_freq = rf_center_freq
#         self.nr_teeth = nr_teeth
#         self.acq_freq = acq_freq
#         self.kwargs = kwargs

#         # Initialize intermediate data
#         self.taus: "dict[int, ndarray]" = {}
#         self.adevs: "dict[int, ndarray]" = {}
#         self.adeverrs: "dict[int, ndarray]" = {}
#         self.adev: "Optional[ndarray]" = None
#         self.adeverr: "Optional[ndarray]" = None

#         # Optional parameters
#         self.nr_points: int = kwargs.get("nr_points", ALLAN_NR_POINTS)
#         self.method: str = kwargs.get("method", ALLAN_METHOD)

#         if self.acq_freq % self.rf_comb_spacing != 0:
#             raise ValueError(
#                 "The acquisition frequency must be an integer "
#                 "multiple of the RF comb spacing."
#             )

#     @property
#     def measurement_duration(self) -> float:
#         """Return the total duration of the measurement in seconds."""
#         return self.time[-1] - self.time[0]

#     @property
#     def interferogram_duration(self) -> float:
#         """Return the duration of a single interferogram in seconds."""
#         return 1 / self.rf_comb_spacing

#     @property
#     def default_taus(self) -> "ndarray":
#         """
#         Return a default array of taus to use for the Allan deviation
#         analysis.
#         """
#         # Taus must be an integer multiple of the interferogram duration,
#         # in order to avoid spectra leakage in the FFT. The minimum tau
#         # is the interferogram duration, and the maximum tau is half of
#         # the measurement duration, in order to have at least 3 chunks to
#         # compare for the Allan deviation calculation.
#         possible_taus = np.arange(
#             self.interferogram_duration,
#             self.measurement_duration / 3,
#             self.interferogram_duration,
#         )
#         nr_possible_taus = len(possible_taus)

#         if nr_possible_taus == 0:
#             return np.array([])

#         # Select a logarithmic number of points from the possible taus,
#         # ensuring that the selected taus are exact multiples of the
#         # interferogram duration. This is done by generating logarithmically
#         # spaced indices to choose from the possible_taus array.

#         # Generate logarithmically spaced floats from 10^0 = 1 to
#         # 10^log10(nr_possible_taus).
#         log_multipliers = np.logspace(
#             start=0, stop=np.log10(nr_possible_taus), num=self.nr_points
#         )

#         # Round to nearest integer and remove duplicates.
#         unique_multipliers = np.unique(np.round(log_multipliers).astype(int))

#         # Convert the 1-based integers to 0-based array indices.
#         indices = unique_multipliers - 1

#         return possible_taus[indices]

#     def get_comb_teeth(self, time, amplitude) -> "ndarray":
#         fftc = FFTCalculator(time - time[0], amplitude)
#         rfce = RadioFrequencyCombExtractor(
#             fftc.fft_x,
#             fftc.fft_y,
#             center_freq=self.rf_center_freq,
#             freq_spacing=self.rf_comb_spacing,
#             number_of_teeth=self.nr_teeth,
#         )
#         return rfce.comb_amp

#     def compute_adev_for_teeth(self, taus: "ndarray" = None):
#         """Compute the Allan deviation for each comb tooth."""
#         if taus is None:
#             taus = self.default_taus

#         nr_taus = len(taus)
#         nr_points_in_measurement = len(self.time)

#         adevs, adeverrs = [], []

#         for i, tau in enumerate(taus):
#             print(f"Processing tau {tau:.6f} s ({i + 1}/{nr_taus}).")

#             if not is_multiple(
#                 tau,
#                 self.interferogram_duration,
#                 atol=1e-6 * self.interferogram_duration,
#             ):
#                 print(
#                     tau, self.interferogram_duration, tau % self.interferogram_duration
#                 )
#                 raise Warning(
#                     f"Tau={tau} s is not an integer multiple of the "
#                     f"interferogram duration ({self.interferogram_duration} s). "
#                     "This may lead to spectral leakage and inaccurate Allan "
#                     "deviation results."
#                 )

#             nr_points_per_chunk = int(tau * self.acq_freq)
#             nr_chunks = int(nr_points_in_measurement / nr_points_per_chunk)

#             comb_teeth_for_chunks = []

#             for i in range(nr_chunks):
#                 start_index = int(i * nr_points_per_chunk)
#                 end_index = int((i + 1) * nr_points_per_chunk)

#                 if end_index > nr_points_in_measurement:
#                     break

#                 submeas_amplitude = self.amplitude[start_index:end_index]
#                 submeas_time = self.time[start_index:end_index]

#                 comb_teeth = self.get_comb_teeth(submeas_time, submeas_amplitude)

#                 comb_teeth_for_chunks.append(comb_teeth)

#             comb_teeth_for_chunks = np.asarray(
#                 comb_teeth_for_chunks
#             ).T  # (nr_teeth, nr_chunks)

#             # Compute the Allan deviation for each tooth
#             diffs = np.diff(comb_teeth_for_chunks, axis=1)  # (nr_teeth, nr_chunks - 1)
#             variance = 0.5 * np.mean(diffs**2, axis=1)  # (nr_teeth,)
#             adev = np.sqrt(variance)  # (nr_teeth,)
#             adeverr = np.sqrt(variance) / np.sqrt(nr_chunks - 1)  # (nr_teeth,)

#             adevs.append(adev)
#             adeverrs.append(adeverr)

#         self.adevs = np.asarray(adevs).T  # (nr_teeth, nr_taus)
#         self.adeverrs = np.asarray(adeverrs).T  # (nr_teeth, nr_taus)
#         self.taus = taus

#         return self.adevs, self.adeverrs

#     def compute_adev(
#         self,
#         taus: "ndarray" = None,
#         exclude_teeth: "list[int]" = None,
#     ):
#         self.compute_adev_for_teeth(taus=taus)

#         vs = self.adevs**2  # (nr_teeth, nr_taus)
#         verrs = 2 * self.adevs * self.adeverrs  # (nr_teeth, nr_taus)

#         if exclude_teeth is not None:
#             mask = np.ones(self.adevs.shape[0], dtype=bool)
#             mask[exclude_teeth] = False
#             vs = vs[mask, :]  # (nr_teeth - nr_excluded_teeth, nr_taus)
#             verrs = verrs[mask, :]  # (nr_teeth - nr_excluded_teeth, nr_taus)

#         avgvs = np.mean(vs, axis=0)  # (nr_taus,)
#         avgverrs = np.mean(verrs, axis=0)  # (nr_taus,)

#         self.adev = np.sqrt(avgvs)  # (nr_taus,)
#         self.adeverr = avgverrs / (2 * self.adev)  # (nr_taus,)

#         return self.taus, self.adev, self.adeverr


class PhaseAllanAnalyser(BaseAllanAnalyser):
    """
    Analyse the phase stability of a sample and reference
    time-series data using Allan deviation.

    Note: the acquisition frequency must be an integer multiple
    of the RF comb spacing.

    Parameters
    ----------
    t : ndarray
        The time array of the time series data.
    amplitude : ndarray
        The amplitude of the time series data.
    rf_comb_spacing : float
        The spacing of the RF comb in Hz.
    rf_center_freq : float
        The center frequency of the RF comb in Hz.
    nr_teeth : int
        The number of teeth to extract for the Allan deviation
        analysis.
    acq_freq : float
        The acquisition frequency in Hz.

    Keyword Arguments
    -----------------
    nr_points : int, optional
        The number of points to use for the Allan deviation analysis.
        Default is 100.
    method : str, optional
        The method to use for the Allan deviation analysis. Can be
        "adev" or "oadev". Default is "adev".
    """

    def extract_comb_teeth(self) -> "tuple[ndarray, ndarray]":
        """
        Extract the complex amplitude of the comb teeth.
        """
        if self.chopped_time is None or self.chopped_amp is None:
            self.chop_interferograms()

        comb_freqs = []
        comb_teeth = []

        for t, amp in zip(self.chopped_time, self.chopped_amp):
            fft_x, fft_y = compute_complex_fft(t - t[0], amp)
            rfce = RadioFrequencyCombExtractor(
                fft_x,
                fft_y,
                center_freq=self.rf_center_freq,
                freq_spacing=self.rf_comb_spacing,
                number_of_teeth=self.nr_teeth,
            )

            comb_freqs.append(rfce.comb_freq)
            comb_teeth.append(rfce.comb_amp)

        self.freqs = np.asarray(comb_freqs)  # (nr_interferograms, nr_teeth)
        self.teeth = np.asarray(comb_teeth)  # (nr_interferograms, nr_teeth)

        return self.freqs, self.teeth

    def compute_allan_for_teeth(
        self,
        taus: "np.ndarray" = None,
        exclude_teeth: "list[int]" = None,
    ) -> None:
        """
        Computes the fractional frequency Allan deviation (stability of phase/timing)
        for a specific comb tooth to evaluate coherence.
        """
        if self.freqs is None or self.teeth is None:
            self.extract_comb_teeth()

        if taus is None:
            taus = self.default_taus

        # Obtain the complex amplitude of the comb teeth across all interferograms.
        teeth_idcs = [
            i
            for i in range(self.nr_teeth)
            if exclude_teeth is None or i not in exclude_teeth
        ]
        self.teeth_idcs = teeth_idcs
        teeth = self.teeth[:, teeth_idcs]  # (nr_interferograms, nr_teeth)
        freqs = self.freqs[:, teeth_idcs]  # (nr_interferograms, nr_teeth)

        # Obtain the phase of the complex comb tooth amplitudes.
        raw_phase = np.angle(teeth)  # (nr_interferograms, nr_teeth)
        unwrapped_phase = np.unwrap(raw_phase, axis=0)  # (nr_interferograms, nr_teeth)

        # Take the derivative with respect to time to get Frequency (Hz).
        dt = self.interferogram_duration
        dphi = np.diff(unwrapped_phase, axis=0)  # (nr_interferograms - 1, nr_teeth)
        df = dphi / (2 * np.pi * dt)

        # Convert to Fractional Frequency.
        fractional_freq = df / freqs[:-1, :]  # (nr_interferograms - 1, nr_teeth)

        # Compute Allan Deviation on the frequency array.
        devs, deverrs = [], []
        compute = allantools.adev if self.method == "adev" else allantools.oadev

        for tooth_idx in range(len(teeth_idcs)):
            taus_out, adev, adeverr, _ = compute(
                fractional_freq[:, tooth_idx],
                rate=self.rf_comb_spacing,
                data_type="freq",
                taus=taus,
            )
            devs.append(adev)
            deverrs.append(adeverr)

        self.devs = np.asarray(devs)  # (nr_teeth, nr_taus)
        self.deverrs = np.asarray(deverrs)  # (nr_teeth, nr_taus)
        self.taus = taus_out

        return taus_out, np.array(devs), np.array(deverrs)
