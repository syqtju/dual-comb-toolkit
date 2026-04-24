import matplotlib.pyplot as plt
import numpy as np
from numpy import inf
from scipy import signal

from lib.analysis.analysis import (
    PhaseAllanAnalyser,
)
from lib.plots import use_latex
from lib.shortcuts import get_measurement

plot_interferograms = False
plot_combs = False
plot_adev = True

####################################################################################################
# Measurement parameters                                                                           #
####################################################################################################

# Measurement name.

baseline_names = []
"""List of baseline measurement names to use for processing."""

# Radio frequency comb specifications.

rf_center_freq = 40000.0  # Hz
"""Center frequency of the radio frequency comb."""
rf_comb_spacing = 200.0  # Hz
"""Frequency spacing of the radio frequency comb."""
acq_freq = 400000.0  # Hz
"""Acquisition frequency of the radio frequency comb."""
flip = False
"""If True, the measured transmission spectrum will be flipped with respect to the center frequency."""

# Optical comb specifications.

nr_teeth = 13
"""Number of teeth in the optical frequency comb."""
optical_comb_spacing = 1250e6  # Hz
"""Optical frequency comb spacing."""
laser_wavelength = 3427.41e-9 + 0.03e-9  # m
"""Wavelength of the laser used to probe the gas mixture."""


####################################################################################################
# Processing parameters                                                                            #
####################################################################################################

# Noise filtering.

tooth_std_threshold = inf
"""Threshold for the standard deviation of the comb teeth. If the standard deviation of a tooth is 
larger than `tooth_std_threshold` times the mean standard deviation of all teeth, the tooth will be 
discarded. Note this could give unexpected results if combined with `remove_teeth_indices`. If using
`remove_teeth_indices`, set this to `inf` to avoid conflicts. This may also conflict with the use
of `baseline_names`, since filtering based on standard deviation will not be applied to the baseline
teeth."""
sub_measurements = None
"""Number of sub-measurements to use for calculating the standard deviation of the teeth."""

# Removing noisy teeth
exclude_teeth = []
"""List of tooth indices to be removed from the fitting."""

# Use LaTeX for plotting.

latex = True
"""If True, use LaTeX for plotting."""

if latex:
    use_latex()

####################################################################################################
#  Functions                                                                                       #
####################################################################################################


def apply_highpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Applies a zero-phase Butterworth high-pass filter to the data.
    """
    # The Nyquist frequency is half the sampling rate
    nyquist = 0.5 * sample_rate

    # Scipy requires the cutoff frequency to be normalized between 0 and 1
    # where 1 represents the Nyquist frequency
    normalized_cutoff = cutoff_freq / nyquist

    # Design the Butterworth filter
    # b and a are the numerator and denominator polynomials of the filter
    b, a = signal.butter(order, normalized_cutoff, btype="high", analog=False)

    # Apply the filter using filtfilt (forward-backward filtering)
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def apply_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Applies a zero-phase Butterworth low-pass filter to the data.
    """
    # The Nyquist frequency is half the sampling rate
    nyquist = 0.5 * sample_rate

    # Scipy requires the cutoff frequency to be normalized between 0 and 1
    # where 1 represents the Nyquist frequency
    normalized_cutoff = cutoff_freq / nyquist

    # Design the Butterworth filter
    # b and a are the numerator and denominator polynomials of the filter
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

    # Apply the filter using filtfilt (forward-backward filtering)
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def join_measurements(measurement_names):
    t = None
    reference_amp = None
    sample_amp = None
    n = 0

    for meas_name in measurement_names:
        measurement = get_measurement(
            meas_name=meas_name,
            center_freq=rf_center_freq,
            freq_spacing=rf_comb_spacing,
            number_of_teeth=nr_teeth,
            laser_wavelength=laser_wavelength,
            optical_comb_spacing=optical_comb_spacing,
            acq_freq=acq_freq,
            baseline_names=baseline_names,
            sub_measurements=sub_measurements,
            tooth_std_threshold=tooth_std_threshold,
            flip=flip,
        )

        if t is None:
            t = measurement.t
            reference_amp = measurement.reference_signal
            sample_amp = measurement.sample_signal
        else:
            t = np.concatenate((t, measurement.t + n * measurement.measurement_time))
            reference_amp = np.concatenate(
                (reference_amp, measurement.reference_signal)
            )
            sample_amp = np.concatenate((sample_amp, measurement.sample_signal))

        n += 1

    return t, reference_amp, sample_amp


####################################################################################################
# Measurement                                                                                      #
####################################################################################################

# Get the measured transmission spectrum.

time, reference_amp, sample_amp = join_measurements(
    [
        "flame_bottom_5s_phi0.7_2025-06-19-11-41-09",
    ]
)

amp = sample_amp

# ---------------------------------------------------------
# Filter low frequencies

amp = apply_highpass_filter(amp, cutoff_freq=15000, sample_rate=acq_freq, order=4)
amp = apply_lowpass_filter(amp, cutoff_freq=65000, sample_rate=acq_freq, order=4)

aa = PhaseAllanAnalyser(
    time=time,
    amplitude=amp,
    rf_comb_spacing=rf_comb_spacing,
    rf_center_freq=rf_center_freq,
    nr_teeth=nr_teeth,
    acq_freq=acq_freq,
    method="adev",
    nr_points=20,
)
aa.compute_allan(exclude_teeth=exclude_teeth)

####################################################################################################
# Plot interferograms                                                                              #
####################################################################################################

if plot_interferograms:
    for i in range(aa.nr_interferograms):
        plt.plot(aa.chopped_time[i], aa.chopped_amp[i], label=f"Interferogram {i}")

    plt.xlabel("Time (s)")
    plt.ylabel("Measured Reference Signal")
    plt.legend()
    plt.show()

####################################################################################################
# Plot combs                                                                                       #
####################################################################################################

if plot_combs:
    for i in range(aa.nr_interferograms):
        plt.plot(aa.teeth[i], label=f"Interferogram {i}")
    plt.xlabel("Tooth Index")
    plt.ylabel("Comb Tooth Amplitude")
    plt.legend()
    plt.show()

####################################################################################################
# Plot Allan deviation of each tooth                                                               #
####################################################################################################

if plot_adev:
    plt.errorbar(
        aa.tau[::2],
        aa.dev[::2],
        yerr=aa.deverr[::2],
        fmt="o-",
        color="black",
        ecolor="black",
        linewidth=2,
        capsize=3,
        label="Combined",
        zorder=1000,
    )
    plt.xlabel("Averaging Time (s)")
    plt.ylabel("Allan Deviation")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.grid(True, which="both", ls="-", color="0.85")
    plt.show()

np.savetxt(
    "allan_deviation.csv",
    np.column_stack((aa.tau, aa.dev, aa.deverr)),
    delimiter=",",
    header="tau,adev,adev_err",
    comments="",
)
