from lib.entities import Baseline

# Specify the name and specifications of the measurement.
meas_baseline = [
    "cell-sweep-10-34-17-03-2025/Position-X11-Y1",
    "cell-sweep-10-34-17-03-2025/Position-X12-Y1",
    "cell-sweep-10-34-17-03-2025/Position-X13-Y1",
    "cell-sweep-10-34-17-03-2025/Position-X14-Y1",
    "cell-sweep-10-34-17-03-2025/Position-X15-Y1",
    "cell-sweep-10-34-17-03-2025/Position-X16-Y1",
]
center_freq = 40000.0  # Hz
freq_spacing = 200.0  # Hz
number_of_teeth = 38
laser_wavelength = 3427.45e-9  # m
optical_comb_spacing = 500e6  # Hz
acq_freq = 400000.0  # Hz

# Compute the baseline.
bl = Baseline(
    measurement_names=meas_baseline,
    center_freq=center_freq,
    freq_spacing=freq_spacing,
    number_of_teeth=number_of_teeth,
    laser_wavelength=laser_wavelength,
    optical_comb_spacing=optical_comb_spacing,
    acq_freq=acq_freq,
)

bl.show_baseline_plot()

bl.show_baselines_plot()

bl.show_standard_deviation_plot()
