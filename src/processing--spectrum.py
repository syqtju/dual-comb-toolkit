from matplotlib import pyplot as plt

from lib.plots import article_tight, use_latex
from lib.shortcuts import get_measurement_spectrum

# Define the molecule, VMR, pressure, temperature, and length.
measurement_name = "cell-sweep-10-34-17-03-2025/Position-X1-Y1-sample.lvm"
acq_freq = 400000.0  # Hz
latex = True # Whether to use LaTeX for plotting.

if latex:
    use_latex()

freq, spec = get_measurement_spectrum(measurement_name, acq_freq=acq_freq)

# Plot the measured FFT spectrum.
plt.plot(freq, spec, color="red")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [a.u.]")
plt.title(f'FFT spectrum of\n{measurement_name}.')
plt.tight_layout(**article_tight)
plt.show()
