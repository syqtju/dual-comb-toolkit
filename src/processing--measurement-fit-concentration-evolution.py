import os
from time import time

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from lib.entities import MeasuredSpectrum
from lib.files import get_animations_path, get_evolutions_path
from lib.fitting import ConcentrationFitter
from lib.measurements import Measurement
from lib.plots import use_latex

####################################################################################################
# Simulation parameters                                                                            #
####################################################################################################

# Molecule and database.

molecule = "CH4"
"""Molecule to simulate."""
database = "hitemp"
"""Database to use for the simulation. Can be 'hitran', 'hitemp', 'exomol' or 'geisa'. Some may not
be available for all molecules."""

# Physical conditions.

pressure = 101325  # Pa
"""Pressure of the gas mixture."""
temperature = 1227  # K
"""Temperature of the gas mixture."""
length = 0.07  # m
"""Path length of the gas mixture."""

# Simulation range.

wl_min = 3427.0  # nm
"""Minimum wavelength of the simulation range."""
wl_max = 3427.9  # nm
"""Maximum wavelength of the simulation range."""

####################################################################################################
# Measurement parameters                                                                           #
####################################################################################################

# Measurement name.

measurement_name = "flame_bottom_5s_phi0.7_2025-06-19-11-41-09"
"""Name of the measurement to process."""
baseline_names = []
"""List of baseline measurement names to use for processing."""

# Radio frequency comb specifications.

center_freq = 40000.0  # Hz
"""Center frequency of the radio frequency comb."""
freq_spacing = 200.0  # Hz
"""Frequency spacing of the radio frequency comb."""
acq_freq = 400000.0  # Hz
"""Acquisition frequency of the radio frequency comb."""
flip = False
"""If True, the measured transmission spectrum will be flipped with respect to the center frequency."""

# Optical comb specifications.

number_of_teeth = 13
"""Number of teeth in the optical frequency comb."""
optical_comb_spacing = 1250e6  # Hz
"""Optical frequency comb spacing."""
laser_wavelength = (wl_max + wl_min) / 2 * 1e-9  # m
"""Wavelength of the laser used to probe the gas mixture."""

####################################################################################################
# Fitting parameters                                                                               #
####################################################################################################

# Fitter, initial guess and allowed concentration bounds.

fitter = "normal_gpu"
"""Fitting algorithm to use. Can be 'normal', 'interp' and 'normal_gpu'."""
initial_guess = 0.0001  # VMR
"""Initial guess for the volume mixing ratio (VMR) of the molecule."""
lower_bound = 0  # VMR
"""Lower bound for the volume mixing ratio (VMR) of the molecule."""
upper_bound = 0.05  # VMR
"""Upper bound for the volume mixing ratio (VMR) of the molecule."""

# Removing noisy teeth

remove_teeth_indices = [1, 2, 12, 13]
"""List of tooth indices to be removed from the fitting."""

# Output and plotting parameters.

verbose = True
"""If True, the fitting process will print detailed information."""

# Use LaTeX for plotting.

latex = True
"""If True, use LaTeX for plotting."""

if latex:
    use_latex()

####################################################################################################
# Fitting                                                                                          #
####################################################################################################

integration_time = 0.00005  # s
time_between_measurements = 0.00005  # s

####################################################################################################
# Animation                                                                                        #
####################################################################################################

m = Measurement(
    measurement_name=measurement_name,
    center_freq=center_freq,
    freq_spacing=freq_spacing,
    number_of_teeth=number_of_teeth,
    laser_wavelength=laser_wavelength,
    optical_comb_spacing=optical_comb_spacing,
    acq_freq=acq_freq,
    molecule=molecule,
    pressure=pressure,
    temperature=temperature,
    length=length,
    flip=flip,
)

total_time = m.measurement_time

intervals = [
    (i * time_between_measurements, i * time_between_measurements + integration_time)
    for i in range(int(total_time / time_between_measurements) + 1)
]

print(f"Total measurement time: {total_time:.2f} s")

measured_spectra: list[MeasuredSpectrum] = []
concentrations = []
fitted_spectra = []
min_abs = 10
simulator = None

for start, end in intervals:
    # Compute the transmission spectrum for the current time interval
    t0 = time()
    m.compute_transmission(start=start, end=end, save=True)
    m.remove_teeth(remove_teeth_indices)
    measured_spectrum = m.transmission_spectrum
    print(f"Time for computing transmission: {(time() - t0)/100:.2f} ms")

    # Fit the concentration for the current time interval
    t0 = time()
    f = ConcentrationFitter(
        meas_transmission=measured_spectrum,
        wl_min=wl_min,
        wl_max=wl_max,
        initial_guess=initial_guess,
        fitter=fitter,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        verbose=verbose,
        database=database,
        simulator=simulator,
        exit_gpu=False,
    )
    concentrations.append(f.concentration)
    print(f"Time for fitting: {time() - t0:.2f} s")

    # Store the results for the current time interval
    fitted_spectra.append(f.simulated_transmission)
    measured_spectra.append(f.measured_transmission)
    min_abs = min(min_abs, measured_spectra[-1].y_nm.min())
    simulator = f.simulator

    print(f"Completed interval number {len(concentrations)} / {len(intervals)}")

if simulator and simulator.use_gpu:
    simulator.exit_gpu()

fig = plt.figure(figsize=(6, 4.5), dpi=(1920 / 6))
ax = plt.gca()  # Get current axes
sim = ax.plot(
    fitted_spectra[0].x_nm,
    fitted_spectra[0].y_nm,
    color="blue",
    label="Simulated Transmission",
)[0]
meas = ax.plot(
    measured_spectra[0].x_nm,
    measured_spectra[0].y_nm,
    "o-",
    color="red",
    label="Measured Transmission",
)[0]
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Transmission [-]")
ax.set_title(
    f"{measurement_name}\n$t = 0$ s\t Concentration = {concentrations[0]:.6f} VMR"
)
ax.set_xlim(wl_min, wl_max)
ax.set_ylim(min_abs * 0.96, 1.03)
ax.legend(loc="upper right")


def update(frame):
    # for each frame, update the data stored on each artist.
    measured_spectrum = measured_spectra[frame]
    x_meas = measured_spectrum.x_nm
    y_meas = measured_spectrum.y_nm
    x_sim = fitted_spectra[frame].x_nm
    y_sim = fitted_spectra[frame].y_nm
    meas.set_xdata(x_meas)
    meas.set_ydata(y_meas)
    sim.set_xdata(x_sim)
    sim.set_ydata(y_sim)
    ax.set_title(
        f"{measurement_name}\n$t = {frame * time_between_measurements:.2f}$ s\t Concentration = {concentrations[frame]:.6f} VMR"
    )
    return meas


output_name = f"{measurement_name.replace('/', '-')}_it{integration_time}s_tbm{time_between_measurements}s"
animation_path = os.path.join(get_animations_path(), output_name + ".gif")
evolution_path = os.path.join(get_evolutions_path(), output_name + ".csv")

ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=len(measured_spectra),
    interval=time_between_measurements * 1000,
)
ani.save(filename=animation_path, writer="pillow")
plt.show()

time_arr = [inter[0] for inter in intervals]
np.savetxt(
    fname=evolution_path,
    X=np.array([time_arr, concentrations]).T,
    delimiter=",",
    header="Time [s],Concentration [VMR]",
    comments="",
)
