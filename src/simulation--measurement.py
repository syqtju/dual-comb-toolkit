from matplotlib import pyplot as plt

from lib.plots import article_tight, use_latex
from lib.simulations import simulate_measurement

# Define the parameters for simulating the measurement

molecule = "CH4"
wl_min = 3427.0  # nm
wl_max = 3427.9  # nm

database = "hitemp"

vmr = 0.01  # volume mixing ratio
pressure = 101325  # Pa
temperature = 1200  # K
length = 0.07  # m
laser_wavelength = 3427.41  # nm

number_of_teeth = 30
optical_comb_spacing = 500e6  # Hz
modulation_intensity = 4.69  # rad

noise_distribution = "bessel"
transmission_std = 0.014  # unitless
nr_teeth_for_transmission_std = 30  # teeth
transmission_std_threshold = 0.1  # unitless
spectrum_shift_range = (-0.02, 0.02)  # nm
scaling_range = (0.2, 1.5)  # unitless
laser_wavelength_slack = (-0.05, 0.05)  # nm

latex = True

if latex:
    use_latex()

# Simulate the transmission spectrum.
x_sim, y_sim = simulate_measurement(
    molecule=molecule,
    wl_min=wl_min,
    wl_max=wl_max,
    laser_wavelength=laser_wavelength,
    optical_comb_spacing=optical_comb_spacing,
    number_of_teeth=number_of_teeth,
    vmr=vmr,
    pressure=pressure,
    temperature=temperature,
    length=length,
    database=database,
    transmission_std=transmission_std,
    nr_teeth_for_transmission_std=nr_teeth_for_transmission_std,
    transmission_std_threshold=transmission_std_threshold,
    scaling_range=scaling_range,
    spectrum_shift_range=spectrum_shift_range,
    laser_wavelength_slack=laser_wavelength_slack,
    noise_distribution=noise_distribution,
    modulation_intensity=modulation_intensity,
)


# Plot the simulated transmission spectrum.
plt.scatter(x_sim, y_sim, label="Simulated Measurement", color="blue", zorder=0)
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
plt.title(
    f"Transmission spectrum of {molecule} at {pressure:.2f} Pa "
    + f"and {temperature:.2f} K.\n{length:.3f} m path length, {vmr:.3f} VMR."
)
plt.tight_layout(**article_tight)
plt.show()
