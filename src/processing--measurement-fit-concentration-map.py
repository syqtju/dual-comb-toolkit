from datetime import datetime

import matplotlib

from lib.files import get_figures_path, get_measurement_names, save_mapping_report
from lib.plots import use_latex
from lib.shortcuts import map_measurement_concentration

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
temperature = 1200  # K
"""Temperature of the gas mixture."""
length = 0.07  # m
"""Path length of the gas mixture."""

# Simulation range.

wl_min = 3426.8  # nm
"""Minimum wavelength of the simulation range."""
wl_max = 3428.1  # nm
"""Maximum wavelength of the simulation range."""

####################################################################################################
# Measurement parameters                                                                           #
####################################################################################################

# Mapping name.

mapping_name = "7a"
"""Name of the mapping to process."""
measurement_names = get_measurement_names(mapping_name)
"""List of measurement names to process."""
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

number_of_teeth = 12
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
upper_bound = 1  # VMR
"""Upper bound for the volume mixing ratio (VMR) of the molecule."""

# Noise filtering.

tooth_std_threshold = 1.5
"""Threshold for the standard deviation of the comb teeth. If the standard deviation of a tooth is 
larger than `tooth_std_threshold` times the mean standard deviation of all teeth, the tooth will be 
discarded. Note this could give unexpected results if combined with `remove_teeth_indices`."""
sub_measurements = 10
"""Number of sub-measurements to use for calculating the standard deviation of the teeth."""

# Removing noisy teeth
remove_teeth_indices = [1]
"""List of tooth indices to be removed from the fitting."""

# Output and plotting parameters.

verbose = True
"""If True, the fitting process will print detailed information."""
spectrum_plot_folder = "fit-measurement-output"
"""Folder to save the spectrum plots. If None, plots will not be saved."""

# Use LaTeX for plotting.

latex = True
"""If True, use LaTeX for plotting."""

if latex:
    use_latex()


####################################################################################################
# Mapping                                                                                          #
####################################################################################################

mapper = map_measurement_concentration(
    meas_names=measurement_names,
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
    wl_min=wl_min,
    wl_max=wl_max,
    baseline_names=baseline_names,
    fitter=fitter,
    database=database,
    initial_guess=initial_guess,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    verbose=verbose,
    spectrum_plot_folder=spectrum_plot_folder,
    tooth_std_threshold=tooth_std_threshold,
    sub_measurements=sub_measurements,
    remove_teeth_indices=remove_teeth_indices,
    flip=flip,
)


####################################################################################################
# Results plots                                                                                    #
####################################################################################################

if spectrum_plot_folder:
    matplotlib.use("Agg")

    folder_path = f"{get_figures_path()}{spectrum_plot_folder}"
    heatmap_path = f"{folder_path}/heatmap.pdf"
    plot_path = f"{folder_path}/plot.pdf"

    plt = mapper.generate_concentration_heatmap()
    plt.savefig(heatmap_path)
    plt.close()

    plt = mapper.generate_concentration_plot(x=1)
    plt.savefig(plot_path)
    plt.close()

    if verbose:
        print(f"Concentration heatmap saved to {heatmap_path}.")
        print(f"Concentration plot saved to {plot_path}.")
else:
    mapper.show_concentration_heatmap()
    mapper.show_concentration_plot(x=1)


####################################################################################################
# Mapping report                                                                                   #
####################################################################################################

report_filename = (
    f"{mapping_name.split('/')[-1]} @ {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
)

save_mapping_report(
    filename=report_filename,
    data={
        "molecule": molecule,
        "pressure": pressure,
        "temperature": temperature,
        "path_length": length,
        "optical_comb_spacing": optical_comb_spacing / 1e9,  # Convert to GHz
        "number_of_teeth": number_of_teeth,
        "laser_wavelength": laser_wavelength * 1e9,  # Convert to nm
        "wl_min": wl_min,
        "wl_max": wl_max,
        "comb_spacing": freq_spacing,
        "center_frequency": center_freq,
        "acquisition_frequency": acq_freq,
        "fitter": fitter,
        "initial_guess": initial_guess,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "sub_measurements": sub_measurements,
        "tooth_std_threshold": tooth_std_threshold,
        "results": mapper.results,
    },
    verbose=verbose,
)
