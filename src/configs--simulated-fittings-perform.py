import argparse
from time import perf_counter, strftime

import matplotlib.pyplot as plt
from numpy import array

from lib.benchmarking import Time
from lib.constants import c
from lib.conversions import delta_frequency_to_delta_wavelength
from lib.files import (
    append_to_csv_report,
    initialize_csv_report,
    initialize_figures_folder,
    read_configurations,
)
from lib.plots import article_tight
from lib.plots import use_latex as latex
from lib.shortcuts import fit_simulated_measurement_concentration

####################################################################################################
#  Simulation parameters                                                                           #
####################################################################################################

# Molecule and database

molecule = "CH4"
"""Molecule to simulate."""
database = "hitemp"
"""Database to use for the simulation. Can be 'hitran', 'hitemp', 'exomol' or 'geisa'. Some may not
be available for all molecules."""

# Physical conditions

vmr = 0.01  # volume mixing ratio
"""Volume mixing ratio of the molecule in the gas mixture."""
pressure = 101325  # Pa
"""Pressure of the gas mixture."""
temperature = 1200  # K
"""Temperature of the gas mixture."""
length = 0.07  # m
"""Path length of the gas mixture."""
laser_wavelength = 3427.43  # nm
"""Wavelength of the laser used to probe the gas mixture."""

# Simulation range

wl_min = 3427.0  # nm
"""Minimum wavelength of the simulation range."""
wl_max = 3427.9  # nm
"""Maximum wavelength of the simulation range."""
min_comb_span = 0.15  # nm
"""Minimum comb span to consider a configuration valid. If the comb span is smaller than this value,
it will be skipped."""

# Noise and error parameters

noise_distribution = "bessel"
"""Distribution of the noise to simulate. Can be 'bessel' or 'uniform'."""
transmission_std = 0.014  # unitless
"""Standard deviation of the transmission spectrum for a given number of teeth (max transmission 
is 1). Used to calculate the standard deviation of the noise added to the transmission spectrum."""
nr_teeth_for_transmission_std = 30  # teeth
"""Used to scale the `transmission_std` value for different numbers of teeth."""
tooth_std_threshold_start = 2.5  # unitless
teeth_start = 5
tooth_std_threshold_end = 1.5  # unitless
teeth_end = 30
"""Threshold for the standard deviation of the comb teeth. If the standard deviation of a tooth is 
larger than `tooth_std_threshold` times the mean standard deviation of all teeth, the tooth will be 
discarded. Here, we specify a start and end value to linearly decrease the threshold as the number 
of teeth increases."""
spectrum_shift_range = (-0.02, 0.02)  # nm
"""Range of the wavelength shift to apply to the final transmission spectrum."""
scaling_range = (0.2, 1.5)  # unitless
"""Range of the scaling factor to apply to the final transmission spectrum."""
modulation_intensities = {
    5: 0.93,
    6: 1.81,
    7: 1.81,
    8: 2.75,
    9: 2.75,
    10: 3.71,
    11: 3.71,
    12: 4.69,
    13: 4.69,
    14: 5.67,
    15: 5.67,
    16: 6.67,
    17: 6.67,
    18: 7.66,
    19: 7.66,
    20: 8.66,
    21: 8.66,
    22: 9.67,
    23: 9.67,
    24: 10.67,
    25: 10.67,
    26: 11.68,
    27: 11.68,
    28: 12.69,
    29: 12.69,
    30: 13.7,
}
"""Modulation intensity used to compute the bessel noise distribution for the different numbers
of teeth."""

# Plots

generate_plots = True
"""If True, plots of the simulated and fitted spectra will be generated."""
use_latex = False
"""If True, LaTeX will be used for plotting. Requires LaTeX to be installed on the system."""

# Reports

detailed_report = True
"""If True, a detailed report containing the fitted concentrations of each configuration will be 
generated in CSV files."""

####################################################################################################
#  Fitting parameters                                                                              #
####################################################################################################

normalize = True
"""If True, the measured transmission spectrum will be normalized to the maximum value before fitting."""
initial_guess = 0.001
"""Initial guess for the concentration to fit in VMR (volume mixing ratio)."""
lower_bound = 0.0  # VMR
"""Lower bound for the concentration to fit in VMR (volume mixing ratio)."""
upper_bound = 0.05  # VMR
"""Upper bound for the concentration to fit in VMR (volume mixing ratio)."""
fitter = "normal_gpu"
"""Fitter to use for the fitting process. Can be 'normal', 'normal_gpu' or 'interp'."""

####################################################################################################
#  Simulation cases                                                                                #
####################################################################################################

nr_simulations_per_config = 100
"""Number of simulations to perform for each configuration of number of teeth and comb spacing."""
comb_spacings = [(i + 1) * 100e6 for i in range(30)] * 26  # Hz
"""Comb spacings to simulate."""
numbers_of_teeth = [i for i in range(5, 31) for _ in range(30)]  # teeth
"""Number of teeth to simulate."""

####################################################################################################
#  Command line arguments                                                                          #
####################################################################################################

# If the command line argument `-c` or `--config` is provided, read the configurations from the
# specified file

parser = argparse.ArgumentParser(
    description="Simulate and fit a measurement of a molecule's concentration using an optical comb."
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="",
    help="Path to the file with configurations to simulate.",
)
parser.add_argument(
    "-r", "--report", type=str, default="", help="Name of the output report."
)
args = parser.parse_args()

if getattr(args, "config"):
    comb_spacings, numbers_of_teeth = read_configurations(args.config)

report_name = None
if getattr(args, "report"):
    report_name = args.report


####################################################################################################
# Info and console output                                                                          #
####################################################################################################

total_nr_configs = min(len(numbers_of_teeth), len(comb_spacings))

print(f"Simulating {total_nr_configs} configurations.\n")


def indentation(i: int) -> str:
    return " " * len(f"{i + 1}. ")


def print_config_params(nr_teeth: int, spacing: float) -> None:
    print(f"Number of teeth: {nr_teeth}, Comb spacing: {spacing / 1e9:.2f} GHz.")


def print_fitting_result(i: int, vmr: float) -> None:
    print(f"{indentation(i)}The fitted concentration is {vmr:.6f} VMR.")


def print_config_result(nr_teeth: int, spacing: float, mean: float, std: float) -> None:
    print(
        f"Number of teeth: {nr_teeth}, Comb spacing: {spacing / 1e9:.2f} GHz, "
        + f"Mean concentration: {mean:.6f} VMR, Standard deviation: {std:.6f} VMR"
    )


def print_skipped_config(nr_config: int, nr_teeth: int, spacing: float) -> None:
    spacing_ghz = spacing / 1e9
    print(
        f"Skipping configuration {nr_config} ({nr_teeth} teeth, "
        + f"{spacing_ghz:.2f} GHz) due to comb span being too large or too small.",
        end="\n\n",
    )


def print_progress(
    nr_config: int, nr_configs_performed: int, time_start: float
) -> None:
    progress = int(nr_config / total_nr_configs * 100)  # %

    time_now = perf_counter()
    time_left = (
        (time_now - time_start)
        / nr_configs_performed
        * (total_nr_configs - nr_config)
        / 60
    )  # min

    simulations_finished = nr_config * nr_simulations_per_config
    simulations_total = total_nr_configs * nr_simulations_per_config

    print(
        f"Progress: [{'#' * progress}{'.' * (100 - progress)}] {simulations_finished} "
        f"of {simulations_total} fittings performed (estimated time left: {time_left:.2f} min).",
        end="\n\n",
    )


####################################################################################################
# Prepare output file                                                                              #
####################################################################################################

timestr = strftime("%Y%m%d-%H%M%S")
csv_filename = report_name or f"report-{timestr}"
initialize_csv_report(
    csv_filename,
    (
        "Number of teeth",
        "Comb spacing (Hz)",
        "Mean concentration (VMR)",
        "Standard deviation (VMR)",
    ),
)

####################################################################################################
# Initialize variables                                                                             #
####################################################################################################

# Counters

nr_config = 0
"""Number of the current configuration being simulated."""
nr_configs_performed = 0
"""Number of configurations that have been performed, excluding those that are skipped."""

# Other variables

time_start = perf_counter()
simulator = None

if use_latex:
    latex()

####################################################################################################
# Simulate for every combination of number of teeth and comb spacing                               #
####################################################################################################


for nr_teeth, spacing in zip(numbers_of_teeth, comb_spacings):
    # Update iteration variables ###################################################################

    nr_config += 1

    # Update simulation parameters #################################################################

    modulation_intensity = modulation_intensities.get(nr_teeth)

    # Make sure the comb span is within the limits

    comb_span = delta_frequency_to_delta_wavelength(
        spacing * (nr_teeth - 1), c / laser_wavelength * 1e9
    )
    comb_spacing = comb_span / (nr_teeth - 1)  # nm

    if comb_span + 2 * comb_spacing > wl_max - wl_min or comb_span < min_comb_span:
        print_skipped_config(nr_config, nr_teeth, spacing)
        continue

    # Laser wavelength slack range

    laser_wavelength_slack = (-comb_spacing, comb_spacing)  # nm

    # Show config parameters #######################################################################

    nr_configs_performed += 1

    print_config_params(nr_teeth, spacing)

    # Simulate and fit `nr_simulations_per_config` times ###########################################

    if detailed_report:
        initialize_csv_report(
            f"simulations-{nr_teeth}-{spacing / 1e9:.2f}", ("Concentration (VMR)",)
        )

    if generate_plots:
        # Create a folder in `figures` with name like `8 x 1.0 GHz` to store the plots

        sp = "{:g}".format(float("{:.{p}g}".format(spacing / 1e9, p=4)))
        folder_name = f"{nr_teeth} x {sp} GHz"
        folder_path = initialize_figures_folder(folder_name)

    fitting_results = []

    for i in range(nr_simulations_per_config):
        # Perform the simulation and fitting ###################################################

        tooth_std_threshold = tooth_std_threshold_start + (
            tooth_std_threshold_end - tooth_std_threshold_start
        ) * (nr_teeth - teeth_start) / (teeth_end - teeth_start)

        with Time(
            f"{i + 1}. Simulating for {nr_teeth} teeth and {spacing / 1e9:.2f} GHz"
        ):
            x_meas, y_meas, f, simulator = fit_simulated_measurement_concentration(
                molecule=molecule,
                wl_min=wl_min,
                wl_max=wl_max,
                vmr=vmr,
                pressure=pressure,
                temperature=temperature,
                length=length,
                laser_wavelength=laser_wavelength,
                optical_comb_spacing=spacing,
                number_of_teeth=nr_teeth,
                database=database,
                transmission_std=transmission_std,
                nr_teeth_for_transmission_std=nr_teeth_for_transmission_std,
                tooth_std_threshold=tooth_std_threshold,
                spectrum_shift_range=spectrum_shift_range,
                scaling_range=scaling_range,
                laser_wavelength_slack=laser_wavelength_slack,
                normalize=normalize,
                initial_guess=initial_guess,
                fitter=fitter,
                noise_distribution=noise_distribution,
                modulation_intensity=modulation_intensity,
                simulator=simulator,
                exit_gpu=False,
                return_simulator=True,
            )

            fitting_results.append(f.concentration)

        print_fitting_result(i, f.concentration)

        if detailed_report:
            append_to_csv_report(
                f"simulations-{nr_teeth}-{spacing / 1e9:.2f}",
                (f.concentration,),
            )

        # Plot the simulated and measured transmission spectra #####################################

        if not generate_plots:
            continue

        meas = f.measured_transmission
        sim = f.simulated_transmission

        x_sim, y_sim, x_meas_fitted, y_meas_fitted = (
            sim.x_nm,
            sim.y_nm,
            meas.x_nm,
            meas.y_nm,
        )

        with Time(f"{indentation(i)}Plotting"):
            plt.plot(
                x_sim,
                y_sim,
                label=f"Simulation for {f.concentration:.3f} VMR",
                color="blue",
                zorder=0,
            )
            plt.scatter(
                x_meas,
                y_meas,
                label="Simulated Measurement (pre-fitting)",
                color="red",
                zorder=1,
            )
            plt.scatter(
                x_meas_fitted,
                y_meas_fitted,
                label="Simulated Measurement (fitted)",
                color="green",
                zorder=2,
            )
            plt.legend()
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Transmission")
            plt.title(
                f"Transmission spectrum of {molecule} at {pressure:.2f} Pa "
                + f"and {temperature:.2f} K.\n{length:.3f} m path length, {vmr:.3f} VMR "
                + f"(fitted as {f.concentration:.3f} VMR)."
            )
            plt.tight_layout(**article_tight)
            plt.savefig(f"{folder_path}/fit-simulated-measurement-{i}.svg")
            plt.clf()

    # Aggregate results from all iterations ########################################################

    fitting_results = array(fitting_results)
    result = (nr_teeth, spacing, fitting_results.mean(), fitting_results.std())

    print_config_result(*result)

    # Append the result to the CSV report ##########################################################

    append_to_csv_report(csv_filename, result)

    # Print progress and time left #################################################################

    print_progress(nr_config, nr_configs_performed, time_start)

# Exit the GPU simulator if it was initialized #####################################################

if simulator is not None and simulator.use_gpu:
    simulator.exit_gpu()
