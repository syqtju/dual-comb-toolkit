"""Input file for the simulated-fittings pipeline (``configs--simulated-fittings-run.py``).

Everything below is a *global default*. Each entry of ``CONDITIONS`` is a physical condition
identified by a unique ``id`` and may override ANY of the globals (molecule, temperature,
laser wavelength, whether to emit plots/detail reports, even the sweep itself). A condition is
resolved as ``{**globals, **condition}``.

Run with::

    python configs--simulated-fittings-run.py            # uses this file
    python configs--simulated-fittings-run.py my_run.py  # uses another file
"""

####################################################################################################
#  Molecule and database                                                                           #
####################################################################################################

molecule = "H2O"
database = "hitemp"  # 'hitran', 'hitemp', 'exomol' or 'geisa'

####################################################################################################
#  Physical conditions (global defaults; override per condition below)                             #
####################################################################################################

vmr = 0.01  # volume mixing ratio
pressure = 101325  # Pa
temperature = 1000  # K
length = 1  # m
laser_wavelength = 1388.14  # nm

####################################################################################################
#  Simulation range                                                                                #
####################################################################################################

wl_range = 0.15  # nm (window centered on laser_wavelength)
wl_step = wl_range / 50  # nm (wavelength grid step; None -> wl_range / 50)
min_comb_span = 0.03  # nm (configs with a smaller comb span are skipped)

####################################################################################################
#  Noise and error model                                                                           #
####################################################################################################

noise_distribution = "bessel"  # 'bessel' or 'uniform'
transmission_std = 0.014
nr_teeth_for_transmission_std = 30
tooth_std_threshold_start = 2.5
teeth_start = 5
tooth_std_threshold_end = 1.5
teeth_end = 30
spectrum_shift_range = (-0.02, 0.02)  # nm
scaling_range = (0.95, 1.05)  # unitless
modulation_intensities = {
    5: 1.6,
    6: 2.8,
    7: 2.8,
    8: 3.1,
    9: 3.1,
    10: 4.4,
    11: 4.4,
    12: 4.6,
    13: 4.6,
    14: 6.0,
    15: 6.0,
    16: 6.2,
    17: 6.2,
    18: 7.6,
    19: 7.6,
    20: 7.8,
    21: 7.8,
    22: 9.2,
    23: 9.2,
    24: 9.3,
    25: 9.3,
    26: 10.8,
    27: 10.8,
    28: 10.9,
    29: 10.9,
    30: 12.4,
}

####################################################################################################
#  Optional outputs (global defaults; switch on per condition for just the ones you want)          #
####################################################################################################

generate_plots = True  # per-fit spectra plots
plot_every = 10  # if generate_plots, plot every Nth simulation
use_latex = False
detailed_report = False  # per-config CSV of every individual fit

####################################################################################################
#  Fitting parameters                                                                              #
####################################################################################################

normalize = True
initial_guess = 0.001  # VMR
lower_bound = 0.0  # VMR
upper_bound = 0.05  # VMR
fitter = "normal_gpu"  # 'normal', 'normal_gpu' or 'interp'

####################################################################################################
#  Sweep (the comb configurations to simulate)                                                     #
####################################################################################################

nr_simulations_per_config = 100
comb_spacings = [(i + 1) * 100e6 for i in range(30)] * 26  # Hz
numbers_of_teeth = [i for i in range(5, 31) for _ in range(30)]  # teeth

####################################################################################################
#  Execution (run-level; NOT overridable per condition)                                            #
####################################################################################################

max_workers = 16  # GPU-bound: this caps concurrency regardless of conditions x shards
shards_per_condition = 6  # how finely each condition's sweep is split across workers

####################################################################################################
#  Physical conditions to simulate                                                                 #
####################################################################################################

CONDITIONS = [
    dict(id="1388.14nm@298K", laser_wavelength=1388.14, temperature=298),
    dict(id="1388.14nm@600K", laser_wavelength=1388.14, temperature=600),
    dict(id="1388.14nm@1000K", laser_wavelength=1388.14, temperature=1000),
    dict(id="1344.88nm@298K", laser_wavelength=1344.88, temperature=298),
    dict(id="1344.88nm@600K", laser_wavelength=1344.88, temperature=600),
    dict(
        id="1344.88nm@1000K",
        laser_wavelength=1344.88,
        temperature=1000,
        # Example of per-condition extras: emit per-fit plots and detail reports for this one.
        generate_plots=True,
        detailed_report=True,
    ),
]
