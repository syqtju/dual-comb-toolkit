from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    import numpy as np

    from lib.entities import Measurement
    from lib.fitting.concentration import ConcentrationFitter
    from lib.mapping import Mapper
    from lib.simulations import Simulator


def simulate_line(
    molecule: str, wl_min: float, wl_max: float, **conditions: dict[str, float]
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Simulate a line spectrum.

    Parameters
    ----------
    molecule : str
        The name of the molecule.
    wl_min : float
        The minimum wavelength in nm.
    wl_max : float
        The maximum wavelength in nm.

    Keyword Arguments
    -----------------
    vmr : float
        The volume mixing ratio of the molecule in air.
    pressure : float
        The pressure in Pa.
    temperature : float
        The temperature in K.
    length : float
        The length of the absorption path in m.
    database : str, optional
        The database to use. Either 'hitran' or 'hitemp'. Defaults to 'hitran' (defined in
        `lib.defaults`).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The wavelength and transmission spectrum.
    """
    from lib.defaults import DATABASE
    from lib.simulations import Simulator

    for key in ["vmr", "pressure", "temperature", "length"]:
        if key not in conditions:
            raise ValueError(f"Missing condition: {key}.")

    s = Simulator(
        molecule=molecule, database=conditions.pop("database", DATABASE), **conditions
    )
    s.compute_transmission_spectrum(wl_min=wl_min, wl_max=wl_max)
    wl, transmission = s.get_transmission_spectrum(wl_min, wl_max)

    return wl, transmission


def get_measurement(
    meas_name: str, **specifications: dict[str, float | int]
) -> "Measurement":
    """
    Get a measurement object.

    Parameters
    ----------
    meas_name : str
        The name of the measurement.

    Keyword Arguments
    -----------------
    center_freq : float
        The center frequency of the measurement.
    freq_spacing : float
        The frequency spacing of the measurement.
    number_of_teeth : int
        The number of teeth in the measurement.
    laser_wavelength : float
        The laser wavelength of the measurement.
    optical_comb_spacing : float
        Spacing between teeth of the measured optical comb in Hz.
    acq_freq : float
        The acquisition frequency of the measurement.

    Other Parameters
    ----------------
    baseline_names: list[str], optional
        The name of the measurements used to obtain the baseline.
    baseline : Baseline, optional
        The baseline object.
    molecule : str, optional
        The molecule measured.
    pressure : float, optional
        The pressure in Pa.
    temperature : float, optional
        The temperature in K.
    length : float, optional
        The length of the absorption path in m.
    concentration : float, optional
        The concentration of the molecule in VMR.
    sub_measurements : int, optional
        Number of sub-measurements used to obtain the standard deviation of the teeth. If not
        specified, tooth filtering based on standard deviation will not be applied.
    tooth_std_threshold : float, optional
        Teeth with a standard deviation above `tooth_std_threshold * mean_std` will be discarded if
        `sub_measurements` is given and is greater than 1.
    flip : bool, optional
        Whether to flip the measured transmission spectrum or not. Default is False.

    Returns
    -------
    Measurement
        The measurement object.
    """
    from lib.measurements import Measurement

    required_specs = [
        "center_freq",
        "freq_spacing",
        "number_of_teeth",
        "laser_wavelength",
        "optical_comb_spacing",
        "acq_freq",
    ]

    for key in required_specs:
        if key not in specifications:
            raise ValueError(f"Missing specification: {key}.")

    baseline_names = specifications.pop("baseline_names", None)
    baseline = specifications.pop("baseline", None)

    if not baseline and baseline_names:
        from lib.entities import Baseline

        baseline = Baseline(measurement_names=baseline_names, **specifications)

    return Measurement(meas_name, **specifications, baseline=baseline)


def fit_measurement_concentration(
    meas_name: str, **specifications
) -> "ConcentrationFitter":
    """
    Fit the concentration of a measurement to a simulation.

    Parameters
    ----------
    meas_name : str
        The name of the measurement.

    Keyword Arguments
    -----------------
    center_freq : float
        The center frequency of the measurement.
    freq_spacing : float
        The frequency spacing of the measurement.
    number_of_teeth : int
        The number of teeth in the measurement.
    laser_wavelength : float
        The laser wavelength of the measurement.
    optical_comb_spacing : float
        Spacing between teeth of the measured optical comb.
    acq_freq : float
        The acquisition frequency of the measurement.
    wl_min : float
        The minimum wavelength for the simulation in nm.
    wl_max : float
        The maximum wavelength for the simulation in nm.
    molecule : str
        The molecule measured.
    database : str, optional
        The database to use. Either 'hitran' or 'hitemp'. Defaults to 'hitran' (defined in
        `lib.defaults`).
    pressure : float
        The pressure in Pa.
    temperature : float
        The temperature in K.
    length : float
        The length of the absorption path in m.

    Other Parameters
    ----------------
    baseline_names: list[str], optional
        The name of the measurements used to obtain the baseline.
    initial_guess : float, optional
        Initial guess for the fitting. Defaults to 0.5.
    fitter : str, optional
        Fitter to use. Defaults to 'normal'. Possible values are 'normal', 'interp' and
        'normal_gpu'.
    lower_bound : float, optional
        Lower bound for the concentration. Defaults to 0.
    upper_bound : float, optional
        Upper bound for the concentration. Defaults to 1.
    verbose : bool, optional
        Whether to print the fitting results. Defaults to False.
    sub_measurements : int, optional
        Number of sub-measurements used to obtain the standard deviation of the teeth. If not
        specified, tooth filtering based on standard deviation will not be applied.
    tooth_std_threshold : float, optional
        Teeth with a standard deviation above `tooth_std_threshold * mean_std` will be discarded if
        `sub_measurements` is given and is greater than 1.
    flip : bool, optional
        Whether to flip the measured transmission spectrum or not. Default is False.
    remove_teeth_indices : list[int], optional
        List of indices of the teeth to be removed from the fitting.

    Returns
    -------
    ConcentrationFitter
        The concentration fitter object.
    """
    required_kwargs = [
        "wl_min",
        "wl_max",
        "molecule",
        "pressure",
        "temperature",
        "length",
    ]

    for key in required_kwargs:
        if key not in specifications:
            raise ValueError(f"Missing specification: {key}.")

    remove_teeth_indices = specifications.pop("remove_teeth_indices", [])

    from lib.fitting import ConcentrationFitter

    meas = get_measurement(meas_name, **specifications)
    meas._compute_transmission()
    meas.remove_teeth(remove_teeth_indices)
    meas_transmission = meas.transmission_spectrum

    wl_min: float = specifications.pop("wl_min")
    wl_max: float = specifications.pop("wl_max")

    fitter = ConcentrationFitter(
        meas_transmission=meas_transmission,
        wl_min=wl_min,
        wl_max=wl_max,
        **specifications,
    )

    return fitter


def map_measurement_concentration(meas_names: list[str], **specifications) -> "Mapper":
    """
    Map the concentration of a set of measurements.

    Parameters
    ----------
    meas_names : list[str]
        List of measurement names.

    Keyword Arguments
    -----------------
    center_freq : float
        Center frequency of the radio frequency comb in Hz.
    freq_spacing : float
        Spacing of the radio frequency comb in Hz.
    number_of_teeth : int
        Number of teeth to consider.
    laser_wavelength : float
        Approximate value of the laser wavelength in nm.
    optical_comb_spacing : float
        Spacing of the optical optical comb in Hz.
    acq_freq : float
        Acquisition frequency used in the measurement in Hz.
    molecule : str
        Name of the molecule measured and simulated.
    database : str, optional
        The database to use. Either 'hitran' or 'hitemp'. Defaults to 'hitran' (defined in
        `lib.defaults`).
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature of the gas in K.
    length : float
        Length of the absorption path in m.
    wl_min : float
        Minimum wavelength for the simulation in nm.
    wl_max : float
        Maximum wavelength for the simulation in nm.
    fitter : str, optional
        Fitter to use. Defaults to 'normal'. Possible values are 'normal', 'interp' and
        'normal_gpu'.

    Other Parameters
    ----------------
    initial_guess : dict[str, float], optional
        Initial guess for the fitting. Defaults to {'concentration': 0.5}.
    verbose : bool, optional
        Print the fitting results. Defaults to False.
    baseline_names : list[str], optional
        Names of the measurements used to obtain the baseline.
    spectrum_plot_folder : str, optional
        Folder to save the spectrum plots. Defaults to None, so no plots are saved.
    lower_bound : float, optional
        Lower bound for the concentration. Defaults to 0.0.
    upper_bound : float, optional
        Upper bound for the concentration. Defaults to 0.5.
    verbose : bool, optional
        Whether to print the fitting results. Defaults to False.
    sub_measurements : int, optional
        Number of sub-measurements used to obtain the standard deviation of the teeth. If not
        specified, tooth filtering based on standard deviation will not be applied.
    tooth_std_threshold : float, optional
        Teeth with a standard deviation above `tooth_std_threshold * mean_std` will be discarded if
        `sub_measurements` is given and is greater than 1.
    remove_teeth_indices : list[int], optional
        List of tooth indices to be removed from the fitting.

    Returns
    -------
    Mapper
        The mapper object.
    """
    from lib.mapping import Mapper

    baseline_names = specifications.pop("baseline_names", None)
    baseline = specifications.pop("baseline", None)
    remove_teeth_indices = specifications.get("remove_teeth_indices", [])

    if not baseline and baseline_names:
        from lib.entities import Baseline

        baseline_specifications = specifications.copy()
        baseline_specifications.pop("remove_teeth_indices", None)
        baseline = Baseline(measurement_names=baseline_names, **baseline_specifications)
        specifications["baseline"] = baseline

    meas_transmissions = []

    for meas_name in meas_names:
        meas = get_measurement(meas_name, **specifications)
        meas._compute_transmission()
        meas.remove_teeth(remove_teeth_indices)
        meas_transmissions.append(meas.transmission_spectrum)

    return Mapper(meas_transmissions, **specifications)


def get_measurement_spectrum(
    meas_name: str, acq_freq: "Optional[float]" = None
) -> None:
    """
    Compute the spectrum of a measurement.

    Parameters
    ----------
    meas_name : str
        The name of the measurement.
    acq_freq : float, optional
        The acquisition frequency of the measurement. If not provided, it will be read from the
        measurement file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The frequency and amplitude of the spectrum.
    """
    from lib.analysis.time_domain import FFTCalculator
    from lib.measurements import read_lvm

    t, a = read_lvm(meas_name, acq_freq=acq_freq)
    fftc = FFTCalculator(t, a)

    return fftc.fft_x, fftc.fft_y


def fit_simulated_measurement_concentration(
    molecule: str, wl_min: float, wl_max: float, **kwargs: dict[str, float]
) -> "tuple[np.ndarray, np.ndarray, ConcentrationFitter] | tuple[np.ndarray, np.ndarray, ConcentrationFitter, Simulator]":
    """
    Fit a simulated spectrum to a simulation of a measured spectrum.

    Parameters
    ----------
    molecule : str
        The name of the molecule.
    wl_min : float
        The minimum wavelength in nm.
    wl_max : float
        The maximum wavelength in nm.
    vmr : float
        The volume mixing ratio of the molecule in air.
    pressure : float
        The pressure in Pa.
    temperature : float
        The temperature in K.
    length : float
        The length of the absorption path in m.
    laser_wavelength : float
        The laser wavelength in nm.
    optical_comb_spacing : float
        Spacing of the optical optical comb in Hz.
    number_of_teeth : int
        The number of teeth in the measurement.

    Other Parameters
    ----------------
    database : str, optional
        The database to use. Either 'hitran' or 'hitemp'. Defaults to 'hitran' (defined in
        `lib.defaults`).
    transmission_std : float, optional
        The standard deviation of the noise. Defaults to 0.005.
    tooth_std_threshold : float, optional
        Teeth with a standard deviation above `tooth_std_threshold * mean_std` will be discarded.
        This is used only when `noise_distribution` is set to 'bessel'.
    spectrum_shift_range : float, optional
        The range of the horizontal spectrum shift. Defaults to (0, 0).
    scaling_range : float, optional
        The range of the scaling factor applied to the simulated spectrum. Defaults to (1, 1).
    laser_wavelength_slack : tuple(float), optional
        Range of the laser wavelength's random variation. Defaults to (-0.05, 0.05).
    normalize : bool, optional
        Normalize the transmission spectrum. Defaults to False.
    initial_guess : float, optional
        Initial guess for the fitting. Defaults to 0.5.
    fitter : str, optional
        Fitter to use. Defaults to 'normal'. Possible values are 'normal', 'interp' and
        'normal_gpu'.
    simulator : Simulator, optional
        A pre-initialized simulator object to use for the simulation. If not provided,
        a new simulator object will be created with the specified parameters.
    use_gpu : bool, optional
        Whether to use GPU for the simulation. Default is False. Only applies if
        `simulator` is not provided.
    exit_gpu : bool, optional
        Whether to exit the GPU after the simulation. Default is True. Only applies if
        `use_gpu` is set to True in the simulator.
    return_simulator : bool, optional
        Whether to return the simulator object after the simulation. Default is False.
    wavelength_step : float, optional
        The step size for the wavelength grid in nm. Defaults to `WAVELENGTH_STEP` (defined in
        `lib.defaults`). Only applies if `simulator` is not provided.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, ConcentrationFitter] | tuple[np.ndarray, np.ndarray, ConcentrationFitter, Simulator]
        The wavelength and transmission spectrum of the measurement, and the concentration fitter.
        Also returns the simulator object if `return_simulator` is True.
    """
    required_kwargs = [
        "vmr",
        "pressure",
        "temperature",
        "length",
        "laser_wavelength",
        "optical_comb_spacing",
        "number_of_teeth",
    ]

    for key in required_kwargs:
        if key not in kwargs:
            raise ValueError(f"Missing specification: {key}.")

    kwargs["concetration"] = kwargs["vmr"]

    from lib.entities import MeasuredSpectrum
    from lib.fitting.concentration import ConcentrationFitter
    from lib.simulations import simulate_measurement

    return_simulator = kwargs.get("return_simulator", False)
    kwargs["return_simulator"] = True
    exit_gpu = kwargs.get("exit_gpu", True)
    kwargs["exit_gpu"] = False

    if kwargs.get("fitter", "normal") == "normal_gpu":
        kwargs["use_gpu"] = True

    x_meas, y_meas, simulator = simulate_measurement(
        molecule=molecule, wl_min=wl_min, wl_max=wl_max, **kwargs
    )
    kwargs["simulator"] = simulator

    x_meas_fitted = x_meas.copy()
    y_meas_fitted = y_meas.copy()

    if kwargs.get("normalize", False):
        from lib.combs import normalize_transmission

        x_meas_fitted, y_meas_fitted = normalize_transmission(
            x_meas_fitted, y_meas_fitted, replace_outliers=False
        )

    # Create a MeasuredSpectrum object.
    meas_transmission = MeasuredSpectrum(
        x_meas_fitted, y_meas_fitted, xu="nm", molecule=molecule, **kwargs
    )

    # Fit the concentration.
    kwargs["exit_gpu"] = exit_gpu
    f = ConcentrationFitter(
        meas_transmission=meas_transmission, wl_min=wl_min, wl_max=wl_max, **kwargs
    )

    if return_simulator:
        return x_meas, y_meas, f, simulator

    return x_meas, y_meas, f
