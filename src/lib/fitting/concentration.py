from functools import cached_property
from typing import TYPE_CHECKING

from lib.fitting import overlap_transmission
from lib.math import bounded

if TYPE_CHECKING:
    from typing import Callable, Optional

    from numpy import ndarray

    from lib.entities import MeasuredSpectrum, Result, SimulatedSpectrum
    from lib.simulations import Simulator


def assemble_normal_fitter(
    molecule: str,
    wl_min: float,
    wl_max: float,
    conditions: dict[str, float],
    initial_guess: float = 0.001,
    **kwargs,
) -> "tuple[Callable[..., tuple[ndarray, ndarray]], Simulator]":
    """
    Assemble a normal fitter for the concentration fitting process.
    This function returns a callable that can be used to simulate the transmission spectrum for a
    given concentration.

    Parameters
    ----------
    molecule : str
        Molecule name for the simulation.
    wl_min : float
        Minimum wavelength for the simulation in nm.
    wl_max : float
        Maximum wavelength for the simulation in nm.
    conditions : dict[str, float]
        Conditions for the simulation, must include 'pressure', 'temperature', and 'length'.
    initial_guess : float, optional
        Initial guess for the concentration.

    Other Parameters
    ----------------
    simulator : Optional[Simulator], optional
        Pre-initialized simulator to use. If not provided, a new simulator will be created.
        Defaults to None.
    use_gpu : bool, optional
        Whether to use the GPU for the simulation. Defaults to False. Only used if `simulator`
        is not provided.

    Returns
    -------
    callable[..., tuple[ndarray, ndarray]]
        A callable that takes a concentration as input and returns the simulated transmission spectrum
        as a tuple of frequency and amplitude arrays.
    Simulator
        Simulator used for the fitting.
    """
    from lib.combs import to_frequency
    from lib.defaults import DATABASE
    from lib.simulations import Simulator

    simulator: "Optional[Simulator]" = kwargs.get("simulator", None)
    use_gpu: bool = kwargs.get("use_gpu", False)
    database: str = kwargs.get("database", DATABASE)

    # Obtain simulator

    if not isinstance(simulator, Simulator):
        s = Simulator(
            molecule=molecule,
            vmr=initial_guess,
            pressure=conditions["pressure"],
            temperature=conditions["temperature"],
            length=conditions["length"],
            use_gpu=use_gpu,
            database=database,
        )
    else:
        s = simulator
        s.vmr = initial_guess
        s.pressure = conditions["pressure"]
        s.temperature = conditions["temperature"]
        s.length = conditions["length"]

    if molecule != s.molecule:
        raise ValueError(
            f"Simulator molecule {s.molecule} does not match the provided molecule {molecule}."
        )

    def simulate_transmission(conc: float, **kwargs) -> "tuple[ndarray, ndarray]":
        """
        Get the simulated transmission spectrum for a given concentration.
        """
        exit_gpu: bool = kwargs.get("exit_gpu", True)
        s.vmr = conc
        s.compute_transmission_spectrum(wl_min=wl_min, wl_max=wl_max, exit_gpu=exit_gpu)
        wl_ref, a_ref = s.get_transmission_spectrum(wl_min, wl_max)
        return to_frequency(wl_ref, a_ref)

    return simulate_transmission, s


def assemble_interpolated_fitter(
    molecule: str,
    wl_min: float,
    wl_max: float,
    conditions: dict[str, float],
    **kwargs,
) -> "Callable[..., tuple[ndarray, ndarray]]":
    """
    Assemble an interpolated fitter for the concentration fitting process.
    This function returns a callable that can be used to simulate the transmission spectrum for a
    given concentration using an interpolated transmission curve.

    Parameters
    ----------
    molecule : str
        Molecule name for the simulation.
    wl_min : float
        Minimum wavelength for the simulation in nm.
    wl_max : float
        Maximum wavelength for the simulation in nm.
    conditions : dict[str, float]
        Conditions for the simulation, must include 'pressure', 'temperature', and 'length'.

    Other Parameters
    ----------------
    n_points : int, optional
        The number of points to use for the interpolation. Default is 3.
    points : list[float], optional
        The points to use for the interpolation. Default is None. If specified, takes precedence
        over `n_points`.
    database : str, optional
        The database to use for the simulation. Either 'hitran' or 'hitemp'. Defaults to 'hitran'
        (defined in `lib.defaults`).

    Returns
    -------
    callable[[float, ...], tuple[ndarray, ndarray]]
        A callable that takes a concentration as input and returns the simulated transmission spectrum
        as a tuple of frequency and amplitude arrays.
    """
    from lib.combs import to_frequency
    from lib.defaults import DATABASE
    from lib.simulations import curry_interpolated_transmission_curve

    database: str = kwargs.get("database", DATABASE)

    transmission_curve = curry_interpolated_transmission_curve(
        wl_min, wl_max, molecule, database=database, **conditions
    )

    def simulate_transmission(conc: float, **kwargs) -> "tuple[ndarray, ndarray]":
        """
        Get the simulated transmission spectrum for a given concentration.
        """
        wl_ref, a_ref = transmission_curve(conc)
        return to_frequency(wl_ref, a_ref)

    return simulate_transmission


def fit_concentration(
    meas_freq: "ndarray",
    meas_amp: "ndarray",
    molecule: str,
    wl_min: float,
    wl_max: float,
    conditions: dict[str, float],
    initial_guess: float = 0.001,
    fitter: str = "normal",
    **kwargs,
) -> "tuple[float, ndarray, ndarray]":
    """
    Fit the concentration of a measured spectrum to a simulated spectrum.

    Parameters
    ----------
    meas_freq : ndarray
        Measured frequency data in Hz.
    meas_amp : ndarray
        Measured amplitude data.
    molecule : str
        Molecule name for the simulation.
    wl_min : float
        Minimum wavelength for the simulation in nm.
    wl_max : float
        Maximum wavelength for the simulation in nm.
    conditions : dict[str, float]
        Conditions for the simulation, must include 'pressure', 'temperature', and 'length'.
    initial_guess : float, optional
        Initial guess for the concentration. Defaults to 0.001.
    fitter : str, optional
        Fitter to use. Defaults to 'normal'. Possible values are 'normal' and 'interp'.

    Other Parameters
    ----------------
    simulator : Optional[Simulator], optional
        Pre-initialized simulator to use. If not provided, a new simulator will be created.
        Only used if `fitter` is 'normal'.
    use_gpu : bool, optional
        Whether to use the GPU for the simulation. Defaults to False. Only used if `simulator` is not provided.
        Only used if `fitter` is 'normal'.
    exit_gpu : bool, optional
        Whether to exit the GPU after the fitting. Defaults to True. Only used if `simulator` uses the GPU or
        if `use_gpu` is provided. Only used if `fitter` is 'normal'.
    n_points : int, optional
        Number of points to use for the interpolation if `fitter` is 'interp'. Defaults to 3.
        Only used if `fitter` is 'interp'.
    points : Optional[list[float]], optional
        Points to use for the interpolation if `fitter` is 'interp'. Defaults to None. If specified, takes precedence
        over `n_points`. Only used if `fitter` is 'interp'.
    lower_bound : float, optional
        Lower bound for the concentration. Defaults to 0.
    upper_bound : float, optional
        Upper bound for the concentration. Defaults to 1.
    database : str, optional
        The database to use for the simulation. Either 'hitran' or 'hitemp'. Defaults to 'hitran' 
        (defined in `lib.defaults`). Only used if `simulator` is not provided.
    verbose : bool, optional
        Whether to print the results of the fitting. Defaults to False.

    Returns
    -------
    concentration : float
        Fitted concentration.
    f_ref : ndarray
        Frequency data of the simulated spectrum in Hz.
    a_ref : ndarray
        Amplitude data of the simulated spectrum.
    f_sample : ndarray
        Frequency data of the measured spectrum in Hz.
    a_sample : ndarray
        Amplitude data of the measured spectrum.
    simulator : Optional[Simulator]
        Simulator used for the fitting. Only returned if `fitter` is 'normal' and the simulator was
        initialized within this function.
    """
    from lib.defaults import DATABASE

    tol = 1e-6
    database = kwargs.get("database", DATABASE)
    verbose = kwargs.get("verbose", False)
    simulator = None

    condition_names = ["pressure", "temperature", "length"]
    for name in condition_names:
        if name not in conditions:
            raise ValueError(f"Missing condition: {name}.")

    import numpy as np
    from scipy.optimize import minimize

    from lib.fitting.interpolation import closest_value_indices

    # Assemble the simulator
    if fitter == "interp":
        n_points: int = kwargs.get("n_points", 3)
        points: "Optional[list[float]]" = kwargs.get("points", None)

        simulate_transmission = assemble_interpolated_fitter(
            molecule,
            wl_min,
            wl_max,
            conditions,
            n_points=n_points,
            points=points,
            database=database,
        )
    elif fitter == "normal":
        simulator: "Optional[Simulator]" = kwargs.get("simulator", None)
        use_gpu: bool = kwargs.get("use_gpu", False)
        exit_gpu: bool = kwargs.get("exit_gpu", True)

        simulate_transmission, simulator = assemble_normal_fitter(
            molecule,
            wl_min,
            wl_max,
            conditions,
            initial_guess=initial_guess,
            simulator=simulator,
            use_gpu=use_gpu,
            database=database,
        )

    # Objective function to minimize
    def f(conc: float, f_sample: "ndarray", a_sample: "ndarray") -> float:
        # Get the simulated curve
        f_ref, a_ref = simulate_transmission(conc, exit_gpu=False)

        # Shift the sample spectrum to overlap with the simulated data as much as possible
        f_sample = overlap_transmission(f_sample, a_sample, f_ref, a_ref)

        # Sample the simulated data to the measurement frequency grid
        idcs = closest_value_indices(f_ref, f_sample)
        a_ref_com = a_ref[idcs]
        a_sample_com: "ndarray" = (
            a_sample * a_ref_com.max()
        )  # To account for measurements not covering the full line width

        return np.abs(a_ref_com - a_sample_com).mean()

    # Set bounds for the concentration and validate the initial guess
    upper_bound = max(min(kwargs.get("upper_bound", 1), 1), 0)
    lower_bound = max(min(kwargs.get("lower_bound", 0), 1), 0)
    initial_guess = bounded(initial_guess, lower_bound, upper_bound)

    # Perform the minimization
    result = minimize(
        f,
        initial_guess,
        args=(meas_freq, meas_amp),
        tol=tol,
        bounds=[(lower_bound, upper_bound)],
        method="Nelder-Mead",
    )
    concentration = result.x[0]
    nr_iterations = result.nit
    nr_function_evaluations = result.nfev

    # Get the simulated curve
    f_ref, a_ref = simulate_transmission(concentration, exit_gpu=exit_gpu)

    # Shift the sample spectrum to overlap with the simulated data as much as possible
    meas_freq = overlap_transmission(meas_freq, meas_amp, f_ref, a_ref)

    # Scale the measured amplitude
    idcs = closest_value_indices(f_ref, meas_freq)
    scaling_factor = a_ref[idcs].max()

    if verbose:
        print(
            f"Fitted concentration: {concentration:.6f} VMR ({nr_iterations} iterations and {nr_function_evaluations} objective function evaluations)."
        )

    return concentration, f_ref, a_ref, meas_freq, meas_amp * scaling_factor, simulator


class ConcentrationFitter:
    """
    Fit the concentration of a measured spectrum to a simulated spectrum. The measured spectrum
    object `meas_transmission` must have the properties `molecule`, `pressure`, `temperature`, and
    `length` set.

    Parameters
    ----------
    meas_transmission : MeasuredSpectrum
        Measured transmission spectrum to fit.
    wl_min : float
        Minimum wavelength for the simulation in nm.
    wl_max : float
        Maximum wavelength for the simulation in nm.

    Other Parameters
    ----------------
    initial_guess : float, optional
        Initial guess for the concentration. Defaults to 0.5.
    fitter : str, optional
        Fitter to use. Defaults to 'normal'. Possible values are 'normal', 'interp' and
        'normal_gpu'.
    simulator : Optional[Simulator], optional
        Pre-initialized simulator to use. If not provided, a new simulator will be created.
        Defaults to None. Only used if `fitter` is 'normal_gpu'.
    exit_gpu : bool, optional
        Whether to exit the GPU after the fitting. Defaults to True. Only used if `fitter` is
        'normal_gpu'.
    n_points : int, optional
        Number of points to use for the interpolation if `fitter` is 'interp'. Defaults to 3.
        Only used if `fitter` is 'interp'.
    points : Optional[list[float]], optional
        Points to use for the interpolation if `fitter` is 'interp'. Defaults to None. If specified, takes precedence
        over `n_points`. Only used if `fitter` is 'interp'.
    lower_bound : float, optional
        Lower bound for the concentration. Defaults to 0.
    upper_bound : float, optional
        Upper bound for the concentration. Defaults to 1.
    database : str, optional
        The database to use for the simulation. Either 'hitran' or 'hitemp'. Defaults to 'hitran' 
        (defined in `lib.defaults`). Only used if `simulator` is not provided.
    verbose : bool, optional
        Whether to print the results of the fitting. Defaults to False.
    """

    def __init__(
        self,
        meas_transmission: "MeasuredSpectrum",
        wl_min: float,
        wl_max: float,
        **kwargs: dict[str, float],
    ) -> None:
        from lib.defaults import DATABASE

        # Measurement parameters

        self._pre_meas_trasmission = meas_transmission

        self._meas_transmission: "Optional[MeasuredSpectrum]" = None

        # Simulation parameters

        self.wl_min = wl_min
        self.wl_max = wl_max
        self.initial_guess: float = kwargs.get("initial_guess", 0.5)
        self.lower_bound: float = bounded(kwargs.get("lower_bound", 0.0), 0, 1)
        self.upper_bound: float = bounded(kwargs.get("upper_bound", 1.0), 0, 1)
        self.fitter: str = kwargs.get("fitter", "normal")
        self.verbose: bool = kwargs.get("verbose", False)
        self.database: str = kwargs.get("database", DATABASE)

        if self.lower_bound >= self.upper_bound:
            raise ValueError("Lower bound must be less than upper bound.")

        self._check_fitter()

        self._sim_transmission: "Optional[SimulatedSpectrum]" = None

        # Simulator

        self.simulator: "Optional[Simulator]" = kwargs.get("simulator", None)
        self._exit_gpu: bool = kwargs.get("exit_gpu", True)

        # Interpolation parameters
        self.n_points: int = kwargs.get("n_points", 3)
        self.points: "Optional[list[float]]" = kwargs.get("points", None)

        # Conditions

        required_conditions = ["molecule", "pressure", "temperature", "length"]

        for condition in required_conditions:
            if getattr(meas_transmission, condition) is None:
                raise ValueError(
                    f"Missing property: {condition}. This property must be set"
                    + " in the measured spectrum `meas_transmission`."
                )

        self.molecule = meas_transmission.molecule
        self.pressure = meas_transmission.pressure
        self.temperature = meas_transmission.temperature
        self.length = meas_transmission.length

    # Properties

    @cached_property
    def result(self) -> "Result":
        from lib.entities import Result

        return Result(self.measured_transmission, self.simulated_transmission)

    @property
    def measured_transmission(self) -> "MeasuredSpectrum":
        if self._meas_transmission is None:
            self._process()
        return self._meas_transmission

    @property
    def simulated_transmission(self) -> "SimulatedSpectrum":
        if self._sim_transmission is None:
            self._process()
        return self._sim_transmission

    @property
    def concentration(self) -> float:
        return self.simulated_transmission.concentration

    @property
    def conditions(self) -> dict[str, float]:
        return {
            "pressure": self.pressure,
            "temperature": self.temperature,
            "length": self.length,
        }

    def _check_fitter(self) -> None:
        if self.fitter not in ["normal", "interp", "normal_gpu"]:
            raise ValueError(
                f'Invalid fitter: {self.fitter}. Possible values are "normal", '
                + '"interp" and "normal_gpu".'
            )

    def _process(self) -> None:
        from lib.entities import MeasuredSpectrum, SimulatedSpectrum

        self._check_fitter()

        use_gpu = False
        fitter = "normal"

        if self.fitter == "normal":
            use_gpu = False
            fitter = "normal"
        elif self.fitter == "normal_gpu":
            use_gpu = True
            fitter = "normal"
        elif self.fitter == "interp":
            use_gpu = False
            fitter = "interp"

        if self.verbose:
            print(f"Fitting {self._pre_meas_trasmission.meas_name} ... ", end="")

        concentration, sim_freq, sim_amp, meas_freq, meas_amp, self.simulator = fit_concentration(
            self._pre_meas_trasmission.x_hz,
            self._pre_meas_trasmission.y_hz,
            self.molecule,
            self.wl_min,
            self.wl_max,
            self.conditions,
            self.initial_guess,
            fitter=fitter,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            verbose=self.verbose,
            simulator=self.simulator,
            use_gpu=use_gpu,
            exit_gpu=self._exit_gpu,
            n_points=self.n_points,
            points=self.points,
            database=self.database,
        )

        self._sim_transmission = SimulatedSpectrum(
            sim_freq,
            sim_amp,
            xu="Hz",
            molecule=self.molecule,
            pressure=self.pressure,
            temperature=self.temperature,
            concentration=concentration,
            length=self.length,
            wl_min=self.wl_min,
            wl_max=self.wl_max,
        )

        self._meas_transmission = MeasuredSpectrum(
            meas_freq,
            meas_amp,
            xu="Hz",
            meas_name=self._pre_meas_trasmission.meas_name,
            center_freq=self._pre_meas_trasmission.center_freq,
            freq_spacing=self._pre_meas_trasmission.freq_spacing,
            number_of_teeth=self._pre_meas_trasmission.number_of_teeth,
            laser_wavelength=self._pre_meas_trasmission.laser_wavelength,
            optical_comb_spacing=self._pre_meas_trasmission.optical_comb_spacing,
            acq_freq=self._pre_meas_trasmission.acq_freq,
            y_sdv=self._pre_meas_trasmission.y_sdv_hz,
        )
