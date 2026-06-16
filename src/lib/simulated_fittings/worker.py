"""The per-shard worker: simulate and fit a chunk of comb configurations for one condition.

Can run inside a process pool (orchestrator) or directly (debug CLI). It is deliberately
self-contained and picklable-by-reference so ``ProcessPoolExecutor`` can call it in a spawned
child.

Progress is reported through ``progress_q`` (a ``multiprocessing`` queue) when given; the
orchestrator drains it to drive the live dashboard and the run log. When ``progress_q`` is
``None`` (debug CLI), events are printed to stdout instead.
"""

import csv
import os
import sys
import warnings
from typing import Any, Optional

import matplotlib

# Workers are headless: never try to open a GUI backend in a child process.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy import array

from lib import defaults
from lib.constants import c
from lib.conversions import delta_frequency_to_delta_wavelength
from lib.plots import article_tight
from lib.shortcuts import fit_simulated_measurement_concentration

# Result row schema (also the results.csv header).
REPORT_HEADER = (
    "Number of teeth",
    "Comb spacing (Hz)",
    "Mean concentration (VMR)",
    "Standard deviation (VMR)",
)

# Set by the pool initializer / run_shard so warnings can be tagged and routed to the run log.
_progress_q: Optional[Any] = None
_current_cid: Optional[str] = None


def _route_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """Send a warning to the run log via the progress queue instead of to the terminal."""
    if _progress_q is not None:
        try:
            location = f"{os.path.basename(filename)}:{lineno}"
            _progress_q.put(
                (_current_cid, "warn", f"{category.__name__}: {message} ({location})")
            )
        except Exception:
            pass


def init_worker(progress_q: Any) -> None:
    """Pool initializer: keep worker output off the live dashboard.

    Worker processes share the parent's terminal, so stray stdout/stderr (library prints, Python
    warnings) would scribble over the ``rich`` live display. We silence worker stdout/stderr and
    route Python warnings to the run log through the progress queue instead. Worker crashes are
    unaffected: exceptions are returned to the orchestrator through the future, not via stderr.
    """
    global _progress_q
    _progress_q = progress_q

    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    warnings.simplefilter("once")  # dedupe repeated warnings to one line per worker
    # Harmless library cleanup noise (RADIS leaves the HITEMP .hdf5 handle / asyncio loop
    # for the GC); don't bother routing it to the run log.
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.showwarning = _route_warning


def _emit(progress_q, event: tuple) -> None:
    """Send a progress event to the queue, or print a human-readable line if no queue."""
    if progress_q is not None:
        progress_q.put(event)
        return

    kind = event[1]
    if kind == "skip":
        _, _, nr_teeth, spacing = event
        print(
            f"  skip {nr_teeth} teeth, {spacing / 1e9:.2f} GHz (comb span out of range)"
        )
    elif kind == "result":
        _, _, nr_teeth, spacing, mean, std, _ = event
        print(
            f"  {nr_teeth} teeth, {spacing / 1e9:.2f} GHz -> "
            f"mean {mean:.6f} VMR, std {std:.6f} VMR"
        )


def _resolve_wavelength_grid(
    resolved: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return (wl_min, wl_max, wl_step, min_comb_span) for the condition."""
    laser_wavelength = resolved["laser_wavelength"]
    wl_range = resolved.get("wl_range", defaults.WL_RANGE)
    wl_min = laser_wavelength - wl_range / 2
    wl_max = laser_wavelength + wl_range / 2
    wl_step = resolved.get("wl_step") or wl_range / 50
    min_comb_span = resolved.get("min_comb_span", defaults.MIN_COMB_SPAN)
    return wl_min, wl_max, wl_step, min_comb_span


def run_shard(
    resolved: dict[str, Any],
    shard: list[tuple[float, int]],
    cond_dir: str,
    progress_q: Optional[Any] = None,
) -> list[tuple[int, float, float, float]]:
    """Simulate and fit every comb configuration in ``shard`` for one physical condition.

    Parameters
    ----------
    resolved : dict
        The fully resolved condition parameters (``{**globals, **condition}``), including
        ``id``.
    shard : list[tuple[float, int]]
        The ``(comb_spacing_hz, number_of_teeth)`` configurations to process.
    cond_dir : str
        Absolute path of this condition's output folder. Detail reports go in
        ``<cond_dir>/details/`` and per-fit plots in ``<cond_dir>/<teeth> x <spacing> GHz/``.
    progress_q : multiprocessing queue, optional
        Channel for progress events. If ``None``, events are printed.

    Returns
    -------
    list[tuple[int, float, float, float]]
        One ``(number_of_teeth, comb_spacing_hz, mean_vmr, std_vmr)`` row per *performed*
        configuration (skipped configurations are omitted).
    """
    cid = resolved["id"]

    global _current_cid, _progress_q
    _current_cid = cid
    if progress_q is not None:
        _progress_q = progress_q

    molecule = resolved["molecule"]
    database = resolved.get("database")
    vmr = resolved["vmr"]
    pressure = resolved["pressure"]
    temperature = resolved["temperature"]
    length = resolved["length"]
    laser_wavelength = resolved["laser_wavelength"]

    wl_min, wl_max, wl_step, min_comb_span = _resolve_wavelength_grid(resolved)

    noise_distribution = resolved.get("noise_distribution", defaults.NOISE_DISTRIBUTION)
    transmission_std = resolved.get("transmission_std", defaults.TRANSMISSION_STD)
    nr_teeth_for_transmission_std = resolved.get("nr_teeth_for_transmission_std")
    tooth_std_threshold_start = resolved.get(
        "tooth_std_threshold_start", defaults.TOOTH_STD_THRESHOLD_START
    )
    teeth_start = resolved.get("teeth_start", defaults.TEETH_START)
    tooth_std_threshold_end = resolved.get(
        "tooth_std_threshold_end", defaults.TOOTH_STD_THRESHOLD_END
    )
    teeth_end = resolved.get("teeth_end", defaults.TEETH_END)
    spectrum_shift_range = resolved.get(
        "spectrum_shift_range", defaults.SPECTRUM_SHIFT_RANGE
    )
    scaling_range = resolved.get("scaling_range", defaults.SCALING_RANGE)
    modulation_intensities = resolved.get("modulation_intensities") or {}

    generate_plots = resolved.get("generate_plots", defaults.GENERATE_PLOTS)
    plot_every = resolved.get("plot_every") or defaults.PLOT_EVERY
    detailed_report = resolved.get("detailed_report", defaults.DETAILED_REPORT)

    normalize = resolved.get("normalize", defaults.NORMALIZE)
    initial_guess = resolved.get("initial_guess", defaults.INITIAL_GUESS)
    fitter = resolved.get("fitter", defaults.FITTER)
    nr_simulations_per_config = resolved["nr_simulations_per_config"]

    rows: list[tuple[int, float, float, float]] = []
    simulator = None

    for spacing, nr_teeth in shard:
        # Keep the comb span within the simulated window. ##############################
        comb_span = delta_frequency_to_delta_wavelength(
            spacing * (nr_teeth - 1), c / laser_wavelength * 1e9
        )
        comb_spacing = comb_span / (nr_teeth - 1)  # nm

        if comb_span + 2 * comb_spacing > wl_max - wl_min or comb_span < min_comb_span:
            _emit(progress_q, (cid, "skip", nr_teeth, spacing))
            continue

        laser_wavelength_slack = (-comb_spacing, comb_spacing)  # nm
        modulation_intensity = modulation_intensities.get(nr_teeth)

        tooth_std_threshold = tooth_std_threshold_start + (
            tooth_std_threshold_end - tooth_std_threshold_start
        ) * (nr_teeth - teeth_start) / (teeth_end - teeth_start)

        if detailed_report:
            detail_path = os.path.join(
                cond_dir, "details", f"sim-{nr_teeth}-{spacing / 1e9:.2f}.csv"
            )
            os.makedirs(os.path.dirname(detail_path), exist_ok=True)
            detail_file = open(detail_path, "w", newline="")
            detail_writer = csv.writer(detail_file)
            detail_writer.writerow(("Concentration (VMR)",))
        else:
            detail_file = None
            detail_writer = None

        if generate_plots:
            sp = "{:g}".format(float("{:.{p}g}".format(spacing / 1e9, p=4)))
            folder_path = os.path.join(cond_dir, f"{nr_teeth} x {sp} GHz")
            os.makedirs(folder_path, exist_ok=True)

        fitting_results = []

        for i in range(nr_simulations_per_config):
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
                wavelength_step=wl_step,
            )

            fitting_results.append(f.concentration)
            if progress_q is None:  # verbose per-fit output only in the debug CLI
                print(
                    f"    fit {i + 1}/{nr_simulations_per_config}: "
                    f"{f.concentration:.6f} VMR"
                )

            if detail_writer is not None:
                detail_writer.writerow((f.concentration,))

            if not generate_plots or (i + 1) % plot_every != 0:
                continue

            _plot_fit(
                folder_path,
                i,
                f,
                x_meas,
                y_meas,
                molecule,
                pressure,
                temperature,
                length,
                vmr,
            )

        if detail_file is not None:
            detail_file.close()

        fitting_results = array(fitting_results)
        mean, std = float(fitting_results.mean()), float(fitting_results.std())
        rows.append((nr_teeth, spacing, mean, std))
        _emit(
            progress_q,
            (cid, "result", nr_teeth, spacing, mean, std, nr_simulations_per_config),
        )

    if simulator is not None and getattr(simulator, "use_gpu", False):
        simulator.exit_gpu()

    return rows


def _plot_fit(
    folder_path: str,
    i: int,
    f,
    x_meas,
    y_meas,
    molecule: str,
    pressure: float,
    temperature: float,
    length: float,
    vmr: float,
) -> None:
    """Save a per-fit plot of the simulated, pre-fit, and fitted transmission spectra."""
    meas = f.measured_transmission
    sim = f.simulated_transmission
    x_sim, y_sim = sim.x_nm, sim.y_nm
    x_meas_fitted, y_meas_fitted = meas.x_nm, meas.y_nm

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
    plt.savefig(os.path.join(folder_path, f"fit-simulated-measurement-{i}.svg"))
    plt.clf()
