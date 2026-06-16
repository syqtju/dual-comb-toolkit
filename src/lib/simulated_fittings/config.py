"""Loading and resolution of the single input config file.

The input file is a plain Python module. Every module-level name listed in
``PARAM_NAMES`` is treated as a *global default*. Each entry of ``CONDITIONS`` is a dict
that may override any of those defaults; a resolved condition is simply
``{**globals, **condition}``. ``max_workers`` and ``shards_per_condition`` are run-level
settings and cannot be overridden per condition.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Per-condition parameters: read from the config module as global defaults and overridable
# by each condition. Anything absent falls back to the defaults in ``lib.defaults`` (applied by
# the worker); the required ones below are validated.
PARAM_NAMES: tuple[str, ...] = (
    # Molecule / database
    "molecule",
    "database",
    # Physical conditions
    "vmr",
    "pressure",
    "temperature",
    "length",
    "laser_wavelength",
    # Simulation range
    "wl_range",
    "wl_step",
    "min_comb_span",
    # Noise / error model
    "noise_distribution",
    "transmission_std",
    "nr_teeth_for_transmission_std",
    "tooth_std_threshold_start",
    "teeth_start",
    "tooth_std_threshold_end",
    "teeth_end",
    "spectrum_shift_range",
    "scaling_range",
    "modulation_intensities",
    # Plots / reports
    "generate_plots",
    "plot_every",
    "use_latex",
    "detailed_report",
    # Fitting
    "normalize",
    "initial_guess",
    "lower_bound",
    "upper_bound",
    "fitter",
    # Simulation cases
    "nr_simulations_per_config",
    "comb_spacings",
    "numbers_of_teeth",
)

# Parameters that must be present in every resolved condition (globally or via override).
REQUIRED_PARAMS: tuple[str, ...] = (
    "molecule",
    "vmr",
    "pressure",
    "temperature",
    "length",
    "laser_wavelength",
    "comb_spacings",
    "numbers_of_teeth",
    "nr_simulations_per_config",
)

DEFAULT_MAX_WORKERS = 16
DEFAULT_SHARDS_PER_CONDITION = 6


@dataclass
class RunConfig:
    """A fully resolved run: the source path, the resolved conditions, and run-level settings."""

    source_path: Path
    conditions: list[dict[str, Any]]
    max_workers: int = DEFAULT_MAX_WORKERS
    shards_per_condition: int = DEFAULT_SHARDS_PER_CONDITION
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return [c["id"] for c in self.conditions]


def load_config(path: str | Path) -> RunConfig:
    """Import the config module at ``path`` and resolve it into a :class:`RunConfig`.

    Parameters
    ----------
    path : str | Path
        Path to the Python config file.

    Returns
    -------
    RunConfig
        The resolved run configuration.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("simulation_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONDITIONS"):
        raise ValueError(
            f"Config file {path.name} must define a `CONDITIONS` list of dicts "
            "(each with at least an `id`)."
        )

    base = {name: getattr(module, name) for name in PARAM_NAMES if hasattr(module, name)}
    conditions = resolve_conditions(base, getattr(module, "CONDITIONS"))

    return RunConfig(
        source_path=path,
        conditions=conditions,
        max_workers=int(getattr(module, "max_workers", DEFAULT_MAX_WORKERS)),
        shards_per_condition=int(
            getattr(module, "shards_per_condition", DEFAULT_SHARDS_PER_CONDITION)
        ),
    )


def resolve_conditions(
    base: dict[str, Any], conditions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge global defaults with each condition's overrides and validate the result.

    Parameters
    ----------
    base : dict
        The global default parameters.
    conditions : list[dict]
        The per-condition override dicts. Each must contain a unique ``id``.

    Returns
    -------
    list[dict]
        The resolved conditions (``{**base, **condition}``), in input order.
    """
    if not conditions:
        raise ValueError("`CONDITIONS` is empty; specify at least one physical condition.")

    resolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for i, condition in enumerate(conditions):
        if "id" not in condition:
            raise ValueError(f"Condition #{i + 1} is missing a required `id` field.")
        cid = str(condition["id"])
        if cid in seen_ids:
            raise ValueError(f"Duplicate condition id: {cid!r}. Ids must be unique.")
        seen_ids.add(cid)

        merged = {**base, **condition}
        merged["id"] = cid

        missing = [p for p in REQUIRED_PARAMS if merged.get(p) is None]
        if missing:
            raise ValueError(
                f"Condition {cid!r} is missing required parameter(s): "
                f"{', '.join(missing)}. Provide them globally or in the condition."
            )

        resolved.append(merged)

    return resolved
