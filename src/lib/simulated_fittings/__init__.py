"""Parallel orchestration of simulated dual-comb fittings over many physical conditions.

A single Python config file declares global default parameters plus a list of physical
``CONDITIONS`` (each identified by an ``id``). The orchestrator sweeps every comb
configuration for every condition in parallel and produces, per condition, a results
report, summary plots, and (optionally) detail reports and per-fit plots.

See ``configs--simulated-fittings-run.py`` for the entry point.
"""

from lib.simulated_fittings.config import RunConfig, load_config, resolve_conditions

__all__ = ["RunConfig", "load_config", "resolve_conditions"]
