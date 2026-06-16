"""Single entry point for the simulated-fittings pipeline.

Reads one Python config file (global defaults + a list of physical ``CONDITIONS``), then
simulates and fits every comb configuration for every condition in parallel, producing per
condition: a results report, two summary plots, and (if enabled) detail reports and
per-fit plots.

Usage
-----
    python configs--simulated-fittings-run.py [config_file]

``config_file`` defaults to ``configurations/default_config.py``.
"""

import sys
from pathlib import Path

from lib.files import get_configurations_path
from lib.simulated_fittings.orchestrator import run

DEFAULT_CONFIG = Path(get_configurations_path()) / "default_config.py"


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    run(config_path)


if __name__ == "__main__":
    main()
