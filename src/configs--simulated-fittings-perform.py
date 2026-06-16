"""Debug CLI: run ONE physical condition's full comb sweep in the foreground.

This is the single-condition counterpart to ``configs--simulated-fittings-run.py``. It uses the
same config file and the same worker (``run_shard``) but runs synchronously, in this process,
with verbose per-fit output — handy for debugging one condition. For full, parallel,
multi-condition runs use the orchestrator (``configs--simulated-fittings-run.py``) instead.

Usage
-----
    python configs--simulated-fittings-perform.py [--config <path>] [--condition <id>]

``--config`` defaults to ``configurations/default_config.py``; with no ``--condition`` the
first condition in the config file is run.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter, strftime

from lib.files import get_configurations_path, get_simulations_path
from lib.simulated_fittings.config import load_config
from lib.simulated_fittings.summary_plots import make_summary_plots
from lib.simulated_fittings.worker import REPORT_HEADER, run_shard

DEFAULT_CONFIG = Path(get_configurations_path()) / "default_config.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single physical condition's comb sweep in the foreground."
    )
    parser.add_argument(
        "-c", "--config", default=str(DEFAULT_CONFIG), help="Path to the config file."
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="Id of the condition to run. Defaults to the first condition in the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.condition is None:
        cond = cfg.conditions[0]
    else:
        matches = [c for c in cfg.conditions if c["id"] == args.condition]
        if not matches:
            raise SystemExit(
                f"Condition {args.condition!r} not found. Available: {cfg.ids}"
            )
        cond = matches[0]

    cid = cond["id"]
    run_id = strftime("debug-%Y%m%d-%H%M%S")
    cond_dir = Path(get_simulations_path()) / run_id / cid
    cond_dir.mkdir(parents=True, exist_ok=True)

    sweep = list(zip(cond["comb_spacings"], cond["numbers_of_teeth"]))
    print(f"Running condition {cid!r}: {len(sweep)} configurations.\n")

    started = perf_counter()
    rows = run_shard(cond, sweep, str(cond_dir), progress_q=None)

    results = cond_dir / "results.csv"
    with open(results, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REPORT_HEADER)
        writer.writerows(rows)

    make_summary_plots(
        str(results), str(cond_dir), molecule=cond["molecule"], vmr_true=cond["vmr"]
    )

    print(f"\nDone in {(perf_counter() - started) / 60:.2f} min.")
    print(f"Output: {cond_dir}/")


if __name__ == "__main__":
    main()
