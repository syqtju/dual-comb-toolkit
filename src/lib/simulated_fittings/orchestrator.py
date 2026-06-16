"""The orchestrator: one config file in, per-condition reports + plots out, all in parallel.

It reads the resolved :class:`RunConfig`, fans every condition's comb sweep across a process
pool (split into shards), drives a live ``rich`` dashboard fed by a worker progress queue,
aggregates each condition's results into its own report, and renders the per-condition summary
plots.
"""

import csv
import logging
import multiprocessing
import shutil
import threading
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from time import perf_counter, strftime
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lib.files import get_simulations_path
from lib.simulated_fittings.config import RunConfig, load_config
from lib.simulated_fittings.summary_plots import make_summary_plots
from lib.simulated_fittings.worker import REPORT_HEADER, init_worker, run_shard

logger = logging.getLogger("simulated_fittings")

_SENTINEL = ("__stop__",)


def run(config_path: str | Path) -> Path:
    """Run all conditions from ``config_path`` and return the run's reports folder.

    Parameters
    ----------
    config_path : str | Path
        Path to the single Python config file.

    Returns
    -------
    Path
        The timestamped run folder under ``simulations/``.
    """
    cfg = load_config(config_path)

    run_id = strftime("run-%Y%m%d-%H%M%S")
    run_dir = Path(get_simulations_path()) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(cfg.source_path, run_dir / "config.snapshot.py")
    _setup_logging(run_dir / "run.log")

    console = Console()
    console.rule(f"[bold]Simulated fittings · {run_id}")
    console.print(
        f"{len(cfg.conditions)} condition(s): {', '.join(cfg.ids)}  ·  "
        f"max_workers={cfg.max_workers}, shards/condition={cfg.shards_per_condition}"
    )
    logger.info("Run %s started: conditions=%s", run_id, cfg.ids)

    sweeps, work = _build_work(cfg, run_dir)

    counters: dict[str, dict[str, int]] = {
        cid: {"performed": 0, "skipped": 0, "fits": 0} for cid in cfg.ids
    }

    started = perf_counter()
    interrupted = False
    manager = multiprocessing.Manager()
    progress_q = manager.Queue()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} cfg"),
        TextColumn("[green]{task.fields[fits]} fits"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        tasks = {
            cid: progress.add_task(cid, total=len(sweeps[cid]), fits=0)
            for cid in cfg.ids
        }
        drain = threading.Thread(
            target=_drain_events,
            args=(progress_q, progress, tasks, counters),
            daemon=True,
        )
        drain.start()

        ctx = multiprocessing.get_context("spawn")
        ex = ProcessPoolExecutor(
            max_workers=cfg.max_workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(progress_q,),
        )
        try:
            futures = {
                ex.submit(run_shard, cond, shard, str(cdir), progress_q): cond["id"]
                for cond, shard, cdir in work
            }
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    rows = fut.result()
                except CancelledError:
                    continue
                except Exception as exc:  # surface a worker crash, keep going
                    logger.exception("[%s] shard failed: %s", cid, exc)
                    continue
                _append_rows(run_dir / cid / "results.csv", rows)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning(
                "Run interrupted by user (Ctrl+C); cancelling remaining shards."
            )
            console.print(
                "\n[yellow]Interrupted — cancelling remaining shards. "
                "Completed shards are already saved."
            )
        except BrokenProcessPool:
            interrupted = True
            logger.warning("Worker pool terminated; stopping run.")
            console.print(
                "\n[yellow]Worker pool terminated — stopping. Completed shards are saved."
            )
        finally:
            # Don't wait on in-flight shards: cancel what's pending and return promptly.
            ex.shutdown(wait=False, cancel_futures=True)
            try:
                progress_q.put(_SENTINEL)
            except Exception:  # manager may already be gone after an interrupt
                pass
            drain.join(timeout=10)

    elapsed = (perf_counter() - started) / 60
    _render_summary_plots(cfg, run_dir, console)
    _print_final_report(cfg, counters, run_dir, elapsed, console)

    state = "interrupted" if interrupted else "finished"
    if interrupted:
        console.print(
            "[yellow]Run stopped early; reports and plots reflect completed shards only."
        )
    logger.info("Run %s %s in %.2f min", run_id, state, elapsed)
    return run_dir


def _build_work(
    cfg: RunConfig, run_dir: Path
) -> tuple[dict[str, list], list[tuple[dict[str, Any], list, Path]]]:
    """Prepare per-condition folders + results-report headers and build the (cond, shard) units."""
    sweeps: dict[str, list] = {}
    work: list[tuple[dict[str, Any], list, Path]] = []

    for cond in cfg.conditions:
        cid = cond["id"]
        cdir = run_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        _init_csv(cdir / "results.csv", REPORT_HEADER)

        sweep = list(zip(cond["comb_spacings"], cond["numbers_of_teeth"]))
        sweeps[cid] = sweep

        for shard in _split(sweep, cfg.shards_per_condition):
            work.append((cond, shard, cdir))

    return sweeps, work


def _drain_events(progress_q, progress: Progress, tasks: dict, counters: dict) -> None:
    """Consume worker progress events: advance the dashboard and write the run log."""
    while True:
        event = progress_q.get()
        if event == _SENTINEL:
            return
        cid, kind = event[0], event[1]
        if kind == "result":
            _, _, nr_teeth, spacing, mean, std, nr_sims = event
            counters[cid]["performed"] += 1
            counters[cid]["fits"] += nr_sims
            progress.advance(tasks[cid])
            progress.update(tasks[cid], fits=counters[cid]["fits"])
            logger.info(
                "[%s] %2d teeth, %.2f GHz -> mean %.6f VMR, std %.6f VMR",
                cid,
                nr_teeth,
                spacing / 1e9,
                mean,
                std,
            )
        elif kind == "skip":
            _, _, nr_teeth, spacing = event
            counters[cid]["skipped"] += 1
            progress.advance(tasks[cid])
            logger.info(
                "[%s] skip %2d teeth, %.2f GHz (comb span out of range)",
                cid,
                nr_teeth,
                spacing / 1e9,
            )
        elif kind == "warn":
            logger.warning("[%s] %s", cid or "-", event[2])


def _render_summary_plots(cfg: RunConfig, run_dir: Path, console: Console) -> None:
    """Render the two stripped paper figures for each condition from its results report."""
    for cond in cfg.conditions:
        cid = cond["id"]
        cdir = run_dir / cid
        try:
            stems = make_summary_plots(
                str(cdir / "results.csv"),
                str(cdir),
                molecule=cond["molecule"],
                vmr_true=cond["vmr"],
            )
            if not stems:
                console.print(f"[yellow]· {cid}: no data to plot")
        except Exception as exc:
            logger.exception("[%s] summary plots failed: %s", cid, exc)
            console.print(f"[red]· {cid}: summary plots failed ({exc})")


def _print_final_report(
    cfg: RunConfig,
    counters: dict,
    run_dir: Path,
    elapsed_min: float,
    console: Console,
) -> None:
    """Print a per-condition summary of what was produced and where."""
    console.rule("[bold]Done")
    for cid in cfg.ids:
        c = counters[cid]
        console.print(
            f"[bold]{cid}[/bold]: {c['performed']} configs, {c['skipped']} skipped, "
            f"{c['fits']} fits"
        )
        console.print(f"    {run_dir / cid}/")
    console.print(f"\nRun log: {run_dir / 'run.log'}")
    console.print(f"Total wall time: {elapsed_min:.2f} min")


# Small IO / partitioning helpers ##################################################################


def _setup_logging(log_path: Path) -> None:
    """Configure the package logger to write the detailed run log to ``log_path`` only."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False


def _init_csv(path: Path, header: tuple[str, ...]) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def _append_rows(path: Path, rows: list[tuple]) -> None:
    if not rows:
        return
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def _split(seq: list, n: int) -> list[list]:
    """Split ``seq`` into at most ``n`` roughly equal, non-empty chunks."""
    if not seq:
        return []
    n = max(1, min(n, len(seq)))
    k, m = divmod(len(seq), n)
    chunks, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(seq[start : start + size])
        start += size
    return [c for c in chunks if c]
