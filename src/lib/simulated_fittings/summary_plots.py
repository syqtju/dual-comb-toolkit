"""Per-condition summary plots: the paper-style configuration figures, stripped down."""

import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.plots import (
    article_dpi,
    article_figsize,
    article_text_size,
    get_cmap,
)


def molecule_label(molecule: str) -> str:
    """Format a molecule name as a LaTeX math label, e.g. ``H2O`` -> ``$\\mathrm{H_2O}$``."""
    subscripted = re.sub(r"(\d+)", r"_{\1}", molecule)
    return rf"$\mathrm{{{subscripted}}}$"


def _read_results(csv_path: str) -> dict[int, dict[str, list[float]]]:
    """Read a results report into ``{n_teeth: {spacings, concentrations, sdvs}}``."""
    data: dict[int, dict[str, list[float]]] = {}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            n_teeth = int(row[0])
            spacing, concentration, sdv = float(row[1]), float(row[2]), float(row[3])
            d = data.setdefault(
                n_teeth, {"spacings": [], "concentrations": [], "sdvs": []}
            )
            d["spacings"].append(spacing / 1e9)
            d["concentrations"].append(concentration)
            d["sdvs"].append(sdv)

    # Sort each teeth series by spacing so the connecting lines are monotonic.
    for d in data.values():
        order = sorted(range(len(d["spacings"])), key=lambda i: d["spacings"][i])
        for key in ("spacings", "concentrations", "sdvs"):
            d[key] = [d[key][i] for i in order]

    return data


def _teeth_legend_label(n_teeth: int, max_teeth: int) -> str:
    """Group teeth into 5-wide bins for the legend (e.g. '20-24 teeth')."""
    if n_teeth % 5 != 0:
        return ""
    if n_teeth >= max_teeth:
        return f"{n_teeth} teeth"
    return f"{n_teeth}-{n_teeth + 4} teeth"


def _value_at(d: dict[str, list[float]], spacing_ghz: float, series_key: str) -> "float | None":
    """Value of ``series_key`` at a comb spacing, exact or linearly interpolated.

    Returns ``None`` if the spacing is outside the available range.
    """
    spacings, ys = d["spacings"], d[series_key]
    for s, y in zip(spacings, ys):
        if abs(s - spacing_ghz) < 1e-9:  # exact (within float tolerance)
            return y
    below = [s for s in spacings if s < spacing_ghz]
    above = [s for s in spacings if s > spacing_ghz]
    if not below or not above:
        return None
    s_lo, s_hi = max(below), min(above)
    y_lo, y_hi = ys[spacings.index(s_lo)], ys[spacings.index(s_hi)]
    return y_lo + (y_hi - y_lo) * (spacing_ghz - s_lo) / (s_hi - s_lo)


def _summary_figure(
    data: dict[int, dict[str, list[float]]],
    series_key: str,
    ylabel: str,
    out_stem: str,
    reference: "float | None" = None,
    legend_loc: str = "best",
    highlight: "list[tuple[int, float]] | None" = None,
) -> None:
    """Render one summary figure (main axes only) and save it as svg + pdf + png.

    ``highlight`` is an optional list of ``(number_of_teeth, comb_spacing_GHz)`` configurations
    to mark on the figure.
    """
    cmap = get_cmap("brg", len(data))
    max_teeth = max(data) if data else 0

    fig, ax = plt.subplots(figsize=article_figsize, dpi=article_dpi)
    fig.tight_layout(**{"pad": 0.1, "rect": (0.15, 0.1, 1, 1)})

    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", zorder=5, linewidth=1)

    for i, n_teeth in enumerate(sorted(data)):
        d = data[n_teeth]
        kwargs = dict(color=cmap(i))
        label = _teeth_legend_label(n_teeth, max_teeth)
        if label:
            kwargs["label"] = label
        ax.plot(d["spacings"], d[series_key], "o-", **kwargs)

    # Mark the selected configurations.
    for n_teeth, spacing_ghz in highlight or ():
        d = data.get(n_teeth)
        value = _value_at(d, spacing_ghz, series_key) if d else None
        if value is None:
            continue
        ax.plot(
            spacing_ghz, value, "o", markersize=9, mfc=(1, 1, 1, 0.6), mec="black", zorder=6
        )
        ax.plot(spacing_ghz, value, "x", color="black", zorder=7)

    ax.set_ylabel(ylabel, fontdict={"size": article_text_size})
    ax.set_xlabel("Comb Spacing [GHz]", fontdict={"size": article_text_size})
    ax.tick_params(axis="both", which="major", labelsize=article_text_size)
    if any(_teeth_legend_label(t, max_teeth) for t in data):
        legend = ax.legend(
            loc=legend_loc,
            prop={"size": 10},
            frameon=True,
            facecolor="white",
            framealpha=0.8,  # translucent white panel so it stands out over the data
            edgecolor="none",
        )
        legend.set_zorder(10)  # keep the legend above the plotted series

    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}")
    plt.close(fig)


def make_summary_plots(
    csv_path: str,
    out_dir: str,
    molecule: str,
    vmr_true: float,
    highlight: "list[tuple[int, float]] | None" = None,
    stem_prefix: str = "summary",
) -> list[str]:
    """Produce the two per-condition summary figures from a results report.

    Parameters
    ----------
    csv_path : str
        Path to the condition's ``results.csv``.
    out_dir : str
        Directory to write the figures into.
    molecule : str
        Molecule name, used to build the concentration y-axis label.
    vmr_true : float
        The condition's true VMR, drawn as a dashed reference line on the concentration plot.
    highlight : list[tuple[int, float]], optional
        ``(number_of_teeth, comb_spacing_GHz)`` configurations to mark on both figures.
    stem_prefix : str, optional
        Filename stem prefix; figures are ``<prefix>-conc`` and ``<prefix>-sdv``. Defaults to
        ``"summary"``.

    Returns
    -------
    list[str]
        The output figure stems that were written.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    data = _read_results(csv_path)
    if not data:
        return []

    conc_stem = os.path.join(out_dir, f"{stem_prefix}-conc")
    sdv_stem = os.path.join(out_dir, f"{stem_prefix}-sdv")

    _summary_figure(
        data,
        series_key="concentrations",
        ylabel=f"{molecule_label(molecule)} Concentration [VMR]",
        out_stem=conc_stem,
        reference=vmr_true,
        legend_loc="lower right",
        highlight=highlight,
    )
    _summary_figure(
        data,
        series_key="sdvs",
        ylabel="Standard Deviation [VMR]",
        out_stem=sdv_stem,
        reference=None,
        legend_loc="upper right",
        highlight=highlight,
    )

    return [conc_stem, sdv_stem]
