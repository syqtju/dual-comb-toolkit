from matplotlib import pyplot as plt

from lib.files import get_figures_path, read_csv_report
from lib.plots import (
    article_figsize,
    config_plot,
    get_cmap,
    toggle_series_plot,
    use_latex,
)

use_latex()

####################################################################################################
# Get data from the report                                                                         #
####################################################################################################

report_name = "report-20251025-123911"

csv_data = read_csv_report(report_name, mapping=["int", "float", "float", "float"])

data: dict[int, dict[str, list[float]]] = {}

for n_teeth, spacing, concentration, sdv in zip(*csv_data):
    if n_teeth not in data:
        data[n_teeth] = {
            "spacings": [],
            "concentrations": [],
            "sdvs": [],
            "bandwidths": [],
        }

    data[n_teeth]["spacings"].append(spacing / 1e9)
    data[n_teeth]["bandwidths"].append(spacing / 1e9 * n_teeth)
    data[n_teeth]["concentrations"].append(concentration)
    data[n_teeth]["sdvs"].append(sdv)

####################################################################################################
# Plot standard deviation and mean concentration vs comb spacing for each number of teeth          #
####################################################################################################


for n_teeth, d in data.items():
    fig, ax = config_plot(
        d["spacings"],
        d["sdvs"],
        title=f"Standard deviation of the concentration as a function\nof the comb spacing for {n_teeth} teeth",
        xlabel="Spacing [GHz]",
        ylabel="Standard deviation [VMR]",
        save_path=f"{get_figures_path()}conf-sdv-{n_teeth}.svg",
    )
    plt.close()

    fig, ax = config_plot(
        d["spacings"],
        d["concentrations"],
        title=f"Concentration as a function of the comb spacing for {n_teeth} teeth",
        xlabel="Spacing [GHz]",
        ylabel="Concentration [VMR]",
        save_path=f"{get_figures_path()}conf-conc-{n_teeth}.svg",
    )
    plt.close()


####################################################################################################
# Plot standard deviation and concentration vs comb spacing for all number of teeth                #
####################################################################################################

cmap = get_cmap("brg", len(data))

fig, ax = toggle_series_plot(
    [(d["spacings"], d["sdvs"], f"{n_teeth} teeth") for n_teeth, d in data.items()],
    title="",
    xlabel="Comb spacing [GHz]",
    ylabel="Standard deviation [VMR]",
    cmap=cmap,
    figsize=article_figsize,
    save_path=f"{get_figures_path()}conf-sdv.svg",
)

plt.clf()

fig, ax = toggle_series_plot(
    [
        (d["spacings"], d["concentrations"], f"{n_teeth} teeth")
        for n_teeth, d in data.items()
    ],
    title="",
    xlabel="Comb spacing [GHz]",
    ylabel="Fitted concentration [VMR]",
    cmap=cmap,
    figsize=article_figsize,
    save_path=f"{get_figures_path()}conf-conc.svg",
)

plt.clf()

####################################################################################################
# Plot standard deviation and concentration vs comb bandwidth for all number of teeth              #
####################################################################################################

fig, ax = toggle_series_plot(
    [(d["bandwidths"], d["sdvs"], f"{n_teeth} teeth") for n_teeth, d in data.items()],
    title="",
    xlabel="Bandwidth [GHz]",
    ylabel="Standard deviation [VMR]",
    cmap=cmap,
    figsize=article_figsize,
    save_path=f"{get_figures_path()}conf-sdv-bw.pdf",
)

plt.clf()

fig, ax = toggle_series_plot(
    [
        (d["bandwidths"], d["concentrations"], f"{n_teeth} teeth")
        for n_teeth, d in data.items()
    ],
    title="",
    xlabel="Bandwidth [GHz]",
    ylabel="Concentration [VMR]",
    cmap=cmap,
    figsize=article_figsize,
    save_path=f"{get_figures_path()}conf-conc-bw.pdf",
)

plt.clf()
