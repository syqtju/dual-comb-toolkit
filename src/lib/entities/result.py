from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from lib.fitting.interpolation import closest_value_indices
from lib.plots import article_tight

if TYPE_CHECKING:
    from lib.entities.spectrum import MeasuredSpectrum, SimulatedSpectrum


class Result:
    def __init__(
        self,
        measured_spectrum: "MeasuredSpectrum",
        simulated_spectrum: "SimulatedSpectrum",
    ) -> None:
        self.measured_spectrum = measured_spectrum
        self.simulated_spectrum = simulated_spectrum

    @property
    def concentration(self) -> float:
        """
        Get the concentration of the molecule.
        """
        return self.simulated_spectrum.concentration

    vmr = concentration

    @property
    def molecule(self) -> str:
        """
        Get the name of the molecule.
        """
        return self.simulated_spectrum.molecule

    @property
    def pressure(self) -> float:
        """
        Get the pressure in Pa.
        """
        return self.simulated_spectrum.pressure

    @property
    def temperature(self) -> float:
        """
        Get the temperature in K.
        """
        return self.simulated_spectrum.temperature

    @property
    def length(self) -> float:
        """
        Get the length of the absorption path in m.
        """
        return self.simulated_spectrum.length

    @cached_property
    def residual(self) -> "np.ndarray":
        """
        Return the percentage residual between the measured and simulated spectra. The x axis is in
        nm.
        """
        measured_y = self.measured_spectrum.y_nm
        simulated_y = self.simulated_spectrum.y_nm
        measured_x = self.measured_spectrum.x_nm
        simulated_x = self.simulated_spectrum.x_nm

        idcs = closest_value_indices(simulated_x, measured_x)
        simulated_y_com = simulated_y[idcs]

        # return self.measured_spectrum.x_nm, measured_y - simulated_y_com
        return self.measured_spectrum.x_nm, (
            measured_y - simulated_y_com
        ) / simulated_y_com * 100

    def generate_plot(self, show_residual: bool = False) -> "plt":
        """
        Generate a plot of the measured and simulated transmission spectra.

        Parameters
        ----------
        show_residual : bool, optional
            If True, show the residual plot below the main plot, by default False.

        Returns
        -------
        plt : matplotlib.pyplot
            The plot object containing the measured and simulated spectra.
        """
        if not show_residual:
            plt.plot(
                self.simulated_spectrum.x_nm,
                self.simulated_spectrum.y_nm,
                label=f"Simulation for {self.concentration:.3f} VMR",
                color="blue",
                zorder=0,
            )
            plt.scatter(
                self.measured_spectrum.x_nm,
                self.measured_spectrum.y_nm,
                label="Measurement",
                color="red",
                zorder=1,
            )

            if self.measured_spectrum.y_sdv_nm is not None:
                x = self.measured_spectrum.x_nm
                y = self.measured_spectrum.y_nm
                yerr = self.measured_spectrum.y_sdv_nm / 2
                plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, color="red")

            plt.legend()
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Transmission")
            plt.title(
                f"Measured and Simulated Transmission spectrum of {self.molecule}\nat {self.pressure:.2f} Pa "
                + f"and {self.temperature:.2f} K. {self.length:.3f} m path length, {self.concentration:.3f} VMR "
            )
            plt.tight_layout(**article_tight)

            return plt

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax1.set_title(
            f"Measured and Simulated Transmission spectrum of {self.molecule}\nat {self.pressure:.2f} Pa "
            + f"and {self.temperature:.2f} K. {self.length:.3f} m path length, {self.concentration:.3f} VMR "
        )

        ax1.plot(
            self.simulated_spectrum.x_nm,
            self.simulated_spectrum.y_nm,
            label=f"Simulation for {self.concentration:.3f} VMR",
            color="blue",
            zorder=0,
        )
        ax1.scatter(
            self.measured_spectrum.x_nm,
            self.measured_spectrum.y_nm,
            label="Data",
            color="red",
            zorder=1,
        )

        x, y = self.residual
        ax2.plot(
            x,
            [0 for _ in x],
            "--",
            color="black",
            zorder=0,
        )
        ax2.plot(
            x,
            y,
            color="red",
            zorder=0,
        )
        ax2.set_xlim(min(x), max(x))

        ax1.set_ylabel("Transmission [a.u.]")
        ax1.tick_params(axis="both", which="major")

        if plt.rcParams["text.usetex"]:
            ax2.set_ylabel("Res. [\%]")
        else:
            ax2.set_ylabel("Res. [%]")
        ax2.set_xlabel("Wavelength [nm]")
        ax2.tick_params(axis="both", which="major")

        legend = ax1.legend(frameon=True)
        legend.set_zorder(1001)
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.7)

        return plt

    def show_plot(self, show_residual=False) -> None:
        """
        Show the plot of the measured and simulated transmission spectra.
        """
        return self.generate_plot(show_residual=show_residual).show()