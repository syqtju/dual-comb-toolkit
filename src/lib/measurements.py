import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    import matplotlib.pyplot as plt
    from numpy import ndarray

    from lib.entities import Baseline, MeasuredSpectrum


class LVMReader:
    """Reads a LabView Measurement file (.lvm) and extracts the data.

    Parameters
    ----------
    filename : str
        The name of the file to read.

    Other Parameters
    ----------------
    acq_freq : float, optional
        The acquisition frequency of the measurement. If given, it will be used instead of the one
        found in the file.
    default_acq_freq : float, optional
        The default acquisition frequency to use if not found in the file and not given as an
        argument.
    """

    def __init__(self, filename: str, **kwargs) -> None:
        import os

        from lib.defaults import ACQ_FREQ

        script_dir = os.path.dirname(__file__)

        self.filename: str = filename
        self.acq_freq: float | None = kwargs.get("acq_freq", None)
        self.data: dict | None = None

        self._full_path: str = os.path.join(
            script_dir, f"../../measurements/{self.filename}"
        )
        self._default_acq_freq: float = kwargs.get("default_acq_freq", ACQ_FREQ)

    def _read(self) -> list[str]:
        with open(self._full_path, "r") as f:
            file_lines = f.readlines()

        if not self.acq_freq:
            try:
                self.acq_freq = float(file_lines[0].split("=")[0].strip())
            except ValueError:
                pass

        if not self.acq_freq:
            warnings.warn(
                f"Acquisition frequency not found in file {self.filename}. Must be specified in "
                + "the first line as `f=<value in Hz>`. Taking default value of "
                + f"{self._default_acq_freq} Hz.",
                RuntimeWarning,
            )
            self.acq_freq = self._default_acq_freq
            return file_lines

        return file_lines[1::]

    def _clean_data(self) -> "ndarray":
        data = self._read()
        data = [d.replace(",", ".") for d in data]
        return [float(d.strip()) for d in data]

    def extract_data(self) -> "dict[str, ndarray]":
        """
        Extract the data from the file.

        Returns
        -------
        dict[str, ndarray]
            The data extracted from the file. The keys are 'time' and 'amplitude'.
        """
        import numpy as np

        amplitude = np.array(self._clean_data())
        time = np.linspace(0, len(amplitude) / self.acq_freq, len(amplitude))

        self.data = {"time": time, "amplitude": amplitude}
        return self.data


def read_lvm(filename: str, **kwargs) -> "tuple[ndarray, ndarray]":
    """
    Read a LabView Measurement file (.lvm) and return the time series data.

    Parameters
    ----------
    filename : str
        The name of the file to read.

    Keyword Args
    ------------
    acq_freq : float, optional
        The acquisition frequency of the measurement.

    Returns
    -------
    t : ndarray
        The time array.
    amplitude : ndarray
        The amplitude array.
    """
    acq_freq = kwargs.get("acq_freq", None)
    reader = LVMReader(filename, acq_freq=acq_freq)
    data = reader.extract_data()
    return data["time"], data["amplitude"]


def read_sample_and_reference_lvm(
    sample_filename: str, reference_filename: str, **kwargs: dict
) -> "tuple[ndarray, ndarray, ndarray]":
    """
    Read the sample and reference files and return the time series data.

    Parameters
    ----------
    sample_filename : str
        The name of the sample file.
    reference_filename : str
        The name of the reference file.

    Keyword Args
    ------------
    acq_freq : float, optional
        The acquisition frequency of the measurement.

    Returns
    -------
    t : ndarray
        The time array.
    sample_amplitude : ndarray
        The sample amplitude array.
    reference_amplitude : ndarray
        The reference amplitude array.
    """
    acq_freq = kwargs.get("acq_freq", None)
    sample_t, sample_amplitude = read_lvm(sample_filename, acq_freq=acq_freq)
    reference_t, reference_amplitude = read_lvm(reference_filename, acq_freq=acq_freq)

    # Check that time arrays are the same in both time series
    if not (sample_t == reference_t).all():
        raise ValueError("Time arrays are not the same in both time series")

    return sample_t, sample_amplitude, reference_amplitude


def compute_transmission(
    t: "ndarray",
    sample_signal: "ndarray",
    reference_signal: "ndarray",
    baseline: "Optional[Baseline]" = None,
    normalize: bool = True,
    **kwargs,
) -> "MeasuredSpectrum":
    """
    Compute the transmission spectrum from the sample and reference signals.

    Parameters
    ----------
    t : ndarray
        The time array.
    sample_signal : ndarray
        The sample signal amplitude.
    reference_signal : ndarray
        The reference signal amplitude.
    baseline : Baseline, optional
        The baseline to correct the measurement. If not given, no correction is applied.
    normalize : bool, optional
        Whether to normalize the transmission spectrum or not. Default is True.
    **kwargs : dict
        Additional keyword arguments to pass to the TransmissionAnalyser.

    Returns
    -------
    MeasuredSpectrum
        The transmission spectrum as a MeasuredSpectrum object.
    """

    from lib.analysis import TransmissionAnalyser
    from lib.combs import normalize_transmission
    from lib.entities import MeasuredSpectrum

    ta = TransmissionAnalyser(t, sample_signal, reference_signal, **kwargs)
    tr_freq, tr_amp = ta.transmission_freq, ta.transmission_amp

    if baseline:
        tr_freq, tr_amp = baseline.correct_transmission(tr_freq, tr_amp)

    if normalize:
        tr_freq, tr_amp = normalize_transmission(
            tr_freq, tr_amp, replace_outliers=False
        )

    return MeasuredSpectrum(tr_freq, tr_amp, xu="Hz")


class Measurement:
    """Represents a measurement of a gas sample and a reference sample.

    Parameters
    ----------
    measurement_name : str
        The name of the measurement.

    Keyword Arguments
    -----------------
    center_freq : float, optional
        The center frequency of the comb.
    freq_spacing : float, optional
        The frequency spacing of the comb.
    number_of_teeth : int, optional
        The number of teeth in the comb.
    laser_wavelength : float, optional
        The wavelength of the laser.
    optical_comb_spacing : float, optional
        Spacing between optical comb teeth.

    Other Parameters
    ----------------
    acq_freq : float, optional
        The acquisition frequency of the measurement.
    normalize : bool, optional
        Whether to normalize or not the transmission spectrum.
    sample_tag : str, optional
        The tag of the sample file.
    reference_tag : str, optional
        The tag of the reference file.
    baseline: list[str], optional
        The baseline to correct the measurement.
    molecule : str, optional
        The molecule measured.
    pressure : float, optional
        The pressure in Pa.
    temperature : float, optional
        The temperature in K.
    length : float, optional
        The length of the absorption path in m.
    concentration : float, optional
        The concentration of the molecule in VMR.
    sub_measurements : int, optional
        Number of sub-measurements used to obtain the standard deviation of the teeth. If not
        specified, tooth filtering based on standard deviation will not be applied.
    tooth_std_threshold : float, optional
        Teeth with a standard deviation above `tooth_std_threshold * mean_std` will be discarded if
        `sub_measurements` is given and is greater than 1.
    flip : bool, optional
        Whether to flip the measured transmission spectrum or not. Default is False.
    """

    time_series_properties = ["t", "sample_amplitude", "reference_amplitude"]
    transmission_properties = ["transmission_freq", "transmission_amp"]

    def __init__(self, measurement_name: str, **kwargs) -> None:
        from lib.defaults import (
            CENTER_FREQ,
            FREQ_SPACING,
            LASER_WAVELENGTH,
            NUMBER_OF_TEETH,
            OPTICAL_COMB_SPACING,
            REFERENCE_TAG,
            SAMPLE_TAG,
            TOOTH_STD_THRESHOLD,
        )

        # Measurement files
        self.measurement_name = measurement_name
        self._sample_tag = kwargs.get("sample_tag", SAMPLE_TAG)
        self._reference_tag = kwargs.get("reference_tag", REFERENCE_TAG)
        self.sample_filename = f"{self.measurement_name}-{self._sample_tag}.lvm"
        self.reference_filename = f"{self.measurement_name}-{self._reference_tag}.lvm"
        self.acq_freq = kwargs.get("acq_freq", None)

        # Time series data
        self._t = None
        self._sample_amplitude = None
        self._reference_amplitude = None

        # Transmission data
        self._transmission_freq: "Optional[ndarray]" = None
        self._transmission_amp: "Optional[ndarray]" = None
        self._transmission_spectrum: "Optional[MeasuredSpectrum]" = None

        # Transmission analysis parameters
        self.center_freq: float = kwargs.get("center_freq", CENTER_FREQ)
        self.freq_spacing: float = kwargs.get("freq_spacing", FREQ_SPACING)
        self.number_of_teeth: int = kwargs.get("number_of_teeth", NUMBER_OF_TEETH)
        self.laser_wavelength: float = kwargs.get("laser_wavelength", LASER_WAVELENGTH)
        self.optical_comb_spacing: float = kwargs.get(
            "optical_comb_spacing", OPTICAL_COMB_SPACING
        )
        self.normalize: bool = kwargs.get("normalize", True)
        self.baseline: "Optional[Baseline]" = kwargs.get("baseline", None)
        self.flip: bool = kwargs.get("flip", False)

        # Noise filtering parameters
        self.sub_measurements: int = kwargs.get("sub_measurements", 1)
        self.tooth_std_threshold: float = kwargs.get(
            "tooth_std_threshold", TOOTH_STD_THRESHOLD
        )
        if self.tooth_std_threshold is None:
            self.tooth_std_threshold = TOOTH_STD_THRESHOLD

        # Other parameters
        self.molecule: str = kwargs.get("molecule", None)
        self.pressure: float = kwargs.get("pressure", None)
        self.temperature: float = kwargs.get("temperature", None)
        self.length: float = kwargs.get("length", None)
        self.concentration: float = kwargs.get("concentration", None)

    @property
    def measurement_time(self) -> float:
        """Total time of the measurement in seconds."""
        return self.t[-1] - self.t[0]

    @property
    def kwargs(self) -> dict:
        return {
            "center_freq": self.center_freq,
            "freq_spacing": self.freq_spacing,
            "number_of_teeth": self.number_of_teeth,
            "laser_wavelength": self.laser_wavelength,
            "optical_comb_spacing": self.optical_comb_spacing,
            "flip": self.flip,
        }

    def get(self, property_name: str) -> "ndarray":
        property_name = property_name.lower().replace(" ", "_").strip("_ ")
        if (
            property_name
            not in self.time_series_properties + self.transmission_properties
        ):
            raise ValueError(f"Property {property_name} not found.")

        if getattr(self, f"_{property_name}") is None:
            if property_name in self.time_series_properties:
                self._t, self._sample_amplitude, self._reference_amplitude = (
                    read_sample_and_reference_lvm(
                        self.sample_filename,
                        self.reference_filename,
                        acq_freq=self.acq_freq,
                    )
                )
            else:
                self._compute_transmission()
        return getattr(self, f"_{property_name}")

    # Time series properties

    @property
    def t(self) -> "ndarray":
        """Time array of the sample and reference signals."""
        return self.get("t")

    @property
    def sample_signal(self) -> "ndarray":
        """Amplitude of the sample signal."""
        return self.get("sample_amplitude")

    @property
    def reference_signal(self) -> "ndarray":
        """Amplitude of the reference signal."""
        return self.get("reference_amplitude")

    # Transmission properties

    @property
    def transmission_spectrum(self) -> "MeasuredSpectrum":
        if self._transmission_spectrum is None:
            self._compute_transmission()
        return self._transmission_spectrum

    def remove_teeth(self, teeth_indices: list[int], normalize: bool = True) -> None:
        """
        Remove teeth from the optical comb.

        Parameters
        ----------
        teeth_indices : list[int]
            The indices of the teeth to remove. Indices assume nm units for the x values.
        normalize : bool, optional
            Whether to normalize the transmission spectrum after removing the teeth. Default is True.
        """
        import numpy as np

        from lib.combs import normalize_transmission, to_frequency

        if self._transmission_spectrum is None:
            raise ValueError(
                "Transmission spectrum must be computed before removing a tooth."
            )

        if not teeth_indices:
            return

        teeth_indices = [ti - 1 for ti in teeth_indices]  # Convert to 0-indexed

        def index_in_range(tooth_index: int) -> bool:
            return 0 <= tooth_index < len(self._transmission_spectrum.x_nm)

        teeth_indices = [ti for ti in teeth_indices if index_in_range(ti)]

        x_nm = np.delete(self._transmission_spectrum.x_nm, teeth_indices)
        y_nm = np.delete(self._transmission_spectrum.y_nm, teeth_indices)
        y_sdv_nm = None
        if self._transmission_spectrum.y_sdv_nm is not None:
            y_sdv_nm = np.delete(self._transmission_spectrum.y_sdv_nm, teeth_indices)

        if normalize:
            _, y_nm_norm = normalize_transmission(x_nm, y_nm, replace_outliers=False)
            scaling_factor = y_nm_norm.mean() / y_nm.mean()

            y_nm = y_nm * scaling_factor
            if y_sdv_nm is not None:
                y_sdv_nm = y_sdv_nm * scaling_factor

        self.transmission_spectrum.x_nm = x_nm
        self.transmission_spectrum.y_nm = y_nm
        self.transmission_spectrum.y_sdv_nm = y_sdv_nm

        self._transmission_spectrum.x_hz, self._transmission_spectrum.y_hz = (
            to_frequency(x_nm, y_nm)
        )
        if y_sdv_nm is not None:
            _, self.transmission_spectrum.y_sdv_hz = to_frequency(x_nm, y_sdv_nm)

    def _apply_metadata(
        self, transmission_spectrum: "Optional[MeasuredSpectrum]" = None
    ) -> None:
        """
        Set the metadata for the transmission spectrum.
        This method is called after computing the transmission spectrum.
        """
        if transmission_spectrum is None:
            transmission_spectrum = self.transmission_spectrum

        transmission_spectrum.meas_name = self.measurement_name
        transmission_spectrum.center_freq = self.center_freq
        transmission_spectrum.freq_spacing = self.freq_spacing
        transmission_spectrum.number_of_teeth = self.number_of_teeth
        transmission_spectrum.laser_wavelength = self.laser_wavelength
        transmission_spectrum.optical_comb_spacing = self.optical_comb_spacing
        transmission_spectrum.acq_freq = self.acq_freq
        transmission_spectrum.molecule = self.molecule
        transmission_spectrum.pressure = self.pressure
        transmission_spectrum.temperature = self.temperature
        transmission_spectrum.length = self.length
        transmission_spectrum.concentration = self.concentration

    def compute_transmission(
        self,
        start: "Optional[float]" = None,
        end: "Optional[float]" = None,
        save: bool = False,
    ) -> "MeasuredSpectrum":
        """
        Compute the transmission spectrum for a specific range of the time series.

        Parameters
        ----------
        start : float
            The start time of the range to compute the transmission spectrum for.
        end : float
            The end time of the range to compute the transmission spectrum for.
        save : bool, optional
            Whether to save the computed transmission spectrum to the object's property. Default is False.

        Returns
        -------
        MeasuredSpectrum
            The transmission spectrum for the specified range.
        """
        if start is not None and end is not None and start >= end:
            raise ValueError("Start time must be less than end time.")

        start_idx = 0 if start is None else self.t[self.t <= start].argmax()
        end_idx = len(self.t) if end is None else self.t[self.t <= end].argmax() + 1

        transmission_spectrum = compute_transmission(
            self.t[start_idx:end_idx],
            self.sample_signal[start_idx:end_idx],
            self.reference_signal[start_idx:end_idx],
            baseline=self.baseline,
            normalize=self.normalize,
            **self.kwargs,
        )

        if save:
            self._transmission_spectrum = transmission_spectrum

        self._apply_metadata(transmission_spectrum)

        return transmission_spectrum

    def _valid_sub_measurements(self) -> bool:
        """
        Check if the sub_measurements parameter is valid.
        """
        return isinstance(self.sub_measurements, int) and self.sub_measurements > 1

    sub_measurements_message = "`sub_measurements` must be an integer greater than 1 to compute sub-measurement intervals."

    @property
    def _sub_measurement_intervals(self) -> "list[tuple[float, float]]":
        """
        Compute the intervals for sub-measurements based on the total time and number of sub-measurements.
        This method divides the total time into equal intervals for sub-measurements.
        """
        if not self._valid_sub_measurements():
            raise ValueError(self.sub_measurements_message)

        sub_measurement_length = (self.t[-1] - self.t[0]) / self.sub_measurements
        return [
            (i * sub_measurement_length, (i + 1) * sub_measurement_length)
            for i in range(self.sub_measurements)
        ]

    def _compute_transmission(self) -> None:
        """
        Compute the transmission spectrum from the sample and reference signals.
        """
        import numpy as np

        from lib.combs import normalize_transmission
        from lib.entities import MeasuredSpectrum

        transmission_spectrum = self.compute_transmission()

        if self._valid_sub_measurements():
            sub_measurements = []
            intervals = self._sub_measurement_intervals

            for start, end in intervals:
                sub_measurement = self.compute_transmission(start=start, end=end)
                sub_measurements.append(sub_measurement.y_nm)

            stacked = np.stack(sub_measurements)
            tooth_std = stacked.std(axis=0) / len(sub_measurements) ** 0.5

            mean_std = np.mean(tooth_std)
            if mean_std == 0:
                mean_std = np.inf

            mask = tooth_std <= mean_std * self.tooth_std_threshold

            if all(~mask):
                raise ValueError(
                    f"No teeth with standard deviation below {self.tooth_std_threshold} * mean_std "
                    + "found. Please adjust the `tooth_std_threshold` parameter."
                )

            x_nm = transmission_spectrum.x_nm[mask]
            y_nm = transmission_spectrum.y_nm[mask]

            if self.normalize:
                x_nm, y_nm = normalize_transmission(x_nm, y_nm, replace_outliers=False)

            transmission_spectrum = MeasuredSpectrum(
                x=x_nm, y=y_nm, xu="nm", y_sdv=tooth_std[mask]
            )

        self._transmission_spectrum = transmission_spectrum
        self._apply_metadata()

    # Plots

    def generate_time_series_plot(self) -> "plt":
        import matplotlib.pyplot as plt

        plt.plot(self.t, self.sample_signal, label="Sample")
        plt.plot(self.t, self.reference_signal, label="Reference")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title("Time Series")
        return plt

    def show_time_series_plot(self) -> None:
        self.generate_time_series_plot().show()

    def generate_transmission_plot(self) -> "plt":
        from matplotlib import pyplot as plt

        if self._valid_sub_measurements():
            x = self.transmission_spectrum.x_nm
            y = self.transmission_spectrum.y_nm
            yerr = self.transmission_spectrum.y_sdv_nm / 2
            plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=3)

        self.transmission_spectrum.generate_plot(xu="nm")

        return plt

    def show_transmission_plot(self) -> None:
        self.generate_transmission_plot().show()
