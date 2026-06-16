import csv
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.entities.result import Result


def get_root_path() -> str:
    """
    Get the root project path.

    Returns
    -------
    str
        The root project path.
    """
    script_dir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(script_dir, "../../"))


def get_figures_path() -> str:
    """
    Get the figures folder path.

    Returns
    -------
    str
        The figures folder path.
    """
    root_path = get_root_path()
    return os.path.join(root_path, "figures/")


def get_reports_path() -> str:
    """
    Get the reports folder path.

    Returns
    -------
    str
        The reports folder path.
    """
    root_path = get_root_path()
    return os.path.join(root_path, "reports/")


def get_simulations_path() -> str:
    """
    Get the simulations folder path (outputs of the simulated-fittings pipeline).

    Returns
    -------
    str
        The simulations folder path.
    """
    root_path = get_root_path()
    return os.path.join(root_path, "simulations/")


def get_configurations_path() -> str:
    """
    Get the configurations folder path (where simulation config `.py` files live).

    Returns
    -------
    str
        The configurations folder path.
    """
    root_path = get_root_path()
    return os.path.join(root_path, "configurations/")


def get_evolutions_path() -> str:
    """
    Get the evolutions folder path.

    Returns
    -------
    str
        The evolutions folder path.
    """
    root_path = get_root_path()
    return os.path.join(root_path, "evolutions/")


def get_measurement_paths(directory: str) -> list[str]:
    """
    Get the measurement paths from the directory.

    Parameters
    ----------
    directory : str
        The directory containing the measurements.

    Returns
    -------
    list[str]
        The full measurement paths.
    """
    root_path = get_root_path()
    directory = os.path.join(root_path, f"measurements/{directory}/")
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".lvm")
    ]


def get_measurement_filenames(directory: str) -> list[str]:
    """
    Get the measurement filenames from the directory.

    Parameters
    ----------
    directory : str
        The directory containing the measurements.

    Returns
    -------
    list[str]
        The measurement filenames.
    """
    root_path = get_root_path()
    directory = os.path.join(root_path, f"measurements/{directory}/")
    return [f for f in os.listdir(directory) if f.endswith(".lvm")]


def get_measurement_names(directory: str) -> list[str]:
    """
    Get the measurement names from the directory.

    Parameters
    ----------
    directory : str
        The directory containing the measurements.

    Returns
    -------
    list[str]
        The names of the measurements.
    """
    measurement_filenames = get_measurement_filenames(directory)
    return sorted(
        [
            f"{directory}/{f[:-14]}"
            for f in measurement_filenames
            if f.endswith("-reference.lvm")
        ]
    )


def initialize_csv_report(filename: str, headers: tuple[str]) -> None:
    """
    Initialize a CSV report file with the given headers.

    Parameters
    ----------
    filename : str
        The name of the CSV file to create.
    headers : list[str]
        The headers for the CSV file.
    """
    csv_path = f"{get_reports_path()}{filename}.csv"

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(headers)


def append_to_csv_report(filename: str, data: tuple) -> None:
    """
    Append data to an existing CSV report file.

    Parameters
    ----------
    filename : str
        The name of the CSV file to append data to.
    data : tuple
        The data to append to the CSV file.
    """
    csv_path = f"{get_reports_path()}{filename}.csv"

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(data)


def read_csv_report(filename: str, mapping: list[str]) -> list[list]:
    """
    Read a CSV report file and return each column as a list.

    Parameters
    ----------
    filename : str
        The name of the CSV file to read.
    mapping : list[str]
        A list of column types to map the data to.
        Possible values are "int", "float", and "str".

    Returns
    -------
    list[list]
        A list of lists, where each inner list contains the data from a column.
    """
    csv_path = f"{get_reports_path()}{filename}.csv"
    data = []

    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")

        # Skip the header row
        next(reader, None)

        # Read the rest of the rows
        for row in reader:
            if row:  # Skip empty rows
                data.append(row)

    data = list(map(list, zip(*data)))  # Transpose the data

    for i, col in enumerate(data):
        if mapping[i] == "int":
            data[i] = list(map(int, col))
        elif mapping[i] == "float":
            data[i] = list(map(float, col))
        elif mapping[i] == "str":
            data[i] = list(map(str, col))

    return data


def initialize_figures_folder(foldername) -> str:
    """
    Create a folder in the figures directory if it does not exist.

    Parameters
    ----------
    foldername : str
        The name of the folder to create in the figures directory.

    Returns
    -------
    str
        The path to the created folder.
    """
    folder_path = os.path.join(get_figures_path(), foldername)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def save_mapping_report(
    filename: str,
    data: dict[str, str | int | float, list["Result"]],
    verbose: bool = False,
) -> None:
    """
    Save a mapping report to a .txt file.

    Parameters
    ----------
    filename : str
        The name of the file to save the report.
    data : dict[str, str | int | float, list[Result]]
        A dictionary containing the report data. It should include:
        - "molecule": The name of the molecule.
        - "pressure": The pressure in Pa.
        - "temperature": The temperature in K.
        - "path_length": The path length in m.
        - "optical_comb_spacing": The optical comb spacing in GHz.
        - "number_of_teeth": The number of teeth in the comb.
        - "laser_wavelength": The laser wavelength in m.
        - "comb_spacing": The RF comb spacing in Hz.
        - "acquisition_frequency": The acquisition frequency in Hz.
        - "fitter": The fitter used for the mapping.
        - "initial_guess": The initial guess for the concentration in VMR.
        - "lower_bound": The lower bound for the concentration in VMR.
        - "upper_bound": The upper bound for the concentration in VMR.
        - "sub_measurements": The number of sub-measurements used to obtain the standard deviation of the teeth.
        - "tooth_std_threshold": The threshold for the standard deviation of the teeth.
        - "remove_teeth_indices": The list of tooth indices that were removed.
        - "wl_min": The minimum wavelength in nm.
        - "wl_max": The maximum wavelength in nm.
        - "center_frequency": The RF central frequency in Hz.
        - "baseline_names": The list of baseline measurement names used.
    - "results": A list of Result objects containing the measured spectra and their concentrations.
    verbose : bool, optional
        Whether to print additional information during the saving process. Defaults to False.
    """
    report_path = f"{get_reports_path()}{filename}.txt"

    optical_spacing = "{:g}".format(
        float("{:.{p}g}".format(data["optical_comb_spacing"], p=4))
    )
    metadata = [
        "Mapping Report",
        "---------------------------------------",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Molecule: {data['molecule']}",
        f"Pressure: {data['pressure']} Pa",
        f"Temperature: {data['temperature']} K",
        f"Path length: {data['path_length']} m\n",
        f"Optical Comb spacing: {optical_spacing} GHz",
        f"Number of teeth: {data['number_of_teeth']}",
        f"Laser wavelength: {data['laser_wavelength']} m",
        f"Minimum wavelength: {data['wl_min']} nm",
        f"Maximum wavelength: {data['wl_max']} nm\n",
        f"RF central frequency: {data['center_frequency']} Hz",
        f"RF comb spacing: {data['comb_spacing']} Hz",
        f"Acquisition frequency: {data['acquisition_frequency']} Hz\n",
        f"Fitter: {data['fitter']}",
        f"Initial guess: {data['initial_guess']} VMR",
        f"Lower bound: {data.get('lower_bound', 0)} VMR",
        f"Upper bound: {data.get('upper_bound', 1)} VMR",
        f"Number of sub-measurements: {data.get('sub_measurements', 0)}",
        f"Tooth standard deviation threshold: {data.get('tooth_std_threshold', float('inf'))}",
        f"Removed tooth indices: {data.get('remove_teeth_indices', [])}",
    ]

    if (baseline_names := data.get("baseline_names", [])):
        metadata.append(f"Baseline measurements used: {len(baseline_names)}")
        for name in baseline_names:
            metadata.append(f" - {name}")

    metadata.append(
        "---------------------------------------",
    )

    def get_y_position(result: "Result") -> float:
        """
        Get the y position for sorting results based on measured spectrum name.
        """
        return int(result.measured_spectrum.meas_name.split("Y")[-1])

    def get_x_position(result: "Result") -> float:
        """
        Get the x position for sorting results based on measured spectrum name.
        """
        return int(result.measured_spectrum.meas_name.split("-")[-2].strip("X"))

    data["results"].sort(key=get_y_position)
    data["results"].sort(key=get_x_position)

    with open(report_path, "w") as file:
        # Write the header
        file.write("\n".join(metadata) + "\n\n")

        # Write the data
        for res in data["results"]:
            name = res.measured_spectrum.meas_name
            conc = res.concentration
            file.write(f"{name}\t{conc:.6f} VMR\n")
        file.write("\n")
        file.write("End of report.\n")
        (file.write("---------------------------------------\n"),)

    if verbose:
        print(f"Mapping report saved to {report_path}.")


def get_animations_path() -> str:
    """
    Get the animations folder path.

    Returns
    -------
    str
        The animations folder path.
    """
    root_path = get_root_path()
    return os.path.join(root_path, "animations/")
