# Dual-Comb Toolkit: A Python-based library for dual-comb spectroscopy data processing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18040669.svg)](https://doi.org/10.5281/zenodo.18040669)

Useful scripts for working with dual-comb spectroscopy (DCS), including:
- Fast simulation of absorption spectra using the [Radis](https://github.com/radis/radis) library,
  compatible with GPU.
- Processing of dual-comb sample/reference measurements (baseline correction, normalization, 
  RF-to-optical mapping, noisy tooth filtering, etc).
- Fitting of processed dual-comb spectra to retrieve gas concentration.
- Mapping of concentration across multiple measurement positions to create 2D maps.
- Analysis of temporal concentration evolution within measurements. 
- Simulation of real, noisy measurements, for optimization of experimental configurations.

## Setup

Before starting, if you are going to be using the HITRAN/HITEMP databases, create a Hitran
login [here](https://hitran.org/login/). Other databases may require their own credentials.

Once you have your credentials, you can proceed with the setup below, depending on your
operating system.

<details>
<summary><b>Linux</b></summary>

1. Clone the repository:

   ```bash
   git clone https://github.com/bfrangi/dual-comb-toolkit.git
   cd dual-comb-toolkit
   ```

2. Install `virtualenv` and create a virtual environment:

   ```bash
   sudo apt-get update
   sudo apt-get install python3-venv
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies and apply the radis patch (this second part is optional):

   ```bash
   pip install -r requirements.txt
   ```

4. [Optional] If you want to use LaTeX for plotting, install the `texlive` package:

   ```bash
   sudo apt-get update
   sudo apt-get install texlive-latex-extra texlive-fonts-extra dvipng
   ```

</details>

<details>
<summary><b>Windows</b></summary>

1. Clone the repository:

   ```bash
   git clone https://github.com/bfrangi/dual-comb-toolkit.git
   cd dual-comb-toolkit
   ```

2. Install `python3.12` from the Microsoft Store.

3. Create a virtual environment:

   ```bash
   python3.12 -m venv .venv
   .\.venv\Scripts\activate.bat
   ```

4. Install dependencies and apply the radis patch (this second part is optional):

   ```bash
   pip install -r requirements-win.txt
   ```

5. [Optional] If you want to use LaTeX for plotting, install the `texlive` package by downloading
   the installer from [here](https://www.tug.org/texlive/windows.html) or the ISO from 
   [here](https://www.tug.org/texlive/acquire-iso.html) and following the installation
   instructions.

</details>

<br>

**Note**: Using LaTeX for plotting is optional, and slows down the plotting process significantly.
It is recommended only for generating publication-quality figures.

## Usage

If using HITRAN/HITEMP, you will need to log in with your HITRAN credentials the first time you
simulate an absorption spectrum. Other databases may require their own credentials.

**Note**: The first time you run the simulator, the database will be downloaded, and this could
take some time. Please be patient!

<details>
<summary><b>Line simulation</b></summary>

You can simulate absorption spectra using the `src/simulation--absorption-lines.py` script. You can
define the simulation parameters directly in the script. Navigate to the `src` directory and run the
script as follows:

```bash
python simulation--absorption-lines.py
```

</details>

<details>
<summary><b>Manual measurement processing</b></summary>

You can manually process dual-comb measurements using the `src/processing--measurement-manual.py`
script. You can define the measurement parameters directly in the script, including the path to the
measurement to process and the path to the set of measurements to use for baseline correction. This
last part is useful to remove etalon effects and other systematic noise from the measurement.
Navigate to the `src` directory and run the script as follows:

```bash
python processing--measurement-manual.py
```

Here is an example of the output plot:

![Example processed plot](assets/process-measurement-example.svg)

To characterize the baseline, there is also the `src/processing--baseline.py` script. You can run it
as follows:

```bash
python processing--baseline.py
```

And there is also a simple script to view the raw spectrum of any measurement,
`src/processing--spectrum.py`:

```bash
python processing--spectrum.py
```

An example of the output plot from this script is shown below:

![Spectrum](assets/measurement-spectrum-example.svg)

</details>

<details>
<summary><b>Concentration fitting</b></summary>

You can fit processed dual-comb measurements to retrieve gas concentration using the
`src/processing--measurement-fit-concentration.py` script. You can define the fitting parameters
directly in the script, very similarly to the `src/processing--measurement-manual.py` script.
Navigate to the `src` directory and run the script as follows:

```bash
python processing--measurement-fit-concentration.py
```

Here is an example of the output plot:

![Example fit plot](assets/fit-measurement-example.svg)

</details>

<details>
<summary><b>2D Concentration mapping</b></summary>

You can map concentration across multiple measurement positions using the
`src/processing--measurement-fit-concentration-map.py` script. You can define the mapping parameters
directly in the script. To ensure correct mapping, the measurements should be named according to
their position, e.g., `Position-X1-Y1`, `Position-X2-Y5`, etc. Navigate to the `src` directory and
run the script as follows:

```bash
python processing--measurement-fit-concentration-map.py
```

A report is generated in the `reports/` directory, containing the concentration map and configuration
details. Here is an example:

```txt
Mapping Report
---------------------------------------
Generated on 2025-12-19 17:47:44

Molecule: CH4
Pressure: 53328.94736842 Pa
Temperature: 298 K
Path length: 0.055 m

Optical Comb spacing: 0.5 GHz
Number of teeth: 38
Laser wavelength: 3427.4500000000003 m
Minimum wavelength: 3427.0 nm
Maximum wavelength: 3427.9 nm

RF central frequency: 40000.0 Hz
RF comb spacing: 200.0 Hz
Acquisition frequency: 400000.0 Hz

Fitter: normal_gpu
Initial guess: 0.0001 VMR
Lower bound: 0 VMR
Upper bound: 1 VMR
Number of sub-measurements: 10
Tooth standard deviation threshold: inf
Removed tooth indices: [9, 12, 13]
Baseline measurements used: 6
 - cell-sweep-10-34-17-03-2025/Position-X11-Y1
 - cell-sweep-10-34-17-03-2025/Position-X12-Y1
 - cell-sweep-10-34-17-03-2025/Position-X13-Y1
 - cell-sweep-10-34-17-03-2025/Position-X14-Y1
 - cell-sweep-10-34-17-03-2025/Position-X15-Y1
 - cell-sweep-10-34-17-03-2025/Position-X16-Y1
---------------------------------------

cell-sweep-10-34-17-03-2025/Position-X1-Y1	0.009261 VMR
cell-sweep-10-34-17-03-2025/Position-X2-Y1	0.009597 VMR
cell-sweep-10-34-17-03-2025/Position-X3-Y1	0.009233 VMR
cell-sweep-10-34-17-03-2025/Position-X4-Y1	0.009390 VMR
cell-sweep-10-34-17-03-2025/Position-X5-Y1	0.009983 VMR
cell-sweep-10-34-17-03-2025/Position-X6-Y1	0.010781 VMR
cell-sweep-10-34-17-03-2025/Position-X7-Y1	0.000000 VMR
cell-sweep-10-34-17-03-2025/Position-X8-Y1	0.000000 VMR
cell-sweep-10-34-17-03-2025/Position-X9-Y1	0.000000 VMR
cell-sweep-10-34-17-03-2025/Position-X10-Y1	0.002576 VMR
cell-sweep-10-34-17-03-2025/Position-X11-Y1	0.000494 VMR
cell-sweep-10-34-17-03-2025/Position-X12-Y1	0.000385 VMR
cell-sweep-10-34-17-03-2025/Position-X13-Y1	0.000090 VMR
cell-sweep-10-34-17-03-2025/Position-X14-Y1	0.000162 VMR
cell-sweep-10-34-17-03-2025/Position-X15-Y1	0.000228 VMR
cell-sweep-10-34-17-03-2025/Position-X16-Y1	0.000409 VMR

End of report.
---------------------------------------
```

</details>

<details>
<summary><b>Time evolution of concentration</b></summary>

You can analyze the temporal evolution of concentration within measurements using the
`src/processing--measurement-fit-concentration-evolution.py` script. You can define the analysis
parameters directly in the script. Navigate to the `src` directory and run the script as follows:

```bash
python processing--measurement-fit-concentration-evolution.py
```

Here is an example of the output `.csv` file and animation:

```csv
Time [s],Concentration [VMR]
0.000000000000000000e+00,6.840625000000015964e-03
2.500000000000000139e-02,7.136250000000015845e-03
5.000000000000000278e-02,7.959375000000018519e-03
7.500000000000001110e-02,8.620000000000020007e-03
1.000000000000000056e-01,7.585000000000017076e-03
1.250000000000000000e-01,7.333125000000017581e-03
1.500000000000000222e-01,7.904375000000015555e-03
1.750000000000000167e-01,9.492500000000021754e-03
...
```

![Evolution animation](assets/evolution-example.gif)

</details>

<details>
<summary><b>Simulation of measurements</b></summary>

You can simulate realistic, noisy dual-comb measurements using the `src/simulation--measurement.py`
script. You can define the simulation parameters, including the amount of noise and the noise model,
directly in the script. Navigate to the `src` directory and run the script as follows:

```bash
python simulation--measurement.py
```

Here is an example of the output plot:

![Example simulated measurement plot](assets/simulated-measurement-example.svg)

</details>

<details>
<summary><b>Optimization of comb parameters</b></summary>

The `src/configs--simulated-fittings-perform.py` script allows you to generate synthetic dual-comb
measurements with varying comb parameters (number of teeth and comb spacing) and fit them to
retrieve gas concentration. This helps in optimizing the comb parameters for specific experimental
setups, by studying their effect on the fitting accuracy. You can define the optimization parameters 
directly in the script. Navigate to the `src` directory and run the script as follows:

```bash
python configs--simulated-fittings-perform.py
```

The script generates a report in the `reports/` directory, containing the fitting results for each set of
comb parameters. Here is an example of the report:

```txt
Number of teeth,Comb spacing (Hz),Mean concentration (VMR),Standard deviation (VMR)
5,1000000000.0,0.008118531250000026,0.0015218513420581644
5,1100000000.0,0.008479265625000027,0.0014387855767247013
5,1200000000.0,0.007970718750000027,0.0015506988937514755
5,1300000000.0,0.008278750000000027,0.001380888514102033
5,1400000000.0,0.008655304687500026,0.0006965584456108906
5,1500000000.0,0.00885082812500003,0.0008205821120921131
5,1600000000.0,0.008879328125000028,0.0007775362153379832
5,1700000000.0,0.008766578125000028,0.0010174894999794189
5,1800000000.0,0.00907967187500003,0.000766652188654768
5,1900000000.0,0.008935507812500031,0.0011732676952238023
5,2000000000.0,0.00903296093750003,0.0009908021498187501
5,2100000000.0,0.00914619531250003,0.0013056479607271966
5,2200000000.0,0.008427351562500027,0.0013634225145963083
5,2300000000.0,0.008553406250000029,0.0013969829923649918
5,2400000000.0,0.008159531250000025,0.001660422555846814
5,2500000000.0,0.007942726562500026,0.0019770452989126648
5,2600000000.0,0.007925281250000023,0.0019076125334136248
5,2700000000.0,0.007906195312500025,0.001968696081401424
...
```

The script `src/configs--simulated-fittings-plot.py` can be used to generate plots from the report
data, to help visualize the optimization results and choose the best comb parameters. Run it as
follows (make sure you change the path to the report file in the script if needed):

```bash
python configs--simulated-fittings-plot.py
```

Here is an example of the output plots:

![Example optimization plots concentration](assets/conf-conc-example.svg) 

![Example optimization plots std](assets/conf-sdv-example.svg)

</details>

## Using GPU Acceleration

Check out the `GPU_DEVICE_ID` setting in `src/lib/defaults.py`. By default, it is set to `"nvidia"`
which will use the NVIDIA GPU if available. If you want to use a different GPU, run the
`src/identify-gpu.py` script to list available GPUs and set the `GPU_DEVICE_ID` accordingly (it can 
be the number of the GPU in the output list or a string contained in the name of the device).

## Issues with plots

If you encounter `UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown plt.show()`,
just install `PyQt6`:

```bash
pip install PyQt6
```

## Issues with LaTeX (on Linux)

If you encounter `FileNotFoundError: Matplotlib's TeX implementation searched for a file named 'cmr10.tfm' in your texmf tree, but could not find it.`, just run the [following command](https://stackoverflow.com/a/79243265/15159198):

```bash
sudo mv /usr/bin/luatex /usr/bin/luatex.bk 
```

If you encounter `LaTeX Error: File 'type1ec.sty' not found.`, install the `cm-super` package (see [here](https://github.com/matplotlib/matplotlib/issues/16911)):

```bash
sudo apt install cm-super
```

If you encounter `RuntimeError: Failed to process string with tex because dvipng could not be found`, install the `dvipng` package:

```bash
sudo apt install dvipng
```

## Cite this repository

```bib
@misc{bernat_frangi_2025_18040669,
  author       = {Frangi, Bernat},
  title        = {Dual-{C}omb {T}oolkit: {A} {P}ython-based library for dual-comb spectroscopy data processing},
  month        = 12,
  year         = 2025,
  howpublished = {{Z}enodo},
  doi          = {10.5281/zenodo.18040669},
}
```
## Acknowledgments

This project uses the [RADIS](https://github.com/radis/radis) library for
high-resolution molecular spectroscopy calculations.
