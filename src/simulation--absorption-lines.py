from lib.simulations import Simulator
from lib.plots import use_latex

# Define the molecule, VMR, pressure, temperature, and length.

molecule = "CH4"
database = "HITEMP"
vmr = 0.01  # volume mixing ratio
pressure = 101325  # Pa
temperature = 1200  # K
length = 0.07  # m

# Simulation configuration.

wl_min = 3390  # nm
wl_max = 3440  # nm
wl_step = (wl_max - wl_min) / 10000  # nm
use_gpu = True

# Latex for plotting.

latex = False
if latex:
    use_latex()

# Simulate the transmission spectrum.

s = Simulator(
    molecule=molecule,
    vmr=vmr,
    pressure=pressure,
    temperature=temperature,
    length=length,
    use_gpu=use_gpu,
    wavelength_step=wl_step,
    database=database,
)
s.compute_transmission_spectrum(wl_min=wl_min, wl_max=wl_max)
s.show_transmission_spectrum()
