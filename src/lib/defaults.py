CENTER_FREQ: float = 40000.0
FREQ_SPACING: float = 200.0
NUMBER_OF_TEETH: int = 38
LASER_WAVELENGTH: float = 1645.56e-9
OPTICAL_COMB_SPACING: float = 1.0e9
ACQ_FREQ: float = 400000.0
SAMPLE_TAG: str = 'sample'
REFERENCE_TAG: str = 'reference'
TOOTH_STD_THRESHOLD: float = 0.1
DATABASE: str = 'hitran'
WAVELENGTH_STEP: float = 0.01
GPU_DEVICE_ID: str | int = 'nvidia'
ALLAN_NR_POINTS: int = 100
ALLAN_METHOD: str = 'adev'

# Simulated fittings (synthetic dual-comb measurement) defaults. Used by the worker as the
# fallback values when a config / condition does not specify them.
WL_RANGE: float = 0.15
MIN_COMB_SPAN: float = 0.03
NOISE_DISTRIBUTION: str = 'uniform'
TRANSMISSION_STD: float = 0.01
TOOTH_STD_THRESHOLD_START: float = 2.5
TEETH_START: int = 5
TOOTH_STD_THRESHOLD_END: float = 1.5
TEETH_END: int = 30
SPECTRUM_SHIFT_RANGE: tuple[float, float] = (0.0, 0.0)
SCALING_RANGE: tuple[float, float] = (1.0, 1.0)
GENERATE_PLOTS: bool = False
PLOT_EVERY: int = 1
DETAILED_REPORT: bool = False
NORMALIZE: bool = True
INITIAL_GUESS: float = 0.5
FITTER: str = 'normal'