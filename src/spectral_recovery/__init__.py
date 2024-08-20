"""
The spectral_recovery module includes the core functions for reading imagery,
processing indices, and computing recovery metrics.

"""

from spectral_recovery.io.raster import read_timeseries
from spectral_recovery.io.polygon import read_restoration_sites
from spectral_recovery.targets import historic, reference
from spectral_recovery.indices import compute_indices
from spectral_recovery.metrics import compute_metrics
from spectral_recovery.plotting import plot_spectral_trajectory
