"""
The spectral_recovery module includes the core functions for reading imagery,
processing indices, and computing recovery metrics.

"""

from spectral_recovery.io.raster import read_timeseries
from spectral_recovery.io.polygon import read_restoration_polygons, read_reference_polygons
from spectral_recovery.restoration import compute_metrics
from spectral_recovery.plotting import plot_spectral_trajectory
