"""
The spectral_recovery module includes the core functions for reading imagery,
processing indices, and computing recovery metrics.

"""
from spectral_recovery.io.raster import read_and_stack_tifs
from spectral_recovery.indices import compute_indices
from spectral_recovery.restoration import RestorationArea
