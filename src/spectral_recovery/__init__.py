"""
The spectral_recovery module includes the core functions for reading imagery,
processing indices, and computing recovery metrics.

"""

from spectral_recovery.io.raster import read_and_stack_tifs
from spectral_recovery.indices import compute_indices
from spectral_recovery.restoration import RestorationArea

import spyndex as spx

# make 'green' and 'rededge' common names unambiguous
# need this here until resolution of: https://github.com/awesome-spectral-indices/awesome-spectral-indices/issues/42
spx.bands["G1"].common_name = "green1"
spx.bands["RE1"].common_name = "rededge1"
spx.bands["RE2"].common_name = "rededge2"
spx.bands["RE3"].common_name = "rededge3"
