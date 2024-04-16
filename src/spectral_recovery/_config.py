"""Configuration for the spectral recovery package.""" ""

import re
import json
import spyndex as spx

DATETIME_FREQ = "YS"

REQ_DIMS = ["band", "time", "y", "x"]

STANDARD_BANDS = list(spx.bands)

VALID_YEAR = re.compile(r"^\d{4}$")