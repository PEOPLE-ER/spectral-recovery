"""Configuration for the spectral recovery package.""" ""

import re
import json
import spyndex as spx

DATETIME_FREQ = "YS"
REQ_DIMS = ["band", "time", "y", "x"]
VALID_YEAR = re.compile(r"^\d{4}$")
# Index configurations
STANDARD_BANDS = list(spx.bands)
SUPPORTED_DOMAINS = ["vegetation", "burn"]
SUPPORTED_INDICES = [ix for ix in list(spx.indices) if spx.indices[ix].application_domain in SUPPORTED_DOMAINS] + ["GCI", "TCW", "TCG"]

