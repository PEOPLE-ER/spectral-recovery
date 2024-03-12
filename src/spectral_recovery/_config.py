"""Configuration for the spectral recovery package.""" ""

import re
import spyndex as spx

DATETIME_FREQ = "YS"

REQ_DIMS = ["band", "time", "y", "x"]

STANDARD_BANDS = list(spx.bands)

VALID_YEAR = re.compile(r"^\d{4}$")

SUPPORTED_DOMAINS = ["vegetation", "burn"]

SUPPORTED_PLATFORMS = []
for index in spx.indices:
    if spx.indices[index].application_domain in SUPPORTED_DOMAINS:
        for platform in spx.indices[index].platforms:
            if platform not in SUPPORTED_PLATFORMS:
                SUPPORTED_PLATFORMS.append(platform)
