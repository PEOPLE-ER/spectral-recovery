"""Enums for supported spectral indices, band common names, and platforms.

Primarily intended for internal use, but may be useful for users to access.
Enums are used to ensure consistent naming and avoid typos for internal 
computations/processing.

"""

from enum import Enum

class Metric(Enum):
    """Supported recovery metric names/acroynms"""

    Y2R = "Y2R"
    RRI = "RRI"
    DNBR = "dNBR"
    R80P = "R80P"
    YRYR = "YrYr"

    def __str__(self) -> str:
        return self.value