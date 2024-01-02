"""Enums for supported spectral indices, band common names, and platforms.

Primarily intended for internal use, but may be useful for users to access.
Enums are used to ensure consistent naming and avoid typos for internal 
computations/processing.

"""
from enum import Enum


class BandCommon(Enum):
    """Common band names for Landsat and Sentinel-2"""

    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    NIR = "NIR"
    SWIR1 = "SWIR1"
    SWIR2 = "SWIR2"
    COASTAL_AEROSOL = "COASTAL_AEROSOL"
    RED_EDGE = "RED_EDGE"

    def __str__(self) -> str:
        return self.value


class Index(Enum):
    """Supported spectral index names/acroynms"""

    NDVI = "NDVI"
    NBR = "NBR"
    GNDVI = "GNDVI"
    EVI = "EVI"
    AVI = "AVI"
    SAVI = "SAVI"
    NDWI = "NDWI"
    TCG = "TCG"
    TCW = "TCW"
    TCB = "TCB"
    SR = "SR"
    NDMI = "NDMI"
    GCI = "GCI"
    NDII = "NDII"

    def __str__(self) -> str:
        return self.value


class Metric(Enum):
    """Supported recovery metric names/acroynms"""

    Y2R = "Y2R"
    RRI = "RRI"
    DNBR = "dNBR"
    R80P = "R80P"
    YRYR = "YrYr"

    def __str__(self) -> str:
        return self.value


class Platform(Enum):
    """Supported satellite platforms"""

    LANDSAT_ETM = 2
    LANDSAT_TM = 3
    LANDSAT_OLI = 4
    SENTINEL_2 = 5
