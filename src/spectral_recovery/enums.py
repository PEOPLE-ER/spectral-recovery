from enum import Enum


class BandCommon(Enum):
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
    Y2R = "Y2R"
    RRI = "RRI"
    DNBR = "dNBR"
    R80P = "R80P"
    YRYR = "YrYr"

    def __str__(self) -> str:
        return self.value


class Platform(Enum):
    LANDSAT_ETM = 2
    LANDSAT_TM = 3
    LANDSAT_OLI = 4
    SENTINEL_2 = 5
