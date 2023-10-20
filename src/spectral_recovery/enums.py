from enum import Enum


class BandCommon(Enum):
    blue = "BLUE"
    green = "GREEN"
    red = "RED"
    nir = "NIR"
    swir1 = "SWIR1"
    swir2 = "SWIR2"

    def __str__(self) -> str:
        return self.value


class Index(Enum):
    ndvi = "NDVI"
    nbr = "NBR"
    gndvi = "GNDVI"
    evi = "EVI"
    avi = "AVI"
    savi = "SAVI"
    ndwi = "NDWI"
    tcg = "TCG"
    tcw = "TCW"
    tcb = "TCB"
    sr = "SR"
    ndmi = "NDMI"
    gci = "GCI"
    ndii = "NDII"

    def __str__(self) -> str:
        return self.value


class Metric(Enum):
    percent_recovered = "percent_recovered"
    Y2R = "Y2R"
    RRI = "RRI"
    dNBR = "dNBR"
    R80P = "R80P"
    YrYr = "YrYr"

    def __str__(self) -> str:
        return self.value


class Platform(Enum):
    landsat_etm = 2
    landsat_tm = 3
    landsat_oli = 4
    landsat = 5  # for a mix of oli/tm/etm+ landsat platforms
    sentinel_2 = 6
