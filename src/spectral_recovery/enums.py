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
    percent_recovered = 1
    Y2R = 2
    RI = 3
    dNBR = 4
    P80R = 5

    def __str__(self) -> str:
        return self.name
