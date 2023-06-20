import functools
from enum import Enum 
from utils import maintain_spatial_attrs

import xarray as xr

class Indices(Enum):
    NDVI = "NDVI"
    NBR = "NBR"

    def __str__(self) -> str:
        return self.name
    
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value.upper():
                return member

@maintain_spatial_attrs
def ndvi(stack: xr.DataArray):
    nir = stack.sel(band="nir") 
    red = stack.sel(band="red")
    ndvi = (nir - red) / (nir + red) 
    return ndvi

@maintain_spatial_attrs
def nbr(stack):
    nir = stack.sel(band="nir") 
    swir = stack.sel(band="swir")
    nbr = (nir - swir) / (nir + swir)
    return nbr
     
indices_map = {
    Indices.NDVI: ndvi,
    Indices.NBR: nbr,
}
