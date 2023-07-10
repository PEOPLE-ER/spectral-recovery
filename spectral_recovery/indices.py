import functools
from enum import Enum
from utils import maintain_spatial_attrs
from enums import Index, BandCommon
import xarray as xr


@maintain_spatial_attrs
def ndvi(stack: xr.DataArray):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    ndvi = (nir - red) / (nir + red)
    return ndvi


@maintain_spatial_attrs
def nbr(stack):
    nir = stack.sel(band=BandCommon.nir)
    swir2 = stack.sel(band=BandCommon.swir2)
    nbr = (nir - swir2) / (nir + swir2)
    return nbr

indices_map = {
    Index.ndvi: ndvi,
    Index.nbr: nbr,
}