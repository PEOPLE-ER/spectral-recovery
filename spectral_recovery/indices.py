import functools
from enum import Enum
from spectral_recovery.utils import maintain_spatial_attrs
from spectral_recovery.enums import Index, BandCommon
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


@maintain_spatial_attrs
def gndvi(stack):
    nir = stack.sel(band=BandCommon.nir)
    green = stack.sel(band=BandCommon.green)
    gndvi = (nir - green) / (nir + green)
    return gndvi


@maintain_spatial_attrs
def evi(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    blue = stack.sel(band=BandCommon.blue)
    evi = 2.5 * ((nir - red)) / (nir + 6.0 * red - 7.5 * blue + 1)
    return evi


@maintain_spatial_attrs
def avi(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    avi = (nir * (1 - red) * (nir - red)) ** (1 / 3)
    return avi


@maintain_spatial_attrs
def savi(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    savi = ((nir - red) / (nir + red + 0.5)) * 0.5
    return savi


@maintain_spatial_attrs
def ndwi(stack):
    green = stack.sel(band=BandCommon.green)
    nir = stack.sel(band=BandCommon.nir)
    ndwi = (green - nir) / (green + nir)
    return ndwi


@maintain_spatial_attrs
def tcg(stack):
    blue = stack.sel(band=BandCommon.blue)
    green = stack.sel(band=BandCommon.green)
    red = stack.sel(band=BandCommon.red)
    nir = stack.sel(band=BandCommon.nir)
    swir1 = stack.sel(band=BandCommon.swir1)
    swir2 = stack.sel(band=BandCommon.swir2)
    tcg = (
        0.2043 * blue
        + 0.4158 * green
        + 0.5524 * red
        + 0.5741 * nir
        + 0.3124 * swir1
        + 0.2303 * swir2
    )
    return tcg


@maintain_spatial_attrs
def tcw(stack):
    blue = stack.sel(band=BandCommon.blue)
    green = stack.sel(band=BandCommon.green)
    red = stack.sel(band=BandCommon.red)
    nir = stack.sel(band=BandCommon.nir)
    swir1 = stack.sel(band=BandCommon.swir1)
    swir2 = stack.sel(band=BandCommon.swir2)
    tcw = (
        0.1509 * blue
        + 0.1973 * green
        + 0.3279 * red
        + 0.3406 * nir
        + 0.7112 * swir1
        + 0.4572 * swir2
    )
    return tcw


@maintain_spatial_attrs
def tcb(stack):
    blue = stack.sel(band=BandCommon.blue)
    green = stack.sel(band=BandCommon.green)
    red = stack.sel(band=BandCommon.red)
    nir = stack.sel(band=BandCommon.nir)
    swir1 = stack.sel(band=BandCommon.swir1)
    swir2 = stack.sel(band=BandCommon.swir2)
    tcb = (
        0.3037 * blue
        + 0.2793 * green
        + 0.4743 * red
        + 0.5585 * nir
        + 0.5082 * swir1
        + 0.1863 * swir2
    )
    return tcb


@maintain_spatial_attrs
def sr(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    sr = nir / red
    return sr


@maintain_spatial_attrs
def ndmi(stack):
    nir = stack.sel(band=BandCommon.nir)
    swir1 = stack.sel(band=BandCommon.swir1)
    ndmi = (nir - swir1) / (nir + swir1)
    return ndmi


@maintain_spatial_attrs
def gci(stack):
    nir = stack.sel(band=BandCommon.nir)
    green = stack.sel(band=BandCommon.green)
    gci = (nir / green) - 1
    return gci


@maintain_spatial_attrs
def ndii(stack):
    swir1 = stack.sel(band=BandCommon.swir1)
    nir = stack.sel(band=BandCommon.nir)
    ndii = (swir1 - nir) / (swir1 + nir)
    return ndii


indices_map = {
    Index.ndvi: ndvi,
    Index.nbr: nbr,
    Index.gndvi: gndvi,
    Index.evi: evi,
    Index.avi: avi,
    Index.savi: savi,
    Index.ndwi: ndwi,
    Index.tcg: tcg,
    Index.tcw: tcw,
    Index.tcb: tcb,
    Index.sr: sr,
    Index.ndmi: ndmi,
    Index.gci: gci,
    Index.ndii: ndii,
}
