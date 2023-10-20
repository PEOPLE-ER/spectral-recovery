import functools
import xarray as xr

from typing import List
from pandas import Index as pdIndex

from spectral_recovery.utils import maintain_rio_attrs
from spectral_recovery.enums import Index, BandCommon, Platform


def compatible_with(platform: List[Platform]):
    def comptaible_with_decorator(func):
        """A wrapper for assigning platform compatibility to a function."""

        @functools.wraps(func)
        def comptaible_with_wrapper(*args, **kwargs):
            if kwargs["stack"].attrs["platform"] not in platform:
                raise ValueError(
                    f"Function {func.__name__} is not compatible with platform"
                    f" {kwargs['stack'].attrs['platform']}"
                )
            return func(*args, **kwargs)
        return comptaible_with_wrapper
    return comptaible_with_decorator


def requires_bands(bands: List[BandCommon]):
    def requires_bands_decorator(func):
        """A wrapper for requiring bands in a function."""

        @functools.wraps(func)
        def requires_bands_wrapper(*args, **kwargs):
            for band in kwargs["stack"]["band"]:
                if band not in bands:
                    raise ValueError(
                        f"Function {func.__name__} requires bands {bands} but"
                        f" image_stack only contains {kwargs['stack']['band']}"
                    )
            return func(*args, **kwargs)
        return requires_bands_wrapper
    return requires_bands_decorator


@compatible_with(
    [
        Platform.landsat,
        Platform.landsat_oli,
        Platform.landsat_tm,
        Platform.landsat_etm,
        Platform.sentinel_2,
    ]
)
@requires_bands([BandCommon.nir, BandCommon.red])
@maintain_rio_attrs
def ndvi(stack: xr.DataArray):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    ndvi = (nir - red) / (nir + red)
    return ndvi


@maintain_rio_attrs
def nbr(stack):
    nir = stack.sel(band=BandCommon.nir)
    swir2 = stack.sel(band=BandCommon.swir2)
    nbr = (nir - swir2) / (nir + swir2)
    return nbr


@maintain_rio_attrs
def gndvi(stack):
    nir = stack.sel(band=BandCommon.nir)
    green = stack.sel(band=BandCommon.green)
    gndvi = (nir - green) / (nir + green)
    return gndvi


@maintain_rio_attrs
def evi(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    blue = stack.sel(band=BandCommon.blue)
    evi = 2.5 * ((nir - red)) / (nir + 6.0 * red - 7.5 * blue + 1)
    return evi


@maintain_rio_attrs
def avi(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    avi = (nir * (1 - red) * (nir - red)) ** (1 / 3)
    return avi


@maintain_rio_attrs
def savi(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    savi = ((nir - red) / (nir + red + 0.5)) * 0.5
    return savi


@maintain_rio_attrs
def ndwi(stack):
    green = stack.sel(band=BandCommon.green)
    nir = stack.sel(band=BandCommon.nir)
    ndwi = (green - nir) / (green + nir)
    return ndwi


@maintain_rio_attrs
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


@maintain_rio_attrs
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


@maintain_rio_attrs
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


@maintain_rio_attrs
def sr(stack):
    nir = stack.sel(band=BandCommon.nir)
    red = stack.sel(band=BandCommon.red)
    sr = nir / red
    return sr


@maintain_rio_attrs
def ndmi(stack):
    nir = stack.sel(band=BandCommon.nir)
    swir1 = stack.sel(band=BandCommon.swir1)
    ndmi = (nir - swir1) / (nir + swir1)
    return ndmi


@maintain_rio_attrs
def gci(stack):
    nir = stack.sel(band=BandCommon.nir)
    green = stack.sel(band=BandCommon.green)
    gci = (nir / green) - 1
    return gci


@maintain_rio_attrs
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
