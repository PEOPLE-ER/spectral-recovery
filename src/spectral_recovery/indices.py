"""Methods for computing spectral indices.

Most functions are decorated with `compatible_with` and `requires_bands` decorators, 
which check that the input stack is compatible with the function and that the stack
contains the required bands.

Most notably, exports the `compute_indices` function, which computes a stack of
spectral indices from a stack of images and str of index names.

"""
import functools
from typing import List

import xarray as xr
from pandas import Index as pdIndex

from spectral_recovery._utils import maintain_rio_attrs
from spectral_recovery.enums import Index, BandCommon, Platform


def compatible_with(platform: List[Platform]):
    """A decorator for assigning platform compatibility to a function.
    
    Parameters
    ----------
    platform : List[Platform]
        List of platforms compatible with the function.

    """
    def comptaible_with_decorator(func):
        """Sub-decorator for assigning platform compatibility to a function."""

        @functools.wraps(func)
        def comptaible_with_wrapper(stack, *args, **kwargs):
            for input_platform in stack.attrs["platform"]:
                if input_platform not in platform:
                    raise ValueError(
                        f"Function {func.__name__} is not compatible with platform"
                        f" {stack.attrs['platform']}. Only compatible with {platform}"
                    ) from None
            return func(stack, *args, **kwargs)

        return comptaible_with_wrapper

    return comptaible_with_decorator


def requires_bands(bands: List[BandCommon]):
    """A decorator for assigning band requirements to a function.
    
    Parameters
    ----------
    bands : List[BandCommon]
        List of bands required by the function.

    """
    def requires_bands_decorator(func):
        """Sub-decorator for assigning band requirements to a function."""

        @functools.wraps(func)
        def requires_bands_wrapper(stack, *args, **kwargs):
            for band in bands:
                if band not in stack["band"].values:
                    raise ValueError(
                        f"Function {func.__name__} requires bands {bands} but"
                        f" image_stack only contains {stack['band'].values}"
                    ) from None
            return func(stack, *args, **kwargs)

        return requires_bands_wrapper

    return requires_bands_decorator


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def ndvi(stack: xr.DataArray):
    """Compute the Normalized Difference Vegetation Index (NDVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    ndvi_v = (nir - red) / (nir + red)
    return ndvi_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.SWIR2])
@maintain_rio_attrs
def nbr(stack):
    """Compute the Normalized Burn Ratio (NBR)"""
    nir = stack.sel(band=BandCommon.NIR)
    swir2 = stack.sel(band=BandCommon.SWIR2)
    nbr_v = (nir - swir2) / (nir + swir2)
    return nbr_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.GREEN])
@maintain_rio_attrs
def gndvi(stack):
    """Compute the Green Normalized Difference Vegetation Index (GNDVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    green = stack.sel(band=BandCommon.GREEN)
    gndvi_v = (nir - green) / (nir + green)
    return gndvi_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.RED, BandCommon.BLUE])
@maintain_rio_attrs
def evi(stack):
    """Compute the Enhanced Vegetation Index (EVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    blue = stack.sel(band=BandCommon.BLUE)
    evi_v = 2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1)).drop_vars("band")
    return evi_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def avi(stack):
    """Compute the Atmospherically Resistant Vegetation Index (AVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    avi_v = (nir * (1 - red) * (nir - red)) ** (1 / 3)
    return avi_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def savi(stack):
    """Compute the Soil Adjusted Vegetation Index (SAVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    savi_v = ((nir - red) / (nir + red + 0.5)) * 0.5
    return savi_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.GREEN, BandCommon.NIR])
@maintain_rio_attrs
def ndwi(stack):
    """Compute the Normalized Difference Water Index (NDWI)"""
    green = stack.sel(band=BandCommon.GREEN)
    nir = stack.sel(band=BandCommon.NIR)
    ndwi_v = (green - nir) / (green + nir)
    return ndwi_v


@compatible_with([Platform.LANDSAT_TM])
@requires_bands([
    BandCommon.BLUE,
    BandCommon.GREEN,
    BandCommon.RED,
    BandCommon.NIR,
    BandCommon.SWIR1,
    BandCommon.SWIR2,
])
@maintain_rio_attrs
def tcg(stack):
    """Compute the Tasseled Cap Greenness Index (TCG)"""
    blue = stack.sel(band=BandCommon.BLUE)
    green = stack.sel(band=BandCommon.GREEN)
    red = stack.sel(band=BandCommon.RED)
    nir = stack.sel(band=BandCommon.NIR)
    swir1 = stack.sel(band=BandCommon.SWIR1)
    swir2 = stack.sel(band=BandCommon.SWIR2)
    tcg_v = (
        0.2043 * blue
        + 0.4158 * green
        + 0.5524 * red
        + 0.5741 * nir
        + 0.3124 * swir1
        + 0.2303 * swir2
    )
    return tcg_v


@compatible_with([Platform.LANDSAT_TM])
@requires_bands([
    BandCommon.BLUE,
    BandCommon.GREEN,
    BandCommon.RED,
    BandCommon.NIR,
    BandCommon.SWIR1,
    BandCommon.SWIR2,
])
@maintain_rio_attrs
def tcw(stack):
    """Compute the Tasseled Cap Wetness Index (TCW)"""
    blue = stack.sel(band=BandCommon.BLUE)
    green = stack.sel(band=BandCommon.GREEN)
    red = stack.sel(band=BandCommon.RED)
    nir = stack.sel(band=BandCommon.NIR)
    swir1 = stack.sel(band=BandCommon.SWIR1)
    swir2 = stack.sel(band=BandCommon.SWIR2)
    tcw_v = (
        0.1509 * blue
        + 0.1973 * green
        + 0.3279 * red
        + 0.3406 * nir
        + 0.7112 * swir1
        + 0.4572 * swir2
    )
    return tcw_v


@compatible_with([Platform.LANDSAT_TM])
@requires_bands([
    BandCommon.BLUE,
    BandCommon.GREEN,
    BandCommon.RED,
    BandCommon.NIR,
    BandCommon.SWIR1,
    BandCommon.SWIR2,
])
@maintain_rio_attrs
def tcb(stack):
    """Compute the Tasseled Cap Brightness Index (TCB)"""
    blue = stack.sel(band=BandCommon.BLUE)
    green = stack.sel(band=BandCommon.GREEN)
    red = stack.sel(band=BandCommon.RED)
    nir = stack.sel(band=BandCommon.NIR)
    swir1 = stack.sel(band=BandCommon.SWIR1)
    swir2 = stack.sel(band=BandCommon.SWIR2)
    tcb_v = (
        0.3037 * blue
        + 0.2793 * green
        + 0.4743 * red
        + 0.5585 * nir
        + 0.5082 * swir1
        + 0.1863 * swir2
    )
    return tcb_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def sr(stack):
    """Compute the Simple Ratio (SR)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    sr_v = nir / red
    return sr_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.NIR, BandCommon.SWIR1])
@maintain_rio_attrs
def ndmi(stack):
    """Compute the Normalized Difference Moisture Index (NDMI)"""
    nir = stack.sel(band=BandCommon.NIR)
    swir1 = stack.sel(band=BandCommon.SWIR1)
    ndmi_v = (nir - swir1) / (nir + swir1)
    return ndmi_v


# TODO: Platform compatibility
@requires_bands([BandCommon.NIR, BandCommon.GREEN])
@maintain_rio_attrs
def gci(stack):
    """Compute the Green Chlorophyll Index (GCI)"""
    nir = stack.sel(band=BandCommon.NIR)
    green = stack.sel(band=BandCommon.GREEN)
    gci_v = (nir / green) - 1
    return gci_v


@compatible_with([
    Platform.LANDSAT_OLI,
    Platform.LANDSAT_TM,
    Platform.LANDSAT_ETM,
    Platform.SENTINEL_2,
])
@requires_bands([BandCommon.SWIR1, BandCommon.NIR])
@maintain_rio_attrs
def ndii(stack):
    """Compute the Normalized Difference Infrared Index (NDII)"""
    swir1 = stack.sel(band=BandCommon.SWIR1)
    nir = stack.sel(band=BandCommon.NIR)
    ndii_v = (swir1 - nir) / (swir1 + nir)
    return ndii_v


_indices_map = {
    Index.NDVI: ndvi,
    Index.NBR: nbr,
    Index.GNDVI: gndvi,
    Index.EVI: evi,
    Index.AVI: avi,
    Index.SAVI: savi,
    Index.NDWI: ndwi,
    Index.TCG: tcg,
    Index.TCW: tcw,
    Index.TCB: tcb,
    Index.SR: sr,
    Index.NDMI: ndmi,
    Index.GCI: gci,
    Index.NDII: ndii,
}


def _bad_index_choice(stack):
    raise ValueError("No index function implemented for current index.") from None


def compute_indices(image_stack: xr.DataArray, indices: list[str]):
    """Compute spectral indices on a stack of images

    Parameters
    ----------
    image_stack : xr.DataArray
        stack of images. The 'band' dimension coordinates must contain
        enums.BandCommon types.
    indices : list[Index]
        list of spectral indices to compute
    platform : Platform
        platform from which images were collected

    Returns
    -------
        xr.DataArray: stack of images with spectral indices stacked along
        the band dimension.
    """
    indices = _to_index_enums(indices)
    index = {}
    for index_choice in indices:
        index[index_choice] = _indices_map.get(index_choice, _bad_index_choice)(
            image_stack
        )
    index_stack = xr.concat(index.values(), dim=pdIndex(index.keys(), name="band"))
    return index_stack


def _to_index_enums(indices: List[str]) -> List[Index]:
    """Convert a list of index names to Index enums"""
    valid_names = []
    for name in indices:
        try:
            val = Index[name.upper()]
            valid_names.append(val)
        except KeyError:
            raise ValueError(
                f"Index '{name}' not found. Valid indices are:"
                f" {[str(i) for i in list(Index)]}"
            ) from None
    return valid_names
