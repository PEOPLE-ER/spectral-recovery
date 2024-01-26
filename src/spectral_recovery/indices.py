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

import spyndex as spx

from spectral_recovery._utils import maintain_rio_attrs
from spectral_recovery.enums import Index, BandCommon, Platform


def compatible_with(platform: List[Platform]):
    """A decorator for assigning platform compatibility to a function.

    Parameters
    ----------
    platform : List[Platform]
        List of platforms compatible with the function.

    """
    def compatible_with_decorator(func):
        """Sub-decorator for assigning platform compatibility to a function."""

        @functools.wraps(func)
        def compatible_with_wrapper(stack, *args, **kwargs):
            for input_platform in stack.attrs["platform"]:
                if input_platform not in platform:
                    raise ValueError(
                        f"Function {func.__name__} is not compatible with platform"
                        f" {stack.attrs['platform']}. Only compatible with {platform}"
                    ) from None
            return func(stack, *args, **kwargs)

        return compatible_with_wrapper

    return compatible_with_decorator


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


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def ndvi(stack: xr.DataArray):
    """Compute the Normalized Difference Vegetation Index (NDVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    ndvi_v = (nir - red) / (nir + red)
    return ndvi_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.SWIR2])
@maintain_rio_attrs
def nbr(stack):
    """Compute the Normalized Burn Ratio (NBR)"""
    nir = stack.sel(band=BandCommon.NIR)
    swir2 = stack.sel(band=BandCommon.SWIR2)
    nbr_v = (nir - swir2) / (nir + swir2)
    return nbr_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.GREEN])
@maintain_rio_attrs
def gndvi(stack):
    """Compute the Green Normalized Difference Vegetation Index (GNDVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    green = stack.sel(band=BandCommon.GREEN)
    gndvi_v = (nir - green) / (nir + green)
    return gndvi_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.RED, BandCommon.BLUE])
@maintain_rio_attrs
def evi(stack):
    """Compute the Enhanced Vegetation Index (EVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    blue = stack.sel(band=BandCommon.BLUE)
    evi_v = 2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1)).drop_vars("band")
    return evi_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def avi(stack):
    """Compute the Atmospherically Resistant Vegetation Index (AVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    avi_v = (nir * (1 - red) * (nir - red)) ** (1 / 3)
    return avi_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def savi(stack):
    """Compute the Soil Adjusted Vegetation Index (SAVI)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    savi_v = ((nir - red) / (nir + red + 0.5)) * (1.0 + 0.5)
    return savi_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.GREEN, BandCommon.NIR])
@maintain_rio_attrs
def ndwi(stack):
    """Compute the Normalized Difference Water Index (NDWI)"""
    green = stack.sel(band=BandCommon.GREEN)
    nir = stack.sel(band=BandCommon.NIR)
    ndwi_v = (green - nir) / (green + nir)
    return ndwi_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.RED])
@maintain_rio_attrs
def sr(stack):
    """Compute the Simple Ratio (SR)"""
    nir = stack.sel(band=BandCommon.NIR)
    red = stack.sel(band=BandCommon.RED)
    sr_v = nir / red
    return sr_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.SWIR1])
@maintain_rio_attrs
def ndmi(stack):
    """Compute the Normalized Difference Moisture Index (NDMI)"""
    nir = stack.sel(band=BandCommon.NIR)
    swir1 = stack.sel(band=BandCommon.SWIR1)
    ndmi_v = (nir - swir1) / (nir + swir1)
    return ndmi_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.NIR, BandCommon.GREEN])
@maintain_rio_attrs
def gci(stack):
    """Compute the Green Chlorophyll Index (GCI)"""
    nir = stack.sel(band=BandCommon.NIR)
    green = stack.sel(band=BandCommon.GREEN)
    gci_v = (nir / green) - 1
    return gci_v


@compatible_with(
    [
        Platform.LANDSAT_OLI,
        Platform.LANDSAT_TM,
        Platform.LANDSAT_ETM,
        Platform.SENTINEL_2,
    ]
)
@requires_bands([BandCommon.SWIR1, BandCommon.NIR])
@maintain_rio_attrs
def ndii(stack):
    """Compute the Normalized Difference Infrared Index (NDII)"""
    swir1 = stack.sel(band=BandCommon.SWIR1)
    nir = stack.sel(band=BandCommon.NIR)
    ndii_v = (nir - swir1) / (nir + swir1)
    return ndii_v


_indices_map = {
    Index.NDVI: ndvi,
    Index.NBR: nbr,
    Index.GNDVI: gndvi,
    Index.EVI: evi,
    Index.AVI: avi,
    Index.SAVI: savi,
    Index.NDWI: ndwi,
    Index.SR: sr,
    Index.NDMI: ndmi,
    Index.GCI: gci,
    Index.NDII: ndii,
}


def _bad_index_choice(stack):
    raise ValueError("No index function implemented for current index.") from None

@maintain_rio_attrs
def compute_indices(image_stack: xr.DataArray, indices: list[str]):
    """Compute spectral indices using the spyndex package.


    Parameters
    ----------
    image_stack : xr.DataArray
        stack of images. The 'band' dimension coordinates must contain
        enums.BandCommon types.
    indices : list of str
        list of spectral indices to compute
    platform : Platform
        platform from which images were collected

    Returns
    -------
        xr.DataArray: stack of images with spectral indices stacked along
        the band dimension.

    """
    params_dict = _build_params_dict(image_stack)
    index_stack = spx.computeIndex(
        indices,
        params=params_dict
    )
    return index_stack


def _build_params_dict(image_stack: xr.DataArray):
    """Build dict of standard names and slices required by computeIndex.
    
    Slices will be taken along the band dimension of image_stack,
    selecting for each of the standard band names that computeIndex
    accepts. Any name that is not in image_stack will not be included
    in the dictionary.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray from which to take slices. Must have a band
        dimension and band coordinates value should be standard names
        for the respective band. For more info, see here:
        https://github.com/awesome-spectral-indices/awesome-spectral-indices

    Returns
    -------
    band_dict : dict
        Dictionary mapping standard names to slice of image_stack.

    """
    standard_names = list(spx.bands)
    params_dict = {}
    for standard in standard_names:
        try:
            band_slice = image_stack.sel(band=standard)
            params_dict[standard] = band_slice
        except KeyError:
            continue
    return params_dict




