import re
import rioxarray

import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path
from typing import List, Dict
from rasterio._err import CPLE_AppDefinedError
from pandas.core.tools.datetimes import DateParseError

from spectral_recovery.enums import BandCommon, Index, Platform
from spectral_recovery.config import VALID_YEAR, REQ_DIMS


def read_and_stack_tifs(
    path_to_tifs: List[str] | str,
    platform: List[str] | str,
    band_names: Dict[int, str | BandCommon | Index] = None,
    path_to_mask: str = None,
):
    """Reads and stacks a list of tifs into a 4D DataArray.

    Parameters
    ----------
    path_to_tifs : list of str
        List of paths to TIFs or path to directory containing TIFs.
    path_to_mask : str, optional
        Path to a 2D data mask to apply over all TIFs.

    Returns
    -------
    stacked_data : xr.DataArray
        A 4D DataArray containing all rasters passed in
        `path_to_tifs` and optionally masked. The 'band' dimension coordinates
        will be either enums.Index or enums.BandCommon types, and 'time' dimension
        will be datetime object dervied from the filename.

    Notes
    -----
    Files must be named in the format 'YYYY.tif' where 'YYYY' is a valid year.

    """
    image_dict = {}
    if isinstance(path_to_tifs, str):
        # check if path is a directory
        if Path(path_to_tifs).is_dir():
            path_to_tifs = list(Path(path_to_tifs).glob("*.tif"))
    for file in path_to_tifs:
        with rioxarray.open_rasterio(Path(file), chunks="auto") as data:
            image_dict[Path(file).stem] = data

    time_keys = []
    for filename in image_dict.keys():
        if _str_is_year(filename):
            time_keys.append(pd.to_datetime(filename))
        else:
            raise ValueError(
                f"TIF filenames must be in format 'YYYY' but recieved: '{filename}'"
            ) from None

    stacked_data = _stack_bands(image_dict.values(), time_keys, dim_name="time")
    if band_names is None:
        try:
            band_names_new = _to_band_or_index_enums(stacked_data.attrs["long_name"])
        except KeyError:
            raise ValueError(
                "Band descriptions not found in TIFs. Please provide band "
                " names for bands {stack_data.band.values} using the `band_names`"
                " argument."
            )
    else:
        band_names_old = stacked_data.band.values
        for b in band_names.keys():
            if b not in band_names_old:
                raise ValueError(
                    f"Band {b} not found in TIFs. Please provide a mapping for only"
                    f" bands: {band_names_old}"
                ) from None

        if not all([k in band_names_old for k in band_names.keys()]):
            raise ValueError(
                f"Band names {band_names.keys()} not found in TIFs. Please provide a"
                f" mapping for bands: {band_names_old}"
            ) from None
        # This is working on the assumption that bands are always integers when no band
        # description is provided e.g band_names_old == [0,1,2]
        for band_num in band_names_old:
            if band_num not in band_names.keys():
                raise ValueError(
                    f"Band {band_num} not found in `band_names` dictionary. Please"
                    f" provide a mapping for all bands: {band_names_old}"
                ) from None
            else:
                band_names[band_num] = _to_band_or_index_enums([band_names[band_num]])[0]

        band_names_new = [
            band_names[k] for k in band_names_old
        ]  # this silently discards any bands in bands_names that are not in the TIFs

    stacked_data = stacked_data.assign_coords(band=band_names_new)
    # TODO: catch missing dimension error here
    stacked_data = stacked_data.transpose(*REQ_DIMS)
    stacked_data = stacked_data.sortby("time")

    if path_to_mask is not None:
        with rioxarray.open_rasterio(Path(path_to_mask), chunks="auto") as mask:
            stacked_data = _mask_stack(stacked_data, mask)

    stacked_data.attrs["platform"] = _to_platform_enums(platform)

    return stacked_data

def _to_platform_enums(platform: List[str]) -> List[Platform]:
    """Convert a list of platform names to Platform enums"""
    valid_names = []
    for name in platform:
        try:
            val = Platform[name.lower()]
            valid_names.append(val)
        except ValueError:
            raise ValueError(
                f"Platform {name} not found. Valid platforms are: {list(Platform)}"
            ) from None
    return valid_names


def _to_band_or_index_enums(names_list: List[str]) -> Dict[str, BandCommon | Index]:
    """Convert a list of band or index names to BandCommon or Index enums"""
    valid_names = []
    for name in names_list:
        try:
            val = BandCommon[name.lower()]
            valid_names.append(val)
            continue
        except ValueError:
            pass
        try:
            val = Index[name.lower()]
            valid_names.append(val)
        except ValueError:
            raise ValueError(
                f"Band or index {name} not found. Valid bands and indices are:"
                f" {list(BandCommon)} and {list(Index)}"
            ) from None
    return valid_names


def _str_is_year(year_str) -> bool:
    """Check if a string is a valid year (YYYY)"""
    if VALID_YEAR.match(year_str) is None:
        return False
    else:
        return True


def _stack_bands(bands, coords, dim_name) -> xr.DataArray:
    """Stack a bands along a named dimension with coordinates"""
    # TODO: Probably doesn't need to be a function...
    stacked_bands = xr.concat(bands, dim=pd.Index(coords, name=dim_name))
    return stacked_bands


def _mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan) -> xr.DataArray:
    """Mask a ND stack with 2D mask"""
    if len(mask.dims) != 2:
        raise ValueError(
            f"Only 2D masks are supported. {len(mask.dims)}D mask provided."
        )
    masked_stack = stack.where(mask, fill)
    return masked_stack


def _metrics_to_tifs(
    metric: xr.DataArray,
    out_dir: str,
) -> None:
    """
    Write a DataArray of metrics to TIFs.

    Parameters
    ----------
    metric : xr.DataArray
        The metric to write to TIFs. Must have dimensions: 'metric', 'band', 'y', and 'x'.
    out_dir : str
        Path to directory to write TIFs.

    """
    # NOTE: out_raster MUST be all null otherwise merging of rasters will fail
    out_raster = xr.full_like(metric[0, 0, :, :], np.nan)
    for m in metric["metric"].values:
        xa_dataset = xr.Dataset()
        for band in metric["band"].values:
            out_metric = metric.sel(metric=m, band=band)

            merged = out_metric.combine_first(out_raster)
            xa_dataset[str(band)] = merged
            try:
                filename = f"{out_dir}/{str(m)}.tif"
                xa_dataset.rio.to_raster(raster_path=filename)
            # TODO: don't except on an error hidden from API users...
            except CPLE_AppDefinedError as exc:
                raise PermissionError(
                    f"Permission denied to overwrite {filename}. Is the existing TIF"
                    " open in an application (e.g QGIS)? If so, try closing it before"
                    " your next run to avoid this error."
                ) from None
    return
