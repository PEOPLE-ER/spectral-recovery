"""Functions for reading and writing raster data .

Handles reading timeseries of TIFs into a single DataArray, ensures
band names and attributes are consistent. Also handles writing.
"""

import json

from pathlib import Path
from typing import List, Dict

import rioxarray

import pandas as pd
import numpy as np
import xarray as xr

from rasterio._err import CPLE_AppDefinedError

from spectral_recovery.enums import BandCommon, Index, Platform
from spectral_recovery._config import VALID_YEAR, REQ_DIMS


def read_and_stack_tifs(
    path_to_tifs: List[str] | str,
    platform: List[str] | str,
    band_names: Dict[int, str] = None,
    path_to_mask: str = None,
    array_type: str = "numpy",
):
    """Reads and stacks a list of tifs into a 4D DataArray.

    Parameters
    ----------
    path_to_tifs : list of str
        List of paths to TIFs or path to directory containing TIFs.
    platform : {"landsat_etm", "landsat_tm", "landsat_oli", "sentinel_2"}
        Platform(s) from which TIF imagery was derived Must be one of: 'landsat_etm', 'landsat_tm',
    band_names : dict, optional
        Dictionary mapping band numbers to band names. If not provided,
        band names will be read from the TIFs band descriptions.
    path_to_mask : str, optional
        Path to a 2D data mask to apply over all TIFs.
    array_type : {"dask", "numpy"}
        The type of array to use store data, either numpy or dask.
        NumPy arrays will be loaded into memory while Dask arrays will be
        lazily evaluated until being explicitly loaded into memory with a
        .compute() call. Default is "numpy".

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
            # Grab all TIFs in directory
            path_to_tifs = list(Path(path_to_tifs).glob("*.tif"))
    for file in path_to_tifs:
        if array_type == "numpy":
            with rioxarray.open_rasterio(Path(file)) as data:
                image_dict[Path(file).stem] = data
        else:
            # Using open_rasterio with chunks="auto" will load the data as a dask array
            with rioxarray.open_rasterio(Path(file), chunks="auto") as data:
                image_dict[Path(file).stem] = data

    # Parse the year of the raster/composite from it's filename and use the year
    # as the DataArray's time dimension coordinate.
    time_keys = []
    for filename in image_dict:
        if _str_is_year(filename):
            time_keys.append(pd.to_datetime(filename))
        else:
            raise ValueError(
                f"TIF filenames must be in format 'YYYY' but recieved: '{filename}'"
            ) from None
    # Stack images along the time dimension
    stacked_data = _stack_bands(image_dict.values(), time_keys, dim_name="time")



    if band_names is None:
        # If band descriptions are present in the images, then rioxarray 
        # will set those descriptions as 'long_name' attributes on the DataArray.
        # If there is no long_name attr then there are no descriptions.
        try:
            band_names_new = _to_band_or_index_enums(stacked_data.attrs["long_name"])
        except KeyError:
            raise ValueError(
                "Band descriptions not found in TIFs. Please provide band "
                " names using the band_names argument."
            ) from None
    else:
        band_names_old = stacked_data.band.values
        # Only accept mappings for bands that exist in the images.
        for b in band_names.keys():
            if b not in band_names_old:
                raise ValueError(
                    f"Band {b} not found in TIFs. Please provide a mapping for only"
                    f" bands: {band_names_old}"
                ) from None
        # This is working on the assumption that bands are always integers when no band
        # description is provided e.g band_names_old == [0,1,2]
        for band_num in band_names_old:
            if band_num not in band_names.keys():
                raise ValueError(
                    f"Band {band_num} not found in `band_names` dictionary. Please"
                    f" provide a mapping for all bands: {band_names_old}"
                ) from None

            band_names[band_num] = _to_band_names([band_names[band_num]])[0]

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
            val = Platform[name.upper()]
            valid_names.append(val)
        except KeyError:
            raise ValueError(
                f"Platform '{name}' not found. Valid platforms are: {list(Platform)}"
            ) from None
    return valid_names


def _to_band_names(names_list: List[str]) -> Dict[str | int, str]:
    """Convert a list of band or index names to """
    valid_names = []
    for name in names_list:
        try:
            val = BandCommon[name.upper()]
            valid_names.append(val)
            continue
        except KeyError:
            pass
        try:
            val = Index[name.upper()]
            valid_names.append(val)
        except KeyError:
            raise ValueError(
                f"Band or index '{name}' not found. Valid bands and indices names are:"
                f" {[str(b) for b in list(BandCommon)]} and"
                f" {[str(i) for i in list(Index)]}"
            ) from None
    return valid_names


def _str_is_year(year_str) -> bool:
    """Check if a string is a valid year (YYYY)"""
    if VALID_YEAR.match(year_str) is None:
        return False
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
            # TODO: Probably shouldn't except on an error hidden from API users...
            except CPLE_AppDefinedError:
                raise PermissionError(
                    f"Permission denied to overwrite {filename}. Is the existing TIF"
                    " open in an application (e.g QGIS)? If so, try closing it before"
                    " your next run to avoid this error."
                ) from None
