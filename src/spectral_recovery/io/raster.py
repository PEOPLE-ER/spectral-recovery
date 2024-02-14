"""Functions for reading and writing raster data .

Handles reading timeseries of TIFs into a single DataArray, ensures
band names and attributes are consistent. Also handles writing.
"""

from pathlib import Path
from typing import List, Dict

import rioxarray

import pandas as pd
import numpy as np
import xarray as xr

from spectral_recovery._utils import bands_pretty_table, common_and_long_to_short
from rasterio._err import CPLE_AppDefinedError

from spectral_recovery._config import VALID_YEAR, REQ_DIMS, SUPPORTED_PLATFORMS, STANDARD_BANDS

COMMON_LONG_SHORT_DICT = common_and_long_to_short(STANDARD_BANDS)
BANDS_TABLE = bands_pretty_table()

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
    stacked_data = xr.concat(image_dict.values(), dim=pd.Index(time_keys, name="time"))

    band_nums = stacked_data.band.values
    if band_names is None:
        # If band descriptions are present in the rasters, then rioxarray
        # sets those descriptions as 'long_name' attributes on the DataArray.
        #
        # T/f check if the "long_name" attribue exists, if it doesn't then
        # there were no band descriptions.
        try:
            long_names = stacked_data.attrs["long_name"]
            band_names = dict(zip(band_nums, long_names))

        except KeyError:
            raise ValueError(
                "Band descriptions not found in TIFs. Please provide band "
                " names using the band_names argument."
            ) from None

    if _valid_band_name_mapping(band_names, band_nums):
        band_names = {num: band_names[num] for num in band_nums}
        standard_names, attr_names = _to_standard_band_names(band_names.values())
    else:
        raise ValueError(
            "Invalid band to name mapping. Rasters have bands"
            f" {list(stacked_data.band.values)} but {list(band_names.keys())} provided."
        )

    stacked_data = stacked_data.assign_coords(band=standard_names)
    # TODO: catch missing dimension error here
    stacked_data = stacked_data.transpose(*REQ_DIMS)
    stacked_data = stacked_data.sortby("time")

    if path_to_mask is not None:
        with rioxarray.open_rasterio(Path(path_to_mask), chunks="auto") as mask:
            stacked_data = _mask_stack(stacked_data, mask)

    stacked_data.attrs["platform"] = _to_standard_platform_names(platform)

    return stacked_data


def _str_is_year(year_str) -> bool:
    """Check if a string is a valid year (YYYY)"""
    if VALID_YEAR.match(year_str) is None:
        return False
    return True

def _valid_band_name_mapping(band_names, band_nums):
    """Check if band_names dict maps each band to a name and vice versa.

    Parameters
    ----------
    band_names : dict
        User input of band numbers to band names
    band_nums : list
        List of band numbers from rasters

    Returns
    ------
    bool
        - False if band number provided that is not in band_nums.
        - False if band number missing from band_names.
        - True otherwise.

    """
    for b in band_names.keys():
        if b not in band_nums:
            return False

    for num in band_nums:
        if num not in band_names.keys():
            False

    return True


def _to_standard_platform_names(platform: List[str]) -> List[str]:
    """Convert a list of platform names to platform names"""
    valid_names = []
    for name in platform:
        if name in SUPPORTED_PLATFORMS:
            valid_names.append(name)
        else:
            raise ValueError(
                f"Platform '{name}' not found. Valid platform names are:"
                f" {list(SUPPORTED_PLATFORMS)}"
            ) from None
    return valid_names


def _to_standard_band_names(in_names):
    """Map given names to standard names.

    Parameters
    -----------
    in_names : list
        Band names

    Returns
    -------
    tuple : tuple of lists
        lists of standard names and attribute (long) names, respectively


    """
    standard_names = []
    attr_names = []
    for given_name in in_names:
        converted = False
        if given_name in COMMON_LONG_SHORT_DICT.keys():
            converted = True
            standard_names.append(COMMON_LONG_SHORT_DICT[given_name])
            attr_names.append(given_name)

        elif given_name in STANDARD_BANDS:
            converted = True
            standard_names.append(given_name)

        if not converted:
            raise ValueError(
                "Band must be named standard, common, or long name. Could not find"
                f" '{given_name}' in catalogue. See table below for accepted names: \n\n"
                f" {BANDS_TABLE} \n\n"
            ).with_traceback(None) from None

    return (standard_names, attr_names)


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
