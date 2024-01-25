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
    stacked_data = xr.concat(image_dict.values(), dim=pd.Index(time_keys, name="time"))

    standard = _get_bands()
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
        
    if _valid_mapping(band_names, band_nums):
        band_names = {num: band_names[num] for num in band_nums}
        standard_names, attr_names = _all_names_to_standard(band_names.values(), standard)
    else:
        raise ValueError(
            f"Invalid band to name mapping. Rasters have bands {list(stacked_data.band.values)} ({list(band_names.keys())} provided)."
        )
    
    stacked_data = stacked_data.assign_coords(band=standard_names)
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


def _str_is_year(year_str) -> bool:
    """Check if a string is a valid year (YYYY)"""
    if VALID_YEAR.match(year_str) is None:
        return False
    return True


def _mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan) -> xr.DataArray:
    """Mask a ND stack with 2D mask"""
    if len(mask.dims) != 2:
        raise ValueError(
            f"Only 2D masks are supported. {len(mask.dims)}D mask provided."
        )
    masked_stack = stack.where(mask, fill)
    return masked_stack


def _get_bands():
    """Gets dict of standard band ids from bands.json"""
    f = open("src/spectral_recovery/data/bands.json")
    bands_info = json.load(f)
    return bands_info


def _common_long_to_short_names(bands):
    """Creates dict with common/long name keys and short name items
    
    Parameters
    ----------
    bands : dict
        Band name and info as read from bands.json.
    
    Returns
    -------
    common_long : dict
        Common and long name keys with short name items.

    """
    common_long = {}
    for short_name, data in bands.items():
        common_name = data['common_name']
        long_name = data['long_name']
        common_long[common_name] = short_name
        common_long[long_name] = short_name

    return common_long


def _valid_mapping(band_names, band_nums):
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


def _all_names_to_standard(in_names, standard):
    """ Map given names to standard names.
    
    Parameters
    -----------
    in_names : list
        Band names
    standard : list
        Standardized band names
    
    Returns
    -------
    tuple
        
        
    """
    short_and_common = _common_long_to_short_names(standard)

    standard_names = []
    attr_names = []
    for given_name in in_names:
        converted = False

        if given_name in short_and_common.keys():
            converted = True
            standard_names.append(short_and_common[given_name])
            attr_names.append(given_name)

        elif given_name in standard.keys():
            converted = True
            standard_names.append(standard[given_name])

        if not converted:
            raise ValueError(f"Could not find standard band name for {given_name}.")
        
    return (standard_names, attr_names)

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
            