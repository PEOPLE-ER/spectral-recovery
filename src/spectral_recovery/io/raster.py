"""Functions for reading and writing raster data .

Handles reading timeseries of TIFs into a single DataArray, ensures
band names and attributes are consistent. Also handles writing.
"""

from pathlib import Path
from typing import List, Dict, Tuple

import rioxarray

import pandas as pd
import numpy as np
import xarray as xr

from spectral_recovery.utils import common_and_long_to_short

from spectral_recovery.config import (
    VALID_YEAR,
    REQ_DIMS,
    STANDARD_BANDS,
    SUPPORTED_INDICES,
)

COMMON_LONG_SHORT_DICT = common_and_long_to_short(STANDARD_BANDS)


def read_timeseries(
    path_to_tifs: str | Dict[str, str],
    band_names: Dict[int, str] = None,
    array_type: str = "dask",
):
    """Reads a timeseries of tifs into a 4D Xarray DataArray.

    Parameters
    ----------
    path_to_tifs : dict or str
        Dict of str paths to TIFs with keys as the year each raster or str path to directory containing TIFs.
    band_names : dict, optional
        Dictionary mapping band numbers to band names. If not provided,
        band names will be read from the TIFs band descriptions. Values can
        be short or long spectral band names or short index names.
    array_type : {"dask", "numpy"}
        The type of array to use store data, either numpy or dask.
        NumPy arrays will be loaded into memory while Dask arrays will be
        lazily evaluated until being explicitly loaded into memory with a
        .compute() call. Default is "dask".

    Returns
    -------
    stacked_data : xr.DataArray
        A 4D DataArray containing all rasters passed to
        `path_to_tifs` with time, band, y, and x coordinate dimensions.


    Notes
    -----
    If passing a directory of tifs to `path_to_tifs`, each file must be named following a 'YYYY.tif' format, where 'YYYY' is a valid year.

    """
    image_dict = {}
    if isinstance(path_to_tifs, str):
        directory_of_tifs = _get_tifs_from_dir(path_to_tifs)
        for file in directory_of_tifs:
            filename_year = Path(file).stem
            if _valid_year_str(filename_year):
                image_dict[pd.to_datetime(filename_year)] = _read_from_path(
                    file=file, array_type=array_type
                )
    elif isinstance(path_to_tifs, dict):
        for key_year, file in path_to_tifs.items():
            if _valid_year_str(str(key_year)):
                image_dict[pd.to_datetime(str(key_year))] = _read_from_path(
                    file=file, array_type=array_type
                )
    else:
        raise TypeError(
            f"Invalid path input. path_to_tifs can be a str path to a directory of TIFs or a dictionary mapping str years to str paths of individual TIF files. Recieved {type(path_to_tifs)}"
        )

    # Stack images along the time dimension
    stacked_data = xr.concat(
        image_dict.values(), dim=pd.Index(image_dict.keys(), name="time")
    )

    band_nums = stacked_data.band.values
    if band_names is None:
        band_names = _names_from_desc(raster_data=stacked_data, band_nums=band_nums)

    if _valid_band_name_mapping(band_names, band_nums):
        band_names = {num: band_names[num] for num in band_nums}
        standard_names, attr_names = _to_standard_band_names(band_names.values())

    stacked_data = stacked_data.assign_coords(band=standard_names)
    # TODO: catch missing dimension error here
    stacked_data = stacked_data.transpose(*REQ_DIMS)
    stacked_data = stacked_data.sortby("time")

    return stacked_data


def _valid_year_str(str_year):
    """Check if str is a valid year"""
    if VALID_YEAR.match(str_year) is None:
        raise ValueError(f"Cannot interpret {str_year} as year (YYYY).") from None
    return True


def _read_from_path(file, array_type):
    """Read TIF file into Xarray DataArray"""
    if array_type == "numpy":
        with rioxarray.open_rasterio(Path(file)) as data:
            xarray_tif = _floating(data)
    else:
        # Using open_rasterio with chunks="auto" will load the data as a dask array
        with rioxarray.open_rasterio(Path(file), chunks="auto") as data:
            xarray_tif = _floating(data)
    return xarray_tif


def _names_from_desc(raster_data: xr.DataArray, band_nums: List) -> Dict[int, str]:
    """Get band names from the raster band descriptions

    If band descriptions are present in the rasters,
    then rioxarray sets those descriptions as 'long_name'
    attributes on the DataArray object. T/f check if the
    "long_name" attribue exists, if it doesn't then there
    are no band descriptions on the raster.

    """
    try:
        long_names = raster_data.attrs["long_name"]
        band_names = dict(zip(band_nums, long_names))
    except KeyError:
        raise ValueError(
            "Band descriptions not found in TIFs. Please provide band "
            " names using the band_names argument."
        ) from None
    return band_names


def _get_tifs_from_dir(path: str) -> List[str]:
    """Return all tif files inside directory as list"""
    if Path(path).is_dir():
        # Grab all TIFs in the directory
        directory_of_tifs = list(Path(path).glob("*.tif"))
        if len(directory_of_tifs) == 0:
            raise ValueError(f"No TIFs found in directory {path}")
    else:
        raise ValueError("path_to_tifs is not a directory.")
    return directory_of_tifs


def _floating(data: xr.DataArray) -> np.float64:
    """Convert int to float64 dtype"""
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    return data


def _valid_band_name_mapping(band_names: Dict[int, str], band_nums: List[int]) -> bool:
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
        - True if all bands numbers in band_nums have a band_name mapping.

    Raises
    ------
    ValueError
        - If a band number is mapped that is not in band_nums
        - If a band number is missing a mapping in band_names

    """
    for b in band_names.keys():
        if b not in band_nums:
            raise ValueError(
                f"Invalid band to name mapping. {b} is not in raster bands of {band_nums} "
            )

    for num in band_nums:
        if num not in band_names.keys():
            raise ValueError(
                "Missing band mapping. Rasters have band numbers"
                f" {list(band_nums)} but names for only {band_names.keys()} provided."
            )

    return True


def _to_standard_band_names(in_names: List[str]) -> Tuple[List[str], List[str]]:
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
        elif given_name in SUPPORTED_INDICES:
            converted = True
            standard_names.append(given_name)

        if not converted:
            raise ValueError(
                "Band must be named standard, common, or long name for a spectral band, or a spectral index. Could not find"
                f" '{given_name}' in supported bands or indices."
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
