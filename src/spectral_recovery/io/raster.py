import re
import rioxarray

import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path
from typing import List
from rasterio._err import CPLE_AppDefinedError
from pandas.core.tools.datetimes import DateParseError

from spectral_recovery.enums import BandCommon, Index

DATETIME_FREQ = "YS"
REQ_DIMS = ["band", "time", "y", "x"]

VALID_YEAR = re.compile(r"^\d{4}$")


def read_and_stack_tifs(
    paths_to_tifs: List[str],
    path_to_mask: str = None,
):
    """Reads and stacks a list of tifs into a 4D DataArray.

    The returned DataArray will have dimensions: 'time', 'band', 'y', and 'x'. The
    'band' dimension coordinates will be either enums.Index or enums.BandCommon types, and 'time' dimension
    will be datetime objected enabled with the '.dt' accessor.

    Parameters
    ----------
    path_to_tifs : list of str,
    per_year : bool, optional
    per_band : bool, optional
    path_to_mask : str, optional

    Returns
    -------
    stacked_data : xr.DataArray

    """
    image_dict = {}
    for file in paths_to_tifs:
        with rioxarray.open_rasterio(Path(file), chunks="auto") as data:
            image_dict[Path(file).stem] = data

    time_keys = []
    for filename in image_dict.keys():
        if _str_is_year(filename):
            time_keys.append(pd.to_datetime(filename))
        else:
            raise ValueError(
                f"TIF filenames must be in format 'YYYY' but recived: '{filename}'"
            ) from None

    stacked_data = _stack_bands(image_dict.values(), time_keys, dim_name="time")
    band_names = _to_band_or_index(stacked_data.attrs["long_name"])
    stacked_data = stacked_data.assign_coords(band=list(band_names.values()))

    # TODO: catch missing dimension error here
    stacked_data = stacked_data.transpose(*REQ_DIMS)
    stacked_data = stacked_data.sortby("time")

    if path_to_mask is not None:
        with rioxarray.open_rasterio(Path(path_to_mask), chunks="auto") as mask:
            stacked_data = _mask_stack(stacked_data, mask)

    return stacked_data


def _to_band_or_index(names_list: List[str]):
    valid_names_mapping = {}
    for name in names_list:
        try:
            val = BandCommon[name.lower()]
            valid_names_mapping[name] = val
            continue
        except ValueError:
            pass
        try:
            val = Index[name.lower()]
            valid_names_mapping[name] = val
        except ValueError:
            # TODO: add accepted values to error message and direct user to documentation
            raise ValueError
    return valid_names_mapping


def _str_is_year(year_str):
    if VALID_YEAR.match(year_str) is None:
        return False
    else:
        return True


def _stack_bands(bands, names, dim_name):
    """Stack 3D image stacks to create 4D image stack"""
    # TODO: handle band dimension/coordinate errors
    stacked_bands = xr.concat(bands, dim=pd.Index(names, name=dim_name))
    return stacked_bands


def _mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan):
    """Mask a ND stack with 2D mask"""
    # TODO: should this allow more than 2D mask?
    if len(mask.dims) != 2:
        raise ValueError(f"Mask must be 2D but {len(mask.dims)}D mask provided.")
    masked_stack = stack.where(mask, fill)
    return masked_stack


def metrics_to_tifs(
    metrics_array: xr.DataArray,
    out_dir: str,
):
    # NOTE: out_raster MUST be all null otherwise merging of rasters will fail
    out_raster = xr.full_like(metrics_array[0, 0, :, :], np.nan)
    for metric in metrics_array["metric"].values:
        xa_dataset = xr.Dataset()
        for band in metrics_array["band"].values:
            out_metric = metrics_array.sel(metric=metric, band=band)

            merged = out_metric.combine_first(out_raster)
            xa_dataset[str(band)] = merged
            try:
                filename = f"{out_dir}/{str(metric)}.tif"
                xa_dataset.rio.to_raster(raster_path=filename)
            # TODO: don't except on an error hidden from API users...
            except CPLE_AppDefinedError as exc:
                raise PermissionError(
                    f"Permission denied to overwrite {filename}. Is the existing TIF"
                    " open in an application (e.g QGIS)? If so, try closing it before"
                    " your next run to avoid this error."
                ) from None
    return
