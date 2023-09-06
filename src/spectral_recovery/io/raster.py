import rioxarray

import pandas as pd
import numpy as np
import xarray as xr

from pathlib import Path
from typing import List
from rasterio._err import CPLE_AppDefinedError

from spectral_recovery.enums import BandCommon, Index

DATETIME_FREQ = "YS"
REQ_DIMS = ["band", "time", "y", "x"]

def read_and_stack_tifs(
    path_to_tifs: List[str],
    per_year: bool = False,
    per_band: bool = False,
    path_to_mask: str = None,
    start_year: str = None,
    end_year: str = None,
):
    """ Reads and stacks a list of tifs into a 4D DataArray.

    The returned DataArray will have dimensions: 'time', 'band', 'y', and 'x'. The
    'band' dimension coordinates will be either enums.Index or enums.BandCommon types, and 'time' dimension
    will be datetime objected enabled with the '.dt' accessor.

    Parameters
    ----------
    path_to_tifs : list of str, 
    per_year : bool, optional 
    per_band : bool, optional
    path_to_mask : str, optional
    start_year : str, optional
    end_year : str, optional

    Returns
    -------
    stacked_data : xr.DataArray

    """
    if not per_year and not per_band:
        raise ValueError("Expected either per_year or per_band to be set.") from None
    if per_year and per_band:
        raise ValueError("Only one of per_year or per_band can be True.") from None

    image_dict = {}
    for file in path_to_tifs:
        with rioxarray.open_rasterio(Path(file), chunks="auto") as data:
            image_dict[Path(file).stem] = data

    if per_year:
        try:
            time_keys = [pd.to_datetime(filename) for filename in image_dict.keys()]
        except ValueError:
            # TODO: add more information here for users. Either link to additional docs or make error message more descriptive.
            raise ValueError(
                "Cannot stack tifs because per_year=True but filenames of TIFs are not"
                " in 'YYYY.tif' format."
            ) from None
        stacked_data = _stack_bands(image_dict.values(), time_keys, dim_name="time")
        band_names = _to_band_or_index(stacked_data.attrs["long_name"])
        stacked_data = stacked_data.assign_coords(band=list(band_names.values()))

    if per_band:
        if not start_year and not end_year:
            raise ValueError("Cannot stack tifs because per_band=True but start_year and end_year are not provided. Start and end years of the timeseries must be defined because it cannot be assumed from the TIFs.")
        
        try:
            band_keys = _to_band_or_index(image_dict.keys())
            image_dict = { band_keys[key] : value for key, value in image_dict.items() }
        except ValueError:
            # TODO: add accepted values to error message and direct user to documentation 
                raise ValueError(
                    "Cannot stack bands because per_band=True but filenames of TIFs"
                    " are not recognized common band name or index acronym."
                ) from None
        
        for key, data in image_dict.items():
            image_dict[key] = data.rename({"band": "time"})
            stacked_data = _stack_bands(
            image_dict.values(), image_dict.keys(), dim_name="band")

        stacked_data = stacked_data.assign_coords(
        time=(pd.date_range(start=start_year, end=end_year, freq=DATETIME_FREQ)))
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
                val = BandCommon(name.upper())
                valid_names_mapping[name] = (val)
                continue
            except ValueError:
                pass
            try:
                val = Index(name.upper())
                valid_names_mapping[name] = (val)
            except ValueError:
                # TODO: add accepted values to error message and direct user to documentation 
                raise ValueError
        return valid_names_mapping

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
            except CPLE_AppDefinedError as exc:
                raise PermissionError(
                    f"Could not write output to {filename} because a TIF already"
                    " exists and could not be overwritten, likely because"
                    " it's open elsewhere. Is the existing TIF open in an"
                    " application (e.g QGIS)? If so, try closing it before your"
                    " next run to avoid this error."
                ) from None
    return