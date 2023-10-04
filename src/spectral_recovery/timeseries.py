import rioxarray

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from typing import Union, Tuple
from datetime import datetime
from shapely.geometry import box
from spectral_recovery.indices import indices_map
from spectral_recovery.enums import Index, BandCommon

DATETIME_FREQ = "YS"
REQ_DIMS = ["band", "time", "y", "x"]


# def _stack_from_user_input(timeseries_dict, mask, timeseries_range=None):
#     """Stack a dictionary of rasters into a 4D DataArray.

#     Dimensions of output DataArray will be (band, time, y, x) and the
#     time dimension will contain np.datetime64 coordinates with datetime
#     frequency set to `DATETIME_FREQ`.

#     Parameters
#     ----------
#     timeseries_dict : Dict of str or DataArray
#         Dict of paths to rasters or DataArrays to stack.
#     mask : DataArray
#         Mask to apply to stacked DataArrays.
#     timeseries_range : list of int, optional
#         The year range for the timeseries data.

#     Returns
#     -------
#     DataArray :
#         A 4D DataArray containing all rasters passed in
#         `timeseries_dict` and optionally masked.

#     """
#     if all([isinstance(data, str) for data in timeseries_dict.values()]):
#         # Need to read in data
#         for name, file in timeseries_dict.items():
#             with rioxarray.open_rasterio(file, chunks="auto") as data:
#                 timeseries_dict[name] = data

#     if all(
#         [
#             (isinstance(key, BandCommon) or isinstance(key, Index))
#             for key in timeseries_dict.keys()
#         ]
#     ):
#         for key, data in timeseries_dict.items():
#             timeseries_dict[key] = data.rename({"band": "time"})
#         stacked_data = _stack_bands(
#             timeseries_dict.values(), timeseries_dict.keys(), dim_name="band"
#         )
#     elif all([isinstance(key, np.datetime64) for key in timeseries_dict.keys()]):
#         stacked_data = _stack_bands(
#             timeseries_dict.values(), timeseries_dict.keys(), dim_name="time"
#         )
#     # TODO: catch missing dimension error here
#     stacked_data = stacked_data.transpose(*REQ_DIMS)
#     stacked_data = stacked_data.sortby("time")

#     if timeseries_range is not None:
#         stacked_data = stacked_data.assign_coords(
#             time=(pd.date_range(*timeseries_range, freq=DATETIME_FREQ))
#         )
#     if not all(
#         [isinstance(index, np.datetime64) for index in stacked_data.coords["time"].data]
#     ):
#         raise ValueError(
#             "Time dimension not initialized as np.datetime64. If rasters passed"
#             " per-band please ensure the `time_series` range parameter is set."
#             " Otherwise... something's gone wrong."
#         ) from None
#     if mask:
#         masked_data = _mask_stack(stacked_data, mask)
#         return masked_data
#     else:
#         return stacked_data


def _stack_bands(bands, names, dim_name):
    """Stack 3D image stacks to create 4D image stack"""
    # TODO: handle band dimension/coordinate errors
    stacked_bands = xr.concat(bands, dim=pd.Index(names, name=dim_name))
    return stacked_bands


# def _mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan):
#     """Mask a ND stack with 2D mask"""
#     # TODO: should this allow more than 2D mask?
#     if len(mask.dims) != 2:
#         raise ValueError(f"Mask must be 2D but {len(mask.dims)}D mask provided.")
#     masked_stack = stack.where(mask, fill)
#     return masked_stack


def datetime_to_index(
    value: Union[datetime, Tuple[datetime]], return_list: bool = False
) -> pd.DatetimeIndex:
    """Convert datetime or range of datetimes into pd.DatetimeIndex

    For ease of indexing through a DataArray object
    """
    if (isinstance(value, tuple) or isinstance(value, list)) and len(value) == 2:
        dt_range = pd.date_range(*value, freq=DATETIME_FREQ)
    else:
        try:
            if len(value) > 2:
                raise ValueError(
                    "Passed value={value} but `datetime` must be a single Timestamp or"
                    " an iterable with exactly two Timestamps."
                )
            dt_range = pd.date_range(start=value[0], end=value[0], freq=DATETIME_FREQ)
        except TypeError:
            dt_range = pd.date_range(start=value, end=value, freq=DATETIME_FREQ)
    if not return_list:
        return dt_range
    return dt_range.to_list()


@xr.register_dataarray_accessor("satts")
class SatelliteTimeSeries:
    """A accessor for operations commonly performed over 
    a timeseries of satellite images stored in an xarray.DataArray.

    See "Extending xarray using accessors" for more information:
    https://docs.xarray.dev/en/stable/internals/extending-xarray.html

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._valid = None

    # TODO: change this method to "is_contiuous" or something and only check for continuity of years
    #   Move the check for valid dim names to a seperate method.
    @property
    def is_annual_composite(self):
        """Return flag for whether DataArray is valid annual comppsite.

        Checks whether the object has the required dimension labels (as
        defined by/for project) and valid coordinate values.

        """
        if self._valid is None:
            years = self._obj.coords["time"].dt.year.values
            if not set(self._obj.dims) == set(REQ_DIMS):
                self._valid = False
            elif not np.all((years == list(range(min(years), max(years) + 1)))):
                self._valid = False
            else:
                self._valid = True
        return self._valid

    def contains_spatial(self, polygons: gpd.GeoDataFrame) -> bool:
        """Check if stack contains polygons."""
        # NOTE: if this changes to looking at individual polygons
        # rather than the bbox of all polygons, consider this algo:
        # https://stackoverflow.com/questions/14697442/
        ext = box(*self._obj.rio.bounds())
        poly_ext = box(*polygons.total_bounds)
        if not ext.contains(poly_ext):
            return False
        return True

    def contains_temporal(self, years: datetime) -> bool:
        """Check if stack contains year/year range."""
        required_years = datetime_to_index(years, return_list=True)
        for year in required_years:
            if not (pd.to_datetime(str(year)) in self._obj.coords["time"].values):
                return False
        return True

    # NOTE: conceptually, does having this method in this class work?
    def indices(self, indices_list) -> xr.DataArray:
        """Compute indices

        Parameters
        ----------
        indices_list : list of str
            The list of indices to compute/produce.

        Returns
        --------
        xr.DataArray
            A 4D (band, time, y, x) DataArray with indices
            stacked inside the band dimension.
        """
        indices_dict = {}
        for indice_input in indices_list:
            indice = Index(indice_input)
            indices_dict[indice] = indices_map[indice](self._obj)
        indices = xr.concat(indices_dict.values(), dim=pd.Index(indices_dict.keys(), name="band"))
        return indices

    def stats(self, percentile, dims):
        "Compute statistics for a given dataarray object."
        stats =  {}
        stats["mean"] = self._obj.mean(dim=dims, skipna=True)
        stats["max"]  = self._obj.max(dim=dims, skipna=True)
        stats["min"]  = self._obj.min(dim=dims, skipna=True)
        stats["median"] = self._obj.median(dim=dims, skipna=True)
        stats["quantile"] =  self._obj.quantile(q=percentile, dim=dims, skipna=True)
        stats["std"] = self._obj.std(dim=dims, skipna=True)
        stats["sum"] =  self._obj.std(dim=dims, skipna=True)
        stats_xr  = xr.concat(stats.values(), dim=pd.Index(stats.keys(), name="stats"))
        return stats_xr


        

