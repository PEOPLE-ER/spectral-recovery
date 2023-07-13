import rioxarray

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from enum import Enum, auto
from typing import Union, Tuple
from datetime import datetime
from shapely.geometry import box
from spectral_recovery.indices import indices_map
from spectral_recovery.enums import Index, BandCommon

DATETIME_FREQ = "YS"
REQ_DIMS = ["band", "time", "y", "x"]


def stack_from_files(timeseries_dict, mask, timeseries_range=None):
    """ """
    if all([isinstance(data, str) for data in timeseries_dict.values()]):
        # Need to read in data
        for name, file in timeseries_dict.items():
            with rioxarray.open_rasterio(file, chunks="auto") as data:
                timeseries_dict[name] = data

    if all(
        [
            (isinstance(key, BandCommon) or isinstance(key, Index))
            for key in timeseries_dict.keys()
        ]
    ):
        for key, data in timeseries_dict.items():
            timeseries_dict[key] = data.rename({"band": "time"})
        stacked_data = stack_bands(
            timeseries_dict.values(), timeseries_dict.keys(), dim_name="band"
        )
    elif all([isinstance(key, np.datetime64) for key in timeseries_dict.keys()]):
        stacked_data = stack_bands(
            timeseries_dict.values(), timeseries_dict.keys(), dim_name="time"
        )
    # TODO: catch missing dimension error here
    stacked_data = stacked_data.transpose(*REQ_DIMS)
    stacked_data = stacked_data.sortby("time")

    if timeseries_range is not None:
        stacked_data = stacked_data.assign_coords(
            time=(pd.date_range(*timeseries_range, freq=DATETIME_FREQ))
        )
    if not all(
        [isinstance(index, np.datetime64) for index in stacked_data.coords["time"].data]
    ):
        raise ValueError(
            "Time dimension not initialized as np.datetime64. If rasters passed per-band please "
            "ensure the `time_series` range parameter is set. Otherwise... something's gone wrong."
        ) from None
    if mask:
        masked_data = mask_stack(stacked_data, mask)
        return masked_data
    else:
        return stacked_data


def stack_bands(bands, names, dim_name):
    """Stack 3D image stacks to create 4D image stack"""
    # TODO: handle band dimension/coordinate errors
    stacked_bands = xr.concat(bands, dim=pd.Index(names, name=dim_name))
    return stacked_bands


def mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan):
    """Mask a ND stack with 2D mask"""
    # TODO: should this allow more than 2D mask?
    if len(mask.dims) != 2:
        raise ValueError(f"Mask must be 2D but {len(mask.dims)}D mask provided.")
    masked_stack = stack.where(mask, fill)
    return masked_stack


def datetime_to_index(
    value: Union[datetime, Tuple[datetime]], list: bool = False
) -> pd.DatetimeIndex:
    """Convert datetime or range of datetimes into pd.DatetimeIndex

    For ease of indexing through a DataArray object
    """
    if isinstance(value, tuple):
        dt_range = pd.date_range(*value, freq=DATETIME_FREQ)
    else:
        dt_range = pd.date_range(start=value, end=value, freq=DATETIME_FREQ)
    if not list:
        return dt_range
    return dt_range.to_list()


@xr.register_dataarray_accessor("yearcomp")
class YearlyCompositeAccessor:
    """A DataArray accessor for annual composite operations.

    For methods related to yearly composite timeseries as well as
    general image stack operations.

    See "Extending xarray using accessors" for more information:
    https://docs.xarray.dev/en/stable/internals/extending-xarray.html

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._valid = None

    @property
    def valid(self):
        """Return flag for whether DataArray is validy yearly comppsite.

        Checks whether the object has the required dimension labels (as
        defined by/for project) and valid coordinate values. Will
        massage object inplace where possible to try to make valid.

        """
        if self._valid is None:
            if not set(self._obj.dims) == set(REQ_DIMS):
                self._valid = False

            # TODO: check for valid band coordinate names (`indices` needs
            # to be able to recognize them)
            # TODO: check that datetime frequency matches DATETIME_FREQ
            # TODO: this will fail/error "time" not in datetime. Allow for non-datetime indices?
            self._obj = self._obj.sortby("time")
            years = self._obj.time.dt.year.data

            if not np.all((years == list(range(min(years), max(years) + 1)))):
                self._valid = False
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
        """Check if stack contains year/year range.

        TODO: again, this func only works on datetime indices
        """
        required_years = datetime_to_index(years, list=True)
        for year in required_years:
            if not (pd.to_datetime(str(year)) in self._obj.coords["time"].values):
                return False
        return True

    def indices(self, indices_list) -> xr.DataArray:
        """Compute indices

        Parameters
        ----------
        indices_list : list of str
            The list of indices to compute/produce.

        Returns
        --------
        xr.DataArray
            A 4D (band, time, y, x) array with indices stacked
            along the band dimension.
        """
        indices_dict = {}
        for indice_input in indices_list:
            indice = Index(indice_input)
            indices_dict[indice] = indices_map[indice](self._obj)
        return stack_bands(indices_dict.values(), indices_dict.keys(), dim_name="band")
