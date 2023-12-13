"""Xarray accessor for timeseries operations. Plus helper functions."""

from typing import Union, Tuple
from datetime import datetime

import rioxarray

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import box
from spectral_recovery._config import DATETIME_FREQ, REQ_DIMS


def _datetime_to_index(
    value: Union[datetime, Tuple[datetime]], return_list: bool = False
) -> pd.DatetimeIndex:
    """Convert datetime or range of datetimes into pd.DatetimeIndex.

    Returns list if desired, otherwise returns DatetimeIndex.
    """
    if isinstance(value, (tuple, list)) and len(value) == 2:
        dt_range = pd.date_range(*value, freq=DATETIME_FREQ)
    else:
        try:
            if len(value) > 2:
                print(type(value))
                raise ValueError(
                    f"Passed value={value} but `datetime` must be a single Timestamp or"
                    " an iterable with exactly two Timestamps."
                )
            dt_range = pd.date_range(start=value[0], end=value[0], freq=DATETIME_FREQ)
        except TypeError:
            dt_range = pd.date_range(start=value, end=value, freq=DATETIME_FREQ)
    if not return_list:
        return dt_range
    return dt_range.to_list()


@xr.register_dataarray_accessor("satts")
class _SatelliteTimeSeries:
    """A accessor for operations commonly performed over
    a timeseries of satellite images stored in an xarray.DataArray.

    See "Extending xarray using accessors" for more information:
    https://docs.xarray.dev/en/stable/internals/extending-xarray.html

    Attributes
    ----------
    _obj : xarray.DataArray
        The xarray.DataArray to which this accessor is attached.
    _valid : bool
        Flag for whether the DataArray is a valid annual composite.

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._valid = None

    # TODO: change this method to "is_continuous" or something and only check
    # for continuity of years. Move the check for valid dim names to new method.
    @property
    def is_annual_composite(self) -> bool:
        """Check if DataArray is contains valid annual composites.

        Checks whether the object has the required dimension labels (as
        defined by/for project) and continuous years in the time dimension.

        Returns
        -------
        bool
            True if the DataArray is a valid annual composite, False otherwise.
        """
        if self._valid is None:
            years = self._obj.coords["time"].dt.year.values
            # TODO: catch value error ("not broadcastable") here, indicates missing years
            if not set(self._obj.dims) == set(REQ_DIMS):
                self._valid = False
            elif not np.all((years == list(range(min(years), max(years) + 1)))):
                self._valid = False
            else:
                self._valid = True
        return self._valid

    def contains_spatial(self, polygons: gpd.GeoDataFrame) -> bool:
        """Check if DataArray spatially contains polygons.

        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            The polygons to check if the DataArray spatially contains.

        Returns
        -------
        bool
            True if the DataArray spatially contains the polygons, False otherwise.
        """
        # NOTE: if this changes to looking at individual polygons
        # rather than the bbox of all polygons, consider this algo:
        # https://stackoverflow.com/questions/14697442/
        ext = box(*self._obj.rio.bounds())
        poly_ext = box(*polygons.total_bounds).buffer(-1e-14)
        if not ext.contains(poly_ext):
            # Permit bboxes that are almost equal (within 1e-14)
            return poly_ext.difference(ext).area < 1e-14
        return True

    def contains_temporal(self, years: Union[datetime, Tuple[datetime]]) -> bool:
        """Check if stack contains year/year range.

        Parameters
        ----------
        years : Union[datetime, Tuple[datetime]]
            The year or year range to check if the DataArray temporally contains.

        Returns
        -------
        bool
            True if the DataArray contains the year(s), False otherwise.
        """
        # Checking values against the Xarray datetime values is a bit
        # of a pain. With a bit of time and thought, the use of _datetime_to_index
        # could likely be avoided. Until then, this works.
        required_years = _datetime_to_index(years, return_list=True)
        for year in required_years:
            if not pd.to_datetime(str(year)) in self._obj.coords["time"].values:
                return False
        return True

    def stats(self) -> xr.DataArray:
        """Compute timeseries statistics.

        Reduces the object along the y and x dimensions. Reduction
        methods (stats methods) are applied sequentially, first along
        the y and then along the x dimension.

        Returns
        -------
        stats_xr : xr.DataArray
            A 3D DataArray containing time, band and stats dimensions.
            The computed statistics accessible as named coordinates in the
            "stats" dimension.
        """
        dims = ["y", "x"]
        stat_funcs = ["mean", "median", "max", "min", "std"]
        stats = {}
        for func_n in stat_funcs:
            func = getattr(self._obj, func_n)
            res = func(dim=dims, skipna=True)
            stats[func.__name__] = res

        stats_xr = xr.concat(stats.values(), dim=pd.Index(stats.keys(), name="stats"))
        return stats_xr
