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

def _datetime_to_index(
    value: Union[datetime, Tuple[datetime]], return_list: bool = False
) -> pd.DatetimeIndex:
    """Convert datetime or range of datetimes into pd.DatetimeIndex. Return as list if desired, otherwise returns DatetimeIndex."""
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

    # TODO: change this method to "is_continuous" or something and only check for continuity of years
    #   Move the check for valid dim names to a seperate method.
    @property
    def is_annual_composite(self) -> bool:
        """Check if DataArray is contains valid annual comppsites.

        Checks whether the object has the required dimension labels (as
        defined by/for project) and continuous years in the time dimension.

        Returns
        -------
        bool
            True if the DataArray is a valid annual composite, False otherwise.
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
        """Check if DataArray spatially contains polygons.
        
        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            The polygons to check if the DataArray spatially contains.

        Returns
        -------
        bool
            True if the DataArray contains the polygons, False otherwise.
        """
        # NOTE: if this changes to looking at individual polygons
        # rather than the bbox of all polygons, consider this algo:
        # https://stackoverflow.com/questions/14697442/
        ext = box(*self._obj.rio.bounds())
        poly_ext = box(*polygons.total_bounds)
        if not ext.contains(poly_ext):
            return False
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
        indices = xr.concat(
            indices_dict.values(), dim=pd.Index(indices_dict.keys(), name="band")
        )
        return indices

    def stats(self, dims, percentile = 0.8) -> xr.DataArray:
        """Compute statistics over a set of dimensions
        
        Parameters
        ----------
        dims : list of str
            The dimensions over which to compute statistics. Must be a 
            subset of the DataArray's dimensions.
        percentile : float
            The percentile to compute.

        Returns
        -------
        stats_xr : xr.DataArray
            A new DataArray with statistics stacked along 'stats'
            dimension.
        """
        stats = {}
        stats["mean"] = self._obj.mean(dim=dims, skipna=True)
        stats["max"] = self._obj.max(dim=dims, skipna=True)
        stats["min"] = self._obj.min(dim=dims, skipna=True)
        stats["median"] = self._obj.median(dim=dims, skipna=True)
        stats["quantile"] = self._obj.quantile(q=percentile, dim=dims, skipna=True)
        stats["std"] = self._obj.std(dim=dims, skipna=True)
        stats["sum"] = self._obj.std(dim=dims, skipna=True)
        stats_xr = xr.concat(stats.values(), dim=pd.Index(stats.keys(), name="stats"))
        return stats_xr

