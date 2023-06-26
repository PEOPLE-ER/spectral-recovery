import rioxarray

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from typing import Dict, Type, Union, List, Tuple
from datetime import datetime
from shapely.geometry import box
from indices import Indices, indices_map

DATETIME_FREQ = "YS"
REQ_DIMS = ["band", "time", "y", "x"]


def stack_from_files(band_dict, timeseries_range, mask):
    """ """
    if all([isinstance(band, str) for band in band_dict.values()]):  # Read TIFs
        read_bands = {}
        for name, file in band_dict.items():
            with rioxarray.open_rasterio(file, chunks="auto") as band:
                band = band.rename({"band": "time"})
                read_bands[name] = band
        stacked_bands = stack_bands(read_bands.values(), read_bands.keys())
    else:
        stacked_bands = stack_bands(band_dict.values(), band_dict.keys())

    years = [str(e) for e in np.arange(timeseries_range[0], timeseries_range[1] + 1)]
    stacked_bands = stacked_bands.assign_coords(time=(pd.to_datetime(years)))
    print(stacked_bands)
    if mask:
        masked_bands = mask_stack(stacked_bands, mask)
        return masked_bands
    else:
        return stacked_bands


def stack_bands(bands, names, dim_name="band"):
    """Stack 3D image stacks to create 4D image stack"""
    # TODO: handle band dimension/coordinate errors
    stacked_bands = xr.concat(bands, dim=pd.Index(names, name=dim_name))
    return stacked_bands


def mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan):
    """Mask a ND stack with 2D mask"""
    # TODO: should this allow more than 2D mask?
    mask = mask.squeeze(dim="band")
    masked_stack = stack.where(mask, fill)
    return masked_stack


def datetime_to_index(
    value: Union[datetime, Tuple[datetime]], list: bool = False
) -> pd.DatetimeIndex:
    """Convert datetime or range of datetimes into pd.DatetimeIndex

    For ease of indexing through a DataArray object
    """
    if isinstance(value, tuple):
        dt_range = pd.date_range(start=value[0], end=value[1], freq=DATETIME_FREQ)
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
            self._obj = self._obj.sortby("time")
            years = self._obj.time.dt.year.data

            # NOTE: this wont work if the time coords aren't 199X/20XX
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
        """Check if stack contains year/year range."""
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

        """
        indices_dict = {}
        for indice_input in indices_list:
            indice = Indices(indice_input)
            indices_dict[str(indice)] = indices_map[indice](self._obj)
        return stack_bands(indices_dict.values(), indices_dict.keys())
