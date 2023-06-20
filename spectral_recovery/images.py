from typing import Dict, Type, Union, List, Tuple
from datetime import datetime

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import box
from indices import Indices, indices_map

DATETIME_FREQ = "YS"
REQ_DIMS = ["band", "time", "y", "x"]

def stack_from_files(band_dict, mask):
    """
    """
    # TODO: handle reading files. rn assumes it's given xr structures
    stacked = stack_bands(band_dict)
    if mask:
        masked = mask_stack(stacked, mask)
        return masked
    else:
        return stacked
    
def stack_bands(bands_dict, dim_name="band"):
    """ Stack 3D image stacks to create 4D image stack """
    # TODO: handle band dimension/coordinate errors
    stacked_bands = xr.concat(
        bands_dict.values(), 
        dim=pd.Index(bands_dict.keys(), name=dim_name)
        )
    return stacked_bands

def mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan):
    """ Mask a ND stack with 2D mask """
    # TODO: should this allow more than 2D mask?
    mask = mask.squeeze(dim="band")
    masked_stack = stack.where(mask, fill)
    return masked_stack

def datetime_to_index(
    value: Union[datetime, Tuple[datetime]],
    list: bool=False
    ) -> pd.DatetimeIndex:
    """ Convert datetime or range of datetimes into pd.DatetimeIndex 
    
    For ease of indexing through a DataArray object
    """
    if isinstance(value, tuple):
        dt_range = pd.date_range(
            start=value[0],
            end=value[1], 
            freq=DATETIME_FREQ
            )
    else:
        dt_range = pd.date_range(
            start=value,
            end=value, 
            freq=DATETIME_FREQ
            )
    if not list:
        return dt_range
    return dt_range.to_list()


@xr.register_dataarray_accessor("yearcomp")
class YearlyCompositeAccessor:
    """ An accessor on the xarray DataArray object.
    
    For methods related to yearly composite timeseries as well as
    general image stack operations.
    
    See "Extending xarray using accessors" for more information: 
    https://docs.xarray.dev/en/stable/internals/extending-xarray.html
    
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._valid = self._is_valid()

    @property
    def valid(self):
        return self._valid
    
    def _is_valid(self) -> bool:
        """ Check whether an xarray object is a valid yearly composite.

        Check if the object has the required dimension labels (as
        defined within project) and valid coordinate values. Will
        massage object inplace where possible to try to make valid.  

        """
        if not set(self._obj.dims) == set(REQ_DIMS):
            return False
        
        # TODO: check for valid band coordinate names (`indices` needs 
        # to be able to recognize them)

        self._obj = self._obj.sortby("time")
        years = self._obj.time.dt.year.data
        print(years)

        if not np.all((years == list(range(min(years), max(years)+1)))):
            return False

        return True
    
    def contains_spatial(self, polygons: gpd.GeoDataFrame) -> bool:
        """ Check if stack contains polygons. """
        # NOTE: if this changes to looking at individual polygons
        # rather than the bbox of all polygons, consider this algo:
        # https://stackoverflow.com/questions/14697442/
        ext = box(*self._obj.rio.bounds())
        poly_ext = box(*polygons.total_bounds)
        if not ext.contains(poly_ext):
            return False
        return True

    def contains_temporal(self, years: datetime) -> bool:
        """ Check if stack contains year/year range. """
        required_years = datetime_to_index(years, list=True)
        for year in required_years:
            if not (pd.to_datetime(str(year)) 
                    in self._obj.coords['time'].values):
                return False
        return True

    def indices(self, indices_list) -> xr.DataArray:
        """ Compute indices 
        
        Parameters
        ----------
        indices_list : list of str
            The list of indices to compute/produce.
        
        """
        indices_dict = {}
        for indice_input in indices_list:
            indice = Indices(indice_input)
            indices_dict[str(indice)] = indices_map[indice](self._obj)
        return stack_bands(indices_dict)