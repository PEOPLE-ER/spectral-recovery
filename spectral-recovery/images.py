from typing import Dict, Type, Union, List, Tuple
from datetime import datetime

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

# from restoration import RestorationArea
from shapely.geometry import box
from indices import Indices, indices_map

DATETIME_FREQ = "YS"

class MultiBandYearlyStack():
    """ A multi-band stack of yearly composites """

    def __init__(
            self, 
            bands: Union[Dict[str, xr.DataArray], xr.DataArray],
            dict = False,
            data_mask: xr.DataArray = None,
            ) -> None:
        # TODO: validate format of bands, continuity of years
        # TODO: check that years are stored as DATETIME_FREQ
        if dict:
            stacked_stack = self.stack_bands(bands)
        else:
            stacked_stack = bands
        if data_mask is not None:
            self.stack = self.mask_stack(stacked_stack, data_mask)
        else:
            self.stack = stacked_stack

    # # TODO: remove this?
    # def contains(self, restoration_area: Type[RestorationArea]):
    #     """ Check if stack contains a RestorationArea.

    #     Method returns bool flag indicating whether the restoration
    #     polygons, reference polygons, restoration date/range, and 
    #     reference date/range are contained within the instance's stack.
    #     """
    #     if not (self.contains_spatial(
    #         restoration_area.restoration_polygon) and
    #             self.contains_temporal(
    #         restoration_area.restoration_year) and
    #             self.contains_temporal(
    #         restoration_area.reference_system.reference_range
    #         )):
    #         return False
    #     return True

    def contains_spatial(self, polygons: gpd.GeoDataFrame):
        """ Check if stack spatially contains polygons. """
        # NOTE: if this changes to looking at individual polygons
        # rather than the bbox of all polygons, consider this algo:
        # https://stackoverflow.com/questions/14697442/
        ext = box(*self.stack.rio.bounds())
        poly_ext = box(*polygons.total_bounds)
        if not ext.contains(poly_ext):
            return False
        return True
 
    def contains_temporal(self, years: datetime):
        """ Check if stack temporally contains year/year range. """
        required_years = self._datetime_to_index(years, list=True)
        # print(required_years)
        for year in required_years:
            if not (pd.to_datetime(str(year)) 
                    in self.stack.coords['time'].values):
                return False
        return True

    def clip(self, polygons: gpd.GeoDataFrame) -> xr.DataArray:
        # filter for relevant years?
        clipped_raster = self.stack.rio.clip(polygons.geometry.values)
        return MultiBandYearlyStack(clipped_raster, dict=False)
    
    def indices(self, indices_list):
        indices_dict = {}
        for indice_input in indices_list:
            indice = Indices(indice_input)
            indices_dict[str(indice)] = indices_map[indice](self.stack)
        # print(indices_dict)
        return MultiBandYearlyStack(indices_dict, dict=True)
    
    @staticmethod
    def _datetime_to_index(
        value: Union[datetime, Tuple[datetime]],
        list: bool=False
        ) -> pd.DatetimeIndex:
        """ Convert datetime or range of datetimes into pd.DatetimeIndex """
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

    @staticmethod
    def mask_stack(stack: xr.DataArray, mask: xr.DataArray, fill=np.nan):
        """ Mask a ND stack with 2D mask """
        # NOTE: should this allow more than 2D mask?
        mask = mask.squeeze(dim="band")
        masked_stack = stack.where(mask, fill)
        return masked_stack
    
    @staticmethod
    def stack_bands(bands_dict, dim_name="band"):
        """ Stack 3D image stacks to create 4D image stack """
        stacked_bands = xr.concat(
            bands_dict.values(), 
            dim=pd.Index(bands_dict.keys(), name=dim_name)
            )
        return stacked_bands

