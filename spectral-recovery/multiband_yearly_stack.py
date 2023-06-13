from typing import Dict, Type, Union, List
from baselines import historic_average

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from restoration import RestorationArea
from shapely.geometry import box

class MultiBandYearlyStack():

    def __init__(
            self, 
            bands_dict: Dict[str, xr.DataArray], 
            data_mask: xr.DataArray = None
            ) -> None:
        # TODO: validate format of bands
        stacked_stack = self.stack_bands(bands_dict)
        if data_mask is not None:
            self.stack = self.mask_stack(stacked_stack, data_mask)
        else:
            self.stack = stacked_stack

    def contains(self, restoration_area: Type[RestorationArea]):
        """ Check if stack contains a RestorationArea.

        Method returns bool flag indicating whether the restoration
        polygons, reference polygons, restoration date/range, and 
        reference date/range are contained within the instance's stack.
        """
        if not (self.contains_spatial(restoration_area.restoration_polygon) and
                self.contains_temporal(restoration_area.restoration_year) and
                self.contains_temporal(
            restoration_area.reference_system.reference_range
            )):
            return False
        return True

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
 
    def contains_temporal(self, years: Union[int, List[int]]):
        """ Check if stack temporally contains year/year range. """
        if isinstance(years, list) and len(years) > 1:
            required_years = range(years[0], years[1]+1)
        else: 
            required_years = [years]
        for year in required_years:
            if not (pd.to_datetime(str(year)) 
                    in self.stack.coords['time'].values):
                return False
        return True

    def clip(self, polygons: gpd.GeoDataFrame) -> xr.DataArray:
        # filter for relevant years?
        clipped_raster = self.stack.rio.clip(polygons.geometry.values)
        return clipped_raster

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

