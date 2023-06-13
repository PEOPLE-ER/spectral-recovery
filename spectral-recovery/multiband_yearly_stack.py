from typing import Dict, Type, Union, List
from baseline import historic_average

import xarray as xr
import geopandas as gpd
import pandas as pd

from restoration import ReferenceSystem, RestorationArea
from shapely.geometry import box

class MultiBandYearlyStack():

    def __init__(
            self, 
            bands_dict: Dict[str, xr.DataArray], 
            data_mask: xr.DataArray = None
            ) -> None:
        # TODO: validate dimension names/format of dict
        self.stack = self._stack_bands(bands_dict)
        self.data_mask = data_mask

    def contains(self, restoration_area: Type[RestorationArea]):
        """ Check whether image stack contains the temporal and spatial
        attributes of a restoration area.
        """
        # otherwise,
        contains_flag = True
        # For now run each statement seperately so that we can know
        # which ones fail/if any. TODO: make this prettier?
        if not self.contains_spatial(restoration_area.restoration_polygon):
            contains_flag = False
        if not self.contains_temporal(restoration_area.restoration_year):
            contains_flag = False
        if isinstance(restoration_area.reference_system, ReferenceSystem):
            ref_years = restoration_area.reference_system.reference_range
            if not self.contains_temporal(ref_years):
                contains_flag = False
        else: 
            if not self.contains_temporal(restoration_area.reference_system):
                contains_flag = False

        return contains_flag

    def contains_spatial(self, polygons: gpd.GeoDataFrame):
        # NOTE: if this changes to looking at individual polygons
        # rather than the bbox of all polygons, consider this algo:
        # https://stackoverflow.com/questions/14697442/
        ext = box(*self.stack.rio.bounds())
        poly_ext = box(*polygons.total_bounds)
        if not ext.contains(poly_ext):
            return False
        return True
 
    def contains_temporal(self, years: Union[int, List[int]]):
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

    def mask(self):
        pass 

    def _stack_bands(self, bands_dict):
        """ Stack 3D dataArrays along new band dimension """
        stacked_bands = xr.concat(
            bands_dict.values(), 
            dim=pd.Index(bands_dict.keys(), name='band')
            )
        # TODO: add band/index names to band dimensions
        return stacked_bands

