from typing import Dict, Type, Union, List
from baseline import historic_average

import xarray as xr
import geopandas as gpd
import pandas as pd

from reference import ReferenceSystem, RestorationArea
from shapely.geometry import box

class MultiBandYearlyStack():

    def __init__(
            self, 
            bands_dict: Dict[str, xr.DataArray], 
            data_mask: xr.DataArray = None
            ) -> None:
        # validate dimension names
        self.stack = self._stack_bands(bands_dict)
        self.data_mask = data_mask

    def contains(self, reference_system: Type[ReferenceSystem]):
        # NOTE: if this changes to looking at individual polygons
        # rather than the bbox of all polygons, consider this algo:
        # https://stackoverflow.com/questions/14697442/faster-way-of-polygon-intersection-with-shapely
        # otherwise,
        contains_flag = True
        if not self._contains_spatial(reference_system.polygons):
            contains_flag = False
        if not self._contains_temporal(reference_system.reference_range):
            contains_flag = False
        if isinstance(reference_system, RestorationArea):
            if not self._contains_temporal(reference_system.restoration_year):
                contains_flag = False
        return contains_flag

    def _contains_spatial(self, polygons: gpd.GeoDataFrame):
        ext = box(*self.stack.rio.bounds())
        poly_ext = box(*polygons.total_bounds)
        if not ext.contains(poly_ext):
            return False
        return True
        
    def _contains_temporal(self, years: Union[int, List[int]]):
        if isinstance(years, list) and len(years) > 1:
            required_years = range(years[0], years[1]+1)
        else: 
            required_years = [years]
        print(required_years)
        for year in required_years:
            if not (year in self.stack.coords['time']):
                return False
        return True

    def clip(self):
        pass
        
    
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

