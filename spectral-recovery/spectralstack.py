from typing import Dict, Type
from baseline import historic_average

import xarray as xr
import geopandas as gpd
import pandas as pd

from reference import ReferenceSystem, RestorationArea

class MultiYearlyStack():

    def __init__(
            self, 
            bands_dict: Dict[str, xr.DataArray], 
            data_mask: xr.DataArray = None,
    ) -> None:
        stacked_stack = self._stack_bands(bands_dict)
        self.stack = stacked_stack
        self.data_mask = data_mask

    def contains(self, polygons: Type[ReferenceSystem]):

        pass

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

