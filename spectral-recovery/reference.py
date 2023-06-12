
from typing import Callable, Optional, Union, List
from baseline import historic_average

import xarray as xr
import geopandas as gpd

class ReferenceSystem():
    """ Encapsulates data and methods related to reference areas
    
    Attributes
    -----------

    """
    def __init__(
            self,
            reference_polygons: gpd.GeoDataFrame,
            reference_range: Union[int, List[int]],
            baseline_method: Callable=None
            ) -> None:
        
        self.reference_polygons = reference_polygons
        self.reference_range = reference_range
        self.baseline_method = baseline_method or historic_average
    
    # TODO: some method related to getting bounding boxes

class RestorationArea():
    """ Encapsulates data and methods related to reference areas. 
    Allows optional composition of ReferenceSystem class.

    Attributes
    -----------

    """
    def __init__(
            self,
            restoration_polygon: gpd.GeoDataFrame, 
            restoration_year: int,
            reference_system: Union[int, List[int], ReferenceSystem],
            ) -> None:
        
        if not self._single_restoration_area(restoration_polygon):
            return ValueError(
                f"restoration_polygons contains more than one Polygon."
                f"A RestorationArea instance can only contain one Polygon." )
        
        self.restoration_polygon = restoration_polygon
        self.restoration_year = restoration_year
        self.reference_system = reference_system
            
    def _single_restoration_area(self, restoration_polygons):
        if restoration_polygons.shape[0] != 1:
            return False
        return True
    




