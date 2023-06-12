
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
            restoration_polygons: gpd.GeoDataFrame,
            reference_range: Union[int, List[int]],
            baseline_method: Callable=None ) -> None:
        
        self.polygons = restoration_polygons
        self.reference_range = reference_range
        self.baseline_method = baseline_method or historic_average
    
    # TODO: some method related to getting bounding boxes

# NOTE: does this sub-class really need to exist?
class RestorationArea(ReferenceSystem):
    """ A sub-class of ReferenceSystem. Defines a single area that has
    been subject to a restoration/disturbance event.

    Attributes
    -----------

    """
    def __init__(self, restoration_year: int, *args, **kwargs) -> None:
        self.restoration_year = restoration_year
        if self._single_restoration_area(kwargs['restoration_polygons']):
            super().__init__(*args, **kwargs)
        else:
            return ValueError(
                f"restoration_polygons contains more than one Polygon."
                f"A RestorationArea instance can only contain one Polygon." )
    
    def _single_restoration_area(self, restoration_polygons):
        if restoration_polygons.shape[0] != 1:
            return False
        return True
    




