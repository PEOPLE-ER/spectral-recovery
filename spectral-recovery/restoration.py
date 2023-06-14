
from typing import Callable, Optional, Union, List
from baselines import historic_average
from utils import to_datetime

import xarray as xr
import geopandas as gpd


class ReferenceSystem():
    """ Encapsulates data and methods related to reference areas.
    
    Attributes
    -----------
    polygons : gpd.GeoDataframe
        A GeoDataframe containing at least one Polygon representing
        the collective area/ecosystems in the reference system
    reference_years : Tuple of datetimes
        The year or range of years to consider as reference
    baseline_method : Callable
        The method of computing the baseline within the reference system
    variation_method : Callable
        THe method for characterizing baseline variation within the 
        reference system. Default is None.

    
    """
    def __init__(
            self,
            reference_polygons: gpd.GeoDataFrame,
            reference_range: Union[int, List[int]],
            baseline_method: Callable=None,
            variation_method: Callable=None
            ) -> None:
        # TODO: convert date inputs into standard form (pd.dt?)
        self.reference_polygons = reference_polygons
        self.reference_range = to_datetime(reference_range)
        self.baseline_method = baseline_method or historic_average
        self.variation_method = variation_method
    
    def baseline(self, stack):
        baseline = self.baseline_method(stack, self.reference_range)
        if self.variation_method is not None:
            variation = self.variation_method(stack, self.reference_range)
            return {"baseline": baseline, "variation": variation}
        return {"baseline": baseline}
    # TODO: some method related to getting bounding boxes


class RestorationArea():
    """ Encapsulates data and methods related to restoration areas. 

    Attributes
    -----------

    """
    def __init__(
            self,
            restoration_polygon: gpd.GeoDataFrame, 
            restoration_year: int,
            reference_system: Union[int, List[int], ReferenceSystem],
            ) -> None:
        
        if restoration_polygon.shape[0] != 1:
            return ValueError(
                f"restoration_polygons contains more than one Polygon."
                f"A RestorationArea instance can only contain one Polygon." )
        
        self.restoration_polygon = restoration_polygon
        self.restoration_year = to_datetime(restoration_year)
        if not isinstance(reference_system, ReferenceSystem):
            # If ReferenceSystem not provided, create historic reference.
            # A historic reference system is a historic average over
            # the restoration area polygon for a specified year range.
            historic_reference = ReferenceSystem(
                reference_polygons=restoration_polygon,
                reference_range=reference_system
                )
            self.reference_system = historic_reference
        else:
            self.reference_system = reference_system
    
    def baseline(self, stack):
        return self.reference_system.baseline(stack)