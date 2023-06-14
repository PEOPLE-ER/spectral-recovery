
import xarray as xr
import geopandas as gpd

from typing import Callable, Optional, Union, List
from baselines import historic_average
from utils import to_datetime
from images import MultiBandYearlyStack
from datetime import datetime
from metrics import Metrics, percent_recovered


class ReferenceSystem():
    # TODO: split date into start and end dates. 
    # If no end date then same as start

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
            restoration_year: Union[str, datetime],
            reference_system: Union[int, List[int], ReferenceSystem],
            stack: MultiBandYearlyStack
            ) -> None:
        
        if restoration_polygon.shape[0] != 1:
            raise ValueError(
                f"restoration_polygons contains more than one Polygon."
                f"A RestorationArea instance can only contain one Polygon." )
        
        self.restoration_polygon = restoration_polygon
        self.restoration_year = to_datetime(restoration_year)

        if not isinstance(reference_system, ReferenceSystem):
            historic_reference = ReferenceSystem(
                reference_polygons=restoration_polygon,
                reference_range=reference_system
                ) 
            self.reference_system = historic_reference
        else:
            self.reference_system = reference_system

        # TODO: Move _within/contains to clip?
        if not self._within(stack):
            raise ValueError(
                f"Restoration not contained by stack. Better message soon!" )
        # print(stack)
        self.stack = stack.clip(self.restoration_polygon)
    

    def _within(self, stack):
        if not (stack.contains_spatial(self.restoration_polygon) and
                    stack.contains_temporal(self.restoration_year) and
                        stack.contains_temporal(
            self.reference_system.reference_range
            )):
            return  False
        return True
    

    def metrics(self, metrics):
        """ Generate recovery metrics over restoration area """
        metrics_dict = {}
        for metrics_input in metrics:
            metric = Metrics(metrics_input)
            try:
                metric_func = getattr(self, f"_{metric}")
            except Exception:
                raise ValueError(f"{metric} not implemented")
            metrics_dict[str(metric)] = metric_func()
            MultiBandYearlyStack.stack_bands(metrics_dict,
                                             dim_name="metric")
            
    def _percent_recovered(self) -> xr.DataArray:
        curr = self.stack.sel(time=self.restoration_year)
        baseline = self.reference_system.baseline(self.stack)
        event = self.stack.sel(time=self.restoration_year)
        return percent_recovered(curr, baseline, event)