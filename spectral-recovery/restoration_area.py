
from typing import Union
import xarray as xr

from shapely.geometry import Polygon

class RestorationArea():
    """ Encapsulates data and methods related to restoration areas """

    def __init__(self, restoration_polygon, year) -> None:
        if not isinstance(restoration_polygon, Polygon):
            return ValueError(
                f"Restoration area must be shapely.geometry.Polygon type, "
                f"recieved {type(restoration_polygon)}"
                )
        self.restoration_polygon = restoration_polygon
        self.restoration_year = year

    def clip_from_stack(self, stack, time_range=None):
        """ Clip band data from timeseries within bbox """
        # clip with rioxarray.clip or get lat/lon from bbox and clip with .sel?
        print("testing stack")

    
class ReferenceSystem():
    """ Encapsulates data and method related to reference areas """
    def __init__(self, reference_polygons, time_range=None) -> None:
        self.restoration_polygon = reference_polygons
        self.restoration_year = time_range

    def clip_from_stack(self, stack, time_range=None):
        """ Clip band data from timeseries within bbox """
        # clip with rioxarray.clip or get lat/lon from bbox and clip with .sel?
        print("testing stack")


class RecoveryMetrics():
    """ Encapsulates data and method related to reference areas """
    def __init__(self, reference_polygons, time_range=None) -> None:
        self.restoration_polygon = reference_polygons
        self.restoration_year = time_range

    def clip_from_stack(self, stack, time_range=None):
        """ Clip band data from timeseries within bbox """
        # clip with rioxarray.clip or get lat/lon from bbox and clip with .sel?
        print("testing stack")