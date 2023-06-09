import rioxarray
import dask
import os
os.environ['USE_PYGEOS'] = '0'
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd

from geopandas import GeoDataFrame
from typing import Optional, Union, List

from reference import RestorationArea
from spectralstack import MultiYearlyStack



def spectral_recovery(
        restoration_poly: Union[str, gpd.GeoDataFrame],
        restoration_year: int,
        reference_range: Union[int, List[int]]
        ) -> None:
        """
        The main calling function. Better doc-string incoming. 

        Parameters
        -----------
        restoration_poly : str or Polygon
            Path to vector file or Polygon object with polygon
            representing restoration area

        restoration_year :
            Year of restoration event.

        """
        # Read file if given filename
        if isinstance(restoration_poly, str):
            restoration_poly = gpd.read_file(restoration_poly)
            # TODO: check if individual polygon given? even though it's checked 
            # at initializiation of RestorationArea instance?      
        restoration_area = RestorationArea(
                 restoration_polygons=restoration_poly,
                 restoration_year=restoration_year,
                 reference_range=reference_range
                )
        
        test_stack = rioxarray.open_rasterio("../data/nir_18_19.tif",
                                             chunks="auto")
        print(test_stack)
        stack = MultiYearlyStack({"b_one": test_stack, "b_two": test_stack})
        if not stack.contains(restoration_area):
            return ValueError(
                f"Restoration polygon is not within temporal or spatial bounds "
                f"of the provided timeseries."
                )
        clipped_stack = stack.clip(restoration_area)
        # for index in indices:
        #     index_array = getattr(spectral_recovery, )
        return

def compute_metrics():
      pass

def compute_index():
      pass


if __name__ == "__main__":
      test_poly = gpd.read_file("../data/test_poly.gpkg")
      rest_year = 2019
      reference_year = 2018 
      spectral_recovery(test_poly, rest_year, reference_year)