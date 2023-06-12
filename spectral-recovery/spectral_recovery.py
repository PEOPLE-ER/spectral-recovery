import rioxarray
import dask
import os
os.environ['USE_PYGEOS'] = '0'
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd

from geopandas import GeoDataFrame
from typing import Optional, Union, List, Dict

from reference import RestorationArea
from spectralstack import MultiBandYearlyStack



def spectral_recovery(
        band_dict: Dict[str, xr.DataArray],
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
                 restoration_polygon=restoration_poly,
                 restoration_year=restoration_year,
                 reference_system=reference_range # just a historic time range
        )
        stack = MultiBandYearlyStack(band_dict)
        if not stack.contains(restoration_area):
            print(
                f"Restoration polygon is not within temporal or spatial bounds "
                f"of the provided timeseries. Exiting..."
                )
            return
        # clipped_stack = stack.clip(restoration_area)
        # for index in indices:
        #     index_array = getattr(spectral_recovery, )
        return

def compute_metrics():
      pass

def compute_index():
      pass


if __name__ == "__main__":
    test_poly = gpd.read_file("../data/test_poly.gpkg")
    bad_poly = gpd.read_file("../data/out_of_bounds_poly.gpkg")
    rest_year = 2019
    reference_year = [2018, 2019]

    test_stack = rioxarray.open_rasterio("../data/nir_18_19.tif",
                                             chunks="auto")
    test_stack = test_stack.rename({"band":"time"})
    test_stack = test_stack.assign_coords(time=([2018]))
    test_stack2 = test_stack.assign_coords(time=([2019]))
    test_band_dict = {"b_one": test_stack, "b_two": test_stack2}
    spectral_recovery(test_band_dict, test_poly, rest_year, reference_year)