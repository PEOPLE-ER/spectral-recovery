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
from enum import Enum

from restoration import RestorationArea
from multiband_yearly_stack import MultiBandYearlyStack

class Indices(Enum):
    NDVI = "NDVI"
    NBR = "NBR"

    def __str__(self) -> str:
        return self.name
    
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == value.upper():
                return member

def ndvi(stack):
    nir = stack.sel(band="nir") 
    red = stack.sel(band="red")
    ndvi = (nir - red) / (nir + red)
    print(ndvi)
    return ndvi

def nbr(stack):
    nir = stack.sel(band="nir") 
    swir = stack.sel(band="swir")
    nbr = (nir - swir) / (nir + swir)
    print(nbr)
    return nbr
     
indices_map = {
    Indices.NDVI: ndvi,
    Indices.NBR: nbr,
}

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
        metric_dict = compute_metrics(
            restoration_area,
            stack,
            [],
            ["ndvi", "nbr"]
            )
        return

# TODO: possibly organize this function into a class
def compute_metrics(restoration_area, stack, metrics, indices):
    if not stack.contains(restoration_area):
        print(
            f"Restoration polygon is not within temporal or spatial bounds "
            f"of the provided timeseries. Exiting..."
            )
    clipped_stack = stack.clip(restoration_area.restoration_polygon)
    indices_dict = {}
    for indice_input in indices:
        indice = Indices(indice_input)
        indices_dict[str(indice)] = indices_map[indice](clipped_stack)
    
    print(restoration_area.reference_system.baseline(indices_dict["NDVI"]))


if __name__ == "__main__":
    test_poly = gpd.read_file("../data/test_poly.gpkg")
    bad_poly = gpd.read_file("../data/out_of_bounds_poly.gpkg")
    rest_year = 2019
    reference_year = 2018

    test_stack = rioxarray.open_rasterio("../data/nir_18_19.tif",
                                             chunks="auto")
    print(test_stack)
    test_stack = test_stack.rename({"band":"time"})
    test_stack = test_stack.assign_coords(time=(pd.to_datetime(["2018"])))
    test_stack2 = test_stack.assign_coords(time=(pd.to_datetime(["2019"])))
    test_band_dict = {"nir": test_stack, "red": test_stack2, "swir": test_stack2}
    spectral_recovery(test_band_dict, test_poly, rest_year, reference_year)