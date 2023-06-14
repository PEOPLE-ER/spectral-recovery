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
from images import MultiBandYearlyStack
from metrics import percent_recovered, Metrics

def spectral_recovery(
        band_dict: Dict[str, xr.DataArray],
        restoration_poly: gpd.GeoDataFrame,
        restoration_year: int,
        reference_years: Union[int, List[int]],
        indices_list: List[str],
        metrics_list: List[str],
        data_mask: xr.DataArray = None
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
        indices = MultiBandYearlyStack(
             bands=band_dict,
             dict=True,
             data_mask=data_mask).indices(indices_list)
        # print(indices.stack)
        metrics = RestorationArea(
             restoration_polygon=restoration_poly,
             restoration_year=restoration_year,
             reference_system=reference_years,
             stack=indices
        ).metrics(metrics_list)
        return

# TODO: possibly organize this function into a class
def compute_metrics(restoration_area, stack, metrics, indices):
    if not stack.contains(restoration_area):
        print(
            f"Restoration polygon is not within temporal or spatial bounds of "
            f"the provided timeseries. Exiting..."
            )
    clipped_stack = stack.clip(restoration_area.restoration_polygon)
    indices_dict = {}
    for indice_input in indices:
        indice = Indices(indice_input)
        indices_dict[str(indice)] = indices_map[indice](clipped_stack)
    indices = MultiBandYearlyStack(indices_dict)
    indices.stack.sel(band="NDVI").rio.to_raster("ndvi_wrapped.tif")
    bl = restoration_area.baseline(indices.stack)
    total_change = restoration_area.change(indices.stack)
    curr
    # print(bl["baseline"].sel(band="NDVI").data)
    import dask.array as da
    pr = percent_recovered(indices.stack, bl["baseline"], event)
    # print(pr.sel(band="NBR").data.compute())
    # print(pr.sel(band="NDVI").data.compute())
    # pr.sel(band="NDVI").rio.to_raster("ndvi_pr.tif")
    # pr.sel(band="NBR").rio.to_raster("nbr_pr.tif")

    # metrics_dict = {}
    # for metrics_input in metrics:
    #     metric = Metrics(metrics_input)
    #     indices_dict[str(indice)] = metrics_map[metric](clipped_stack)


if __name__ == "__main__":
    test_poly = gpd.read_file("../../data/small_poly.gpkg")
    bad_poly = gpd.read_file("../../data/out_of_bounds_poly.gpkg")
    rest_year = pd.to_datetime("2018")
    reference_year = (pd.to_datetime("2018"),pd.to_datetime("2019"))

    test_stack = rioxarray.open_rasterio("../../data/nir_18_19.tif",
                                             chunks="auto")
    test_stack = xr.ones_like(test_stack)
    test_stack = test_stack.rename({"band":"time"})
    test_stack =  xr.concat(
            [test_stack,test_stack,test_stack], 
            dim=pd.Index(["2018", "2019", "2020"], name="time")
            )
    test_mask = xr.where(test_stack > 15515, True, False)
    test_stack = test_stack.assign_coords(time=(pd.to_datetime(["2018", "2019", "2020"])))
    test_stack2 = test_stack.assign_coords(time=(pd.to_datetime(["2018", "2019", "2020"])))

    # print(test_stack.rio.crs, test_stack.encoding, test_stack.attrs)

    test_band_dict = {"nir": test_stack, "red": test_stack2 * 2, "swir": test_stack2 * 3}
    spectral_recovery(
         band_dict=test_band_dict,
         restoration_poly=test_poly,
         restoration_year=rest_year,
         reference_years=reference_year,
         indices_list=["ndvi"],
         metrics_list=["percent_recovered"],
         )