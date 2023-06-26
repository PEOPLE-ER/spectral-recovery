import rioxarray
import dask
import os

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import images

from typing import Optional, Union, List, Dict

from restoration import RestorationArea
from metrics import percent_recovered, Metrics


def spectral_recovery(
    band_dict: Dict[str, xr.DataArray],
    restoration_poly: gpd.GeoDataFrame,
    restoration_year: int,
    reference_years: Union[int, List[int]],
    indices_list: List[str],
    metrics_list: List[str],
    data_mask: xr.DataArray = None,
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

    reference_years : int or list of int
    indices_list : list of str
    metrics_list : list of

    """
    timeseries = images.stack_from_files(band_dict, data_mask)
    if not timeseries.yearcomp.valid:
        raise ValueError("Stack not a valid yearly composite stack.")
    # indices = timeseries.yearcomp.indices(indices_list)
    metrics = RestorationArea(
        restoration_polygon=restoration_poly,
        restoration_year=restoration_year,
        reference_system=reference_years,
        composite_stack=timeseries,
    ).metrics(metrics_list)
    data = metrics.data.compute()

    # data = metrics.sel(metric="years_to_recovery").data.compute()
    print(data)
    # vals = ra.stack.sel(time=slice(ra.restoration_year,ra.end_year)).data.compute()
    # intercept =  ra.stack.sel(time=slice(ra.restoration_year)).squeeze().data.compute()
    # y_vals = intercept[0] + data[0]*vals
    # y_vals = y_vals[0].flatten()
    # x_vals = ra.stack["time"].dt.year.compute()[5:]

    # print(y_vals, x_vals)
    # plt.plot(x_vals, y_vals, '--')
    # plt.plot(x_vals, vals[0].flatten(), 'o-')
    # plt.show()

    return


if __name__ == "__main__":
    test_poly = gpd.read_file("../1p_test.gpkg")
    bad_poly = gpd.read_file("../../data/out_of_bounds_poly.gpkg")
    rest_year = pd.to_datetime("2012")
    reference_year = (pd.to_datetime("2009"), pd.to_datetime("2011"))

    test_stack = rioxarray.open_rasterio("../test_recovered_early.tif", chunks="auto")
    print(test_stack.band)
    test_stack = test_stack.rename({"band": "time"})
    # # test_mask = xr.where(test_stack > 15515, True, False)
    # test_stack = test_stack.assign_coords(
    #       time=(
    #             pd.to_datetime(["2008","2009","2011", "2010","2012","2013","2014", "2015","2016","2017","2018", "2019"])))

    # test_band_dict = {"NDVI":test_stack}
    # spectral_recovery(
    #      band_dict=test_band_dict,
    #      restoration_poly=bad_poly,
    #      restoration_year=rest_year,
    #      reference_years=reference_year,
    #      indices_list=["ndvi"],
    #      metrics_list=["percent_recovered", "years_to_recovery"],
    #      )
