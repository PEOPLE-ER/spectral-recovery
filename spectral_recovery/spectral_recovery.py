import os

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import pandas as pd
import images

from typing import Union, List, Dict
from enums import Index, Metric, BandCommon
from restoration import RestorationArea


def spectral_recovery(
    timeseries_dict: Dict[str | int, xr.DataArray | str],
    restoration_poly: gpd.GeoDataFrame | str,
    restoration_year: int,
    reference_range: Union[int, List[int]],
    indices_list: List[str],
    metrics_list: List[str],
    timeseries_range: List[int] = None,
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
    if isinstance(restoration_poly, str):
        restoration_poly = gpd.read_file(restoration_poly)
    timeseries = images.stack_from_files(timeseries_dict, data_mask, timeseries_range)
    print(timeseries)
    if not timeseries.yearcomp.valid:
        raise ValueError("Stack not a valid yearly composite stack.") 
    
    indices = timeseries.yearcomp.indices(indices_list)

    metrics = RestorationArea(
        restoration_polygon=restoration_poly,
        restoration_year=restoration_year,
        reference_system=reference_range,
        composite_stack=timeseries,
    ).metrics(metrics_list)
    metrics = metrics.compute()

    return metrics


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    import numpy as np 
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)

    rest_year = pd.to_datetime("2012")
    reference_year = pd.to_datetime("2008")

    metrics = spectral_recovery(
        timeseries_dict={
            BandCommon.red: "../test_recovered.tif",
            BandCommon.nir: "../test_recovered.tif",
        },
        timeseries_range=["2008", "2019"],
        restoration_poly="../1p_test.gpkg",
        restoration_year=rest_year,
        reference_range=reference_year,
        indices_list=[Index.ndvi],
        metrics_list=[Metric.percent_recovered],
    )
    print(metrics)
    # print(metrics.sel(metric="percent_recovered"))
