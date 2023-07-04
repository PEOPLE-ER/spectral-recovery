import os

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import pandas as pd
import images

from typing import Union, List, Dict
from restoration import RestorationArea


def spectral_recovery(
    band_dict: Dict[str, xr.DataArray | str],
    restoration_poly: gpd.GeoDataFrame | str,
    timeseries_range: List[int],
    restoration_year: int,
    reference_range: Union[int, List[int]],
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
    if isinstance(restoration_poly, str):
        restoration_poly = gpd.read_file(restoration_poly)

    timeseries = images.stack_from_files(band_dict, timeseries_range, data_mask)
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

    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)

    rest_year = pd.to_datetime("2012")
    reference_year = (pd.to_datetime("2009"), pd.to_datetime("2011"))

    metrics = spectral_recovery(
        band_dict={
            "nir": "../test_500.tif",
            "red": "../test_500.tif",
        },
        restoration_poly="../test_500.gpkg",
        timeseries_range=[2008, 2021],
        restoration_year=rest_year,
        reference_range=reference_year,
        indices_list=["ndvi"],
        metrics_list=["percent_recovered", "years_to_recovery"],
    )
    print(metrics.sel(metric="percent_recovered"))
