import os

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import pandas as pd

from typing import Union, List, Dict
from spectral_recovery.images import stack_from_files
from spectral_recovery.enums import Index, Metric, BandCommon
from spectral_recovery.restoration import RestorationArea


def spectral_recovery(
    timeseries_dict: Dict[str | int, xr.DataArray | str],
    restoration_poly: gpd.GeoDataFrame | str,
    restoration_year: int,
    reference_range: Union[int, List[int]],
    metrics_list: List[str],
    indices_list: List[str] = None,
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
    timeseries = stack_from_files(timeseries_dict, data_mask, timeseries_range)
    if not timeseries.yearcomp.valid:
        raise ValueError("Stack not a valid yearly composite stack.")

    if indices_list is not None and len(indices_list) != 0:
        timeseries_for_metrics = timeseries.yearcomp.indices(indices_list)
    else:
        timeseries_for_metrics = timeseries

    metrics = RestorationArea(
        restoration_polygon=restoration_poly,
        restoration_year=restoration_year,
        reference_system=reference_range,
        composite_stack=timeseries_for_metrics,
    ).metrics(metrics_list)
    metrics = metrics.compute()

    return metrics


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster, progress

    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)

    rest_year = pd.to_datetime("2012")
    reference_year = pd.to_datetime("2008")

    metrics = spectral_recovery(
        timeseries_dict={
            Index.ndvi: "test_recovered.tif",
            Index.tcw: "test_recovered.tif",
        },
        timeseries_range=["2008", "2019"],
        restoration_poly="1p_test.gpkg",
        restoration_year=rest_year,
        reference_range=reference_year,
        # indices_list=[Index.ndvi, Index.sr],
        metrics_list=[Metric.percent_recovered],
    )
    progress(metrics)
    print(metrics)
    # print(metrics.sel(metric="percent_recovered"))
