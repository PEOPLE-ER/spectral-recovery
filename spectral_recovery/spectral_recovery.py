import os

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from typing import Union, List, Dict
from rasterio import merge
from spectral_recovery.timeseries import _stack_from_user_input
from spectral_recovery.enums import Index, Metric, BandCommon
from spectral_recovery.restoration import ReferenceSystem, RestorationArea
from rasterio._err import CPLE_AppDefinedError


# TODO: generalize "*_list" types to non-Enums.
def spectral_recovery(
    timeseries_dict: Dict[str | int, xr.DataArray | str],
    restoration_poly: gpd.GeoDataFrame | str,
    restoration_year: int,
    reference_range: Union[int, List[int]],
    reference_poly: gpd.GeoDataFrame | str,
    metrics_list: List[Metric],
    indices_list: List[Index] = None,
    timeseries_range: List[str] = None,
    data_mask: xr.DataArray = None,
    write: bool = False,
) -> None:
    """The main calling function. Better doc-string is on the TO-DO.

    Parameters
    -----------
    timeseries_dict : Dict of str or xr.DataArray
        Dictionary of paths to or DataArrays of per-pixel timeseries data.
        The dictionary is expected to contain band rasters (i.e a single
        band over time), indices rasters, or year rasters (i.e multiple-bands
        for a given year), with keys being the respective band name or year for
        the given raster item. Keys for band rasters must be the common name.
        Rasters will be stacked along new 4th dimension, so dimensions must match.
    restoration_poly :
        Path to vector file containing a Polygon of restoration area.
    restoration_year :
        Year of restoration event.
    reference_range :
        Year or year(s) from which to derive the reference/recovery target value.
    metrics_list : str
        List of recovery metrics to compute.
    indices_list : list of str, optional
        List of indices to compute with the data provided in `timeseries_dict`.
        If given, recovery metrics will be compute over all indices. If not, then
        recovery metrics are computed using the data passed with `timeseries_dict`.
    timeseries_range : list of str, optional
        The year range of the timeseries data in `timeseries_dict`. Must be provided
        if `timeseries_dict` contains per-band/indice data.
    data_mask : str or xr.DataArray, optional
        Path to or DataArray of mask. Must be broadcastable to dim of size (N,M,y,x)
        where N is # of bands/indices and M is # of timesteps.

    Returns
    -------
    metrics : xr.DataArray
        A 3D (metrics, y, x) DataArray of recovery metrics for the restoration
        area and period. NaN values represent data-gaps and/or undetermined metrics.
    """
    # TODO: check that the out file names don't already exist... abort early if they do
    if isinstance(restoration_poly, str):
        restoration_poly = gpd.read_file(restoration_poly)
    timeseries = _stack_from_user_input(timeseries_dict, data_mask, timeseries_range)
    if not timeseries.yearcomp.valid:
        raise ValueError("Stack is not a valid yearly composite stack.")

    if indices_list is not None and len(indices_list) != 0:
        timeseries_for_metrics = timeseries.yearcomp.indices(indices_list)
    else:
        timeseries_for_metrics = timeseries
    if reference_poly is not None:
        ref_sys = ReferenceSystem(
            reference_polygons=reference_poly,
            reference_range=reference_range,
            baseline_method=None,
            variation_method=None
        )
    else:
        ref_sys = reference_range
    metrics = RestorationArea(
        restoration_polygon=restoration_poly,
        restoration_year=restoration_year,
        reference_system=ref_sys,
        composite_stack=timeseries_for_metrics,
    ).metrics(metrics_list)

    if write:
        out_raster = xr.full_like(timeseries_for_metrics[0, 0, :, :], np.nan)
        for metric in metrics["metric"].values:
            xa_dataset = xr.Dataset()
            for band in metrics["band"].values:
                out_metric = metrics.sel(metric=metric, band=band)
                # NOTE: This takes non-null values between each raster so `out_raster` (the full AOI) must be all null
                merged = out_metric.combine_first(out_raster)
                xa_dataset[str(band)] = merged
                try:
                    out_name = f"{metric!s}.tif"
                    xa_dataset.rio.to_raster(raster_path=out_name)
                except CPLE_AppDefinedError as exc:
                    raise PermissionError(
                        f"Could not write output to {out_name} because a TIF already"
                        " exists and could not be overwritten, likely because"
                        " it's open elsewhere. Is the existing TIF open in an"
                        " application (e.g QGIS)? If so, try closing it before your"
                        " next run to avoid this error."
                    ) from None
    return metrics


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster, progress

    rest_year = pd.to_datetime("2010")
    reference_year = pd.to_datetime("2007")

    print(
        "Tool currently only supports Landsat data. Please ensure any rasters passed in"
        " `timeseries_dic` are Landsat-derived.\n"
    )

    with LocalCluster() as cluster, Client(cluster) as client:
        metrics = spectral_recovery(
            timeseries_dict={
                Index.ndvi: "test_500.tif",
                Index.tcw: "test_500.tif",
            },
            timeseries_range=["2006", "2019"],
            restoration_poly="test_200.gpkg",
            restoration_year=rest_year,
            # reference_poly="reference_polys_3.gpkg",
            reference_range=reference_year,
            # indices_list=[Index.ndvi, Index.sr],
            metrics_list=[
                Metric.percent_recovered,
                Metric.years_to_recovery,
                Metric.recovery_indicator,
                Metric.dNBR,
            ],
            write=True,
        )
        # TODO: figure out how to display progress to users
        # progress(metrics)
        print(metrics.compute())
