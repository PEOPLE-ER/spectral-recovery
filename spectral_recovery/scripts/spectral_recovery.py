import os
from pathlib import Path
import click

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from typing import Union, List, Dict
from rasterio import merge

from spectral_recovery.timeseries import _stack_from_user_input
from spectral_recovery.enums import Index, Metric
from spectral_recovery.restoration import ReferenceSystem, RestorationArea
from spectral_recovery.io.raster import read_and_stack_tifs, metrics_to_tifs

# TODO: generalize "*_list" types to non-Enums.
@click.command()
@click.argument("tif_dir", type=click.Path(exists=True))
@click.option("--per-band", is_flag=True)
@click.option("--per-year", is_flag=True)
@click.argument("start_year", required=False)
@click.argument("end_year", required=False)
@click.argument("restoration_poly")
@click.argument("restoration_year")
@click.argumet("reference_poly", required=False)
@click.argument("reference_dates")
@click.indices("indices", required=False)
@click.argument("metrics")
@click.argument("mask_path", type=click.Path(exists=True), required=False)
@click.option("--write", is_flag=True, show_default=True, default=True)
def cli(
    tif_dir: List[str],
    per_band,
    per_year,
    restoration_poly: str,
    restoration_year: int,
    reference_dates: Union[int, List[int]],
    reference_poly: str,
    metrics: List[str],
    indices: List[str] = None,
    start_year: List[str] = None,
    end_year: List[str] = None,
    mask_path: xr.DataArray = None,
    write: bool = False,
) -> None:
    """The main calling function. Better doc-string is on the TO-DO.

    Parameters
    -----------
    tif_directory : list of str
        Directory containing tifs for timeseries.
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
    write : bool, optional
        Flag for whether to write recovery metrics to raster (TIF) or not.

    Returns
    -------
    metrics : xr.DataArray
        A 3D (metrics, y, x) DataArray of recovery metrics for the restoration
        area and period. NaN values represent data-gaps and/or undetermined metrics.
    """
    
    p = Path(tif_dir).glob('**/*')
    tifs = [x for x in p if x.is_file()]
    timeseries = read_and_stack_tifs(path_to_tifs=tifs,
                                     per_band=per_band,
                                     per_year=per_year,
                                     path_to_mask=data_mask,
                                     start_year=start_year,
                                     end_year=end_year
                                     )
    if not timeseries.satts.valid:
        raise ValueError("Stack is not a valid yearly composite stack.")

    if indices_list is not None and len(indices_list) != 0:
        timeseries_for_metrics = timeseries.satts.indices(indices_list)
    else:
        timeseries_for_metrics = timeseries
    if reference_poly is not None:
        if isinstance(restoration_poly, str):
            reference_poly_gdf = gpd.read_file(reference_poly)
        ref_sys = ReferenceSystem(
            reference_polygons=reference_poly_gdf,
            reference_stack=timeseries_for_metrics,
            reference_range=reference_dates,
            recovery_target_method=None,
        )
    else:
        ref_sys = reference_range
    metrics = RestorationArea(
        restoration_polygon=restoration_poly_gdf,
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

    rest_year = pd.to_datetime("2009")
    reference_year = pd.to_datetime("2007")

    print(
        "Tool currently only supports Landsat data. Please ensure any rasters passed in"
        " `timeseries_dic` are Landsat-derived.\n"
    )

    with LocalCluster() as cluster, Client(cluster) as client:
        metrics = spectral_recovery(
            timeseries_dict={
                Index.ndvi: "spectral_recovery/tests/test_data/time17_xy2_epsg3005.tif",
                Index.tcw: "spectral_recovery/tests/test_data/time17_xy2_epsg3005.tif",
            },
            timeseries_range=["2005", "2021"],
            restoration_poly=(
                "spectral_recovery/tests/test_data/polygon_inbound_epsg3005.gpkg"
            ),
            restoration_year=rest_year,
            reference_poly=(
                "spectral_recovery/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
            ),
            reference_range=reference_year,
            # indices_list=[Index.ndvi, Index.sr],
            metrics_list=[
                Metric.percent_recovered,
                Metric.years_to_recovery,
            ],
            write=False,
        ).compute()
        # TODO: figure out how to display progress to users
        # progress(metrics)
        print(metrics.compute())
