import os
from pathlib import Path
import click

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import pandas as pd

from typing import List
from dask.distributed import Client, LocalCluster

from spectral_recovery.enums import Index, Metric
from spectral_recovery.restoration import ReferenceSystem, RestorationArea
from spectral_recovery.io.raster import read_and_stack_tifs, metrics_to_tifs

INDEX_CHOICE = [i.value for i in Index]
METRIC_CHOICE = [str(m) for m in Metric]

@click.command()
@click.argument("tif_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--years", type=click.DateTime(formats=["%Y"]), nargs=2, required=True)
@click.option("--per-band", is_flag=True)
@click.option("--per-year", is_flag=True)
@click.option("-rest", "--rest-poly",type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--rest-year", type=click.DateTime(formats=["%Y"]), nargs=1, required=False)
@click.option("-ref","--ref-poly", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--ref-years", type=click.DateTime(formats=["%Y"]), nargs=2, required=True)
@click.option("-i","--indices", type=click.Choice(
    INDEX_CHOICE, 
    case_sensitive=False,
),
multiple=True,
required=False)
@click.option("-m","--metrics", 
type=click.Choice(
    METRIC_CHOICE, 
    case_sensitive=False,
),
multiple=True,
required=True)
@click.option("--mask", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def cli(
    tif_dir: List[str],
    years,
    per_band,
    per_year,
    rest_poly: str,
    rest_year: int,
    ref_poly,
    ref_years: str,
    metrics: List[str],
    out,
    indices: List[str] = None,
    mask: xr.DataArray = None,
) -> None:
    """ CLI-tool for computing recovery metrics over a desired restoration area. """
    start_year, end_year = years

    # TODO: move user input prep/validation into own function?
    start_year = pd.to_datetime(start_year)
    end_year = pd.to_datetime(end_year)
    rest_year = pd.to_datetime(rest_year)
    ref_years = [pd.to_datetime(ref_years[0]), pd.to_datetime(ref_years[1])]
    try:
        valid_metrics = [Metric[name] for name in metrics]
    except KeyError as e:
        raise e from None
    
    try:
        valid_indices = [Index[name.lower()] for name in indices]
    except KeyError as e:
        raise e from None
        
    p = Path(tif_dir).glob('*.tif')
    tifs = [x for x in p if x.is_file()]

    with LocalCluster() as cluster, Client(cluster) as client:
        timeseries = read_and_stack_tifs(path_to_tifs=tifs,
                                        per_band=per_band,
                                        per_year=per_year,
                                        path_to_mask=mask,
                                        start_year=start_year,
                                        end_year=end_year
                                        )

        if not timeseries.satts.valid:
            raise ValueError("Stack is not a valid yearly composite stack.")

        if indices is not None and len(indices) != 0:
            timeseries_for_metrics = timeseries.satts.indices(valid_indices)
        else:
            timeseries_for_metrics = timeseries
        if ref_poly is not None:
            reference_poly_gdf = gpd.read_file(ref_poly)
            ref_sys = ReferenceSystem(
                reference_polygons=reference_poly_gdf,
                reference_stack=timeseries_for_metrics,
                reference_range=ref_years,
                recovery_target_method=None,
            )
        else:
            ref_sys = ref_years
        
        try:
            valid_metrics = [Metric[name] for name in metrics]
        except ValueError as e:
            raise e from None
        
        rest_poly_gdf = gpd.read_file(rest_poly)
        metrics_array = RestorationArea(
            restoration_polygon=rest_poly_gdf,
            restoration_year=rest_year,
            reference_system=ref_sys,
            composite_stack=timeseries_for_metrics,
        ).metrics(valid_metrics)

        out.mkdir(parents=True, exist_ok=True)
        metrics_to_tifs(
            metrics_array=metrics_array,
            out_dir=out,
        )
    return metrics


# if __name__ == "__main__":
#     , progress

#     rest_year = pd.to_datetime("2009")
#     reference_year = pd.to_datetime("2007")

#     print(
#         "Tool currently only supports Landsat data. Please ensure any rasters passed in"
#         " `timeseries_dic` are Landsat-derived.\n"
#     )

#     with LocalCluster() as cluster, Client(cluster) as client:
#         metrics = spectral_recovery(
#             timeseries_dict={
#                 Index.ndvi: "spectral_recovery/tests/test_data/time17_xy2_epsg3005.tif",
#                 Index.tcw: "spectral_recovery/tests/test_data/time17_xy2_epsg3005.tif",
#             },
#             timeseries_range=["2005", "2021"],
#             restoration_poly=(
#                 "spectral_recovery/tests/test_data/polygon_inbound_epsg3005.gpkg"
#             ),
#             restoration_year=rest_year,
#             reference_poly=(
#                 "spectral_recovery/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
#             ),
#             reference_range=reference_year,
#             # indices_list=[Index.ndvi, Index.sr],
#             metrics_list=[
#                 Metric.percent_recovered,
#                 Metric.years_to_recovery,
#             ],
#             write=False,
#         ).compute()
#         # TODO: figure out how to display progress to users
#         # progress(metrics)
#         print(metrics.compute())
