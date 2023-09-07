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


# TODO: simplify this function by grouping commands across multiple (3-4) smaller functions.
@click.command()
@click.argument("tif_dir", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "rest_poly",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "rest_year",
    type=click.DateTime(formats=["%Y"]),
    nargs=1,
)
@click.option(
    "-ref",
    "--ref-poly",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    help="Path to reference polygon(s).",
)
@click.argument(
    "ref_years",
    type=click.DateTime(formats=["%Y"]),
    nargs=2,
)
@click.option(
    "-i",
    "--indices",
    type=click.Choice(
        INDEX_CHOICE,
        case_sensitive=False,
    ),
    multiple=True,
    required=False,
    help="The indices on which to compute recovery metrics.",
)
@click.option(
    "-m",
    "--metrics",
    type=click.Choice(
        METRIC_CHOICE,
        case_sensitive=False,
    ),
    multiple=True,
    required=True,
    help="The recovery metrics to compute.",
)
@click.option(
    "--mask",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    help="Path to a data mask for annual composites.",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write output rasters.",
)
def cli(
    tif_dir: List[str],
    rest_poly: Path,
    rest_year: str,
    ref_poly: Path,
    ref_years: str,
    metrics: List[str],
    out: Path,
    indices: List[str] = None,
    mask: xr.DataArray = None,
) -> None:
    """Compute recovery metrics.

    This script will compute recovery METRICS over the area of REST_POLY
    using spectral data from annual composites in TIF_DIR. Recovery targets
    will be derived from REF_YEARS and restoration event occurs in REST_YEAR.

    \b
    TIF_DIR        Path to a directory containing annual composites.
    REST_POLY      Path to the restoration area polygon.
    REST_YEAR      Year of the restoration event.
    REF_YEARS      Start and end years over which to derive a recovery target.
    
    """
    # TODO: move user input prep/validation into own function?
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

    p = Path(tif_dir).glob("*.tif")
    tifs = [x for x in p if x.is_file()]

    with LocalCluster() as cluster, Client(cluster) as client:
        timeseries = read_and_stack_tifs(
            path_to_tifs=tifs,
            path_to_mask=mask,
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
