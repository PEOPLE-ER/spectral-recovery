import os
from pathlib import Path
import click

os.environ["USE_PYGEOS"] = "0"
import xarray as xr
import geopandas as gpd
import pandas as pd

from typing import List
from dask.distributed import Client, LocalCluster

from spectral_recovery.enums import Metric
from spectral_recovery.restoration import RestorationArea
from spectral_recovery.indices import compute_indices
from spectral_recovery.io.raster import read_timeseries, _metrics_to_tifs
from spectral_recovery.io.polygon import read_reference_polygons, read_restoration_polygons

METRIC_CHOICE = [str(m) for m in Metric]

# NOTE: multi-year disturbances are not implemented in CLI yet. Users cannot provide disturbance AND restoration years.


@click.group(chain=True)
@click.argument("tif_dir", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "out",
    type=click.Path(path_type=Path),
)
@click.argument(
    "rest_poly",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-ref",
    "--reference",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to reference polygon(s) with dates."
)
@click.option(
    "-i",
    "--indices",
    multiple=True,
    required=True,
    help="The indices on which to compute recovery metrics.",
)
@click.option(
    "--mask",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    help="Path to a data mask for annual composites.",
)
@click.pass_context
def cli(
    ctx,
    tif_dir: List[str],
    rest_poly: Path,
    out: Path,
    reference: Path = None,
    indices: List[str] = None,
    mask: xr.DataArray = None,
) -> None:
    """Compute recovery metrics.

    This script will compute recovery metrics over the area of REST_POLY
    using spectral data from annual composites in TIF_DIR. Recovery targets
    will be derived over REF_POLYGON between reference start and end year.

    \b
    TIF_DIR        Path to a directory containing annual composites.
    OUT            Path to directory to write output rasters.
    REST_POLY      Path to the restoration area polygon.
    REFERNCE       (optional) Path to reference polygon(s).

    """
    p = Path(tif_dir).glob("*.tif")
    tifs = [x for x in p if x.is_file()]

    with LocalCluster() as cluster, Client(cluster) as client:
        click.echo(f"\nReading in annual composites from '{tif_dir}'")
        timeseries = read_timeseries(
            path_to_tifs=tifs,
            path_to_mask=mask,
        )
        if not timeseries.satts.is_annual_composite:
            raise ValueError("Stack is not a valid annual composite stack.")

        click.echo(f"Computing indices: {indices}")
        timeseries_for_metrics = compute_indices(timeseries, indices)

        rest_poly_gdf = gpd.read_restoration_polygons(rest_poly)
        if reference:
            ref_poly_gdf = gpd.read_reference_polygons(reference)
        ra = RestorationArea(
            restoration_polygon=rest_poly_gdf,
            reference_polygon=ref_poly_gdf,
            composite_stack=timeseries_for_metrics,
        )
        ctx.obj = ra


@cli.command("RRI")
@click.pass_obj
@click.option("-t", "--timestep", type=int, required=False)
def RRI(obj, timestep):
    click.echo(f"RRI is not released in v0.3.0b0... skipping")
    return
    if timestep:
        rri = obj.RRI(timestep=timestep)
    else:
        rri = obj.RRI()
    return rri


@cli.command("Y2R")
@click.pass_obj
@click.option("-p", "--percent", type=int, required=False)
def Y2R(obj, percent):

    click.echo(f"Computing Y2R")
    if percent:
        y2r = obj.y2r(percent_of_target=percent)
    else:
        y2r = obj.y2r()
    return y2r


@cli.command("YrYr")
@click.pass_obj
@click.option("-t", "--timestep", type=int, required=False)
def YrYr(obj, timestep):
    click.echo(f"Computing YrYr")
    if timestep:
        yryr = obj.yryr(timestep=timestep)
    else:
        yryr = obj.yryr()
    return yryr


@cli.command("dNBR")
@click.pass_obj
@click.option("-t", "--timestep", type=int, required=False)
def dNBR(obj, timestep):
    click.echo(f"Computing dNBR")
    if timestep:
        dnbr = obj.dnbr(timestep=timestep)
    else:
        dnbr = obj.dnbr()
    return dnbr


@cli.command("R80P")
@click.pass_obj
@click.option("-p", "--percent", type=int, required=False)
@click.option("-t", "--timestep", type=int, required=False)
def R80P(obj, percent, timestep):
    click.echo(f"Computing R80P")
    if percent:
        if timestep:
            p80r = obj.r80p(percent_of_target=percent, timestep=timestep)
        else:
            p80r = obj.r80p(percent_of_target=percent)
    else:
        if timestep:
            p80r = obj.r80p(timestep=timestep)
        else:
            p80r = obj.r80p()
    return p80r


@cli.result_callback()
def write_metrics(result, **kwargs):
    concated_metrics = xr.concat(result, dim="metric")
    click.echo(f"Writing metrics to '{kwargs['out']}'...")
    kwargs["out"].mkdir(parents=True, exist_ok=True)
    _metrics_to_tifs(
        metric=concated_metrics,
        out_dir=kwargs["out"],
    )
    click.echo(f"Finished.")
