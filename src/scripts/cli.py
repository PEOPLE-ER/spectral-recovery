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
from spectral_recovery.restoration import RestorationArea
from spectral_recovery.io.raster import read_and_stack_tifs, metrics_to_tifs

INDEX_CHOICE = [i.value for i in Index]
METRIC_CHOICE = [str(m) for m in Metric]

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
@click.argument(
    "rest_year",
    type=click.DateTime(formats=["%Y"]),
)
@click.argument(
    "ref_poly",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "ref_years",
    type=(click.DateTime(formats=["%Y"]), click.DateTime(formats=["%Y"])),
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
    rest_year: str,
    ref_poly: Path,
    ref_years: str,
    out: Path,
    indices: List[str] = None,
    mask: xr.DataArray = None,
) -> None:
    """Compute recovery metrics.

    This script will compute recovery metrics over the area of REST_POLY
    using spectral data from annual composites in TIF_DIR. Recovery targets
    will be derived over REF_POLYGON for REF_YEARS. 

    \b
    TIF_DIR        Path to a directory containing annual composites.
    OUT            Path to directory to write output rasters.
    REST_POLY      Path to the restoration area polygon.
    REST_YEAR      Year of the restoration event.
    REF_POLY       Path to reference polygon(s).
    REF_YEARS      Start and end years over which to derive a recovery target.
    
    """
    # TODO: move user input prep/validation into own function?
    rest_year = pd.to_datetime(rest_year)
    ref_years = [pd.to_datetime(ref_years[0]), pd.to_datetime(ref_years[1])]
   
    try:
        valid_indices = [Index[name.lower()] for name in indices]
    except KeyError as e:
        raise e from None

    p = Path(tif_dir).glob("*.tif")
    tifs = [x for x in p if x.is_file()]

    with LocalCluster() as cluster, Client(cluster) as client:
        click.echo(f"\nReading in annual composites from {tif_dir}")
        timeseries = read_and_stack_tifs(
            path_to_tifs=tifs,
            path_to_mask=mask,
        )
        if not timeseries.satts.valid:
            raise ValueError("Stack is not a valid yearly composite stack.")

        if indices is not None and len(indices) != 0:
            click.echo(f"Computing indices: {indices}")
            timeseries_for_metrics = timeseries.satts.indices(valid_indices)
        else:
            timeseries_for_metrics = timeseries

        rest_poly_gdf = gpd.read_file(rest_poly)
        ref_poly_gdf = gpd.read_file(ref_poly)
        ra = RestorationArea(
            restoration_polygon=rest_poly_gdf,
            restoration_year=rest_year,
            reference_polygon=ref_poly_gdf,
            reference_years=ref_years,
            composite_stack=timeseries_for_metrics,
        )
        ctx.obj = ra

@cli.command("RI")
@click.pass_obj
@click.option("-t", "--timestep", type=int, required=False)
def RI(obj, timestep):
    click.echo(f"Computing RI")
    if timestep:
        ri = obj.RI(timestep=timestep)
    else:
        ri = obj.RI()
    return ri

@cli.command("Y2R")
@click.pass_obj
@click.option("-p", "--percent", type=int, required=False)
def Y2R(obj, percent):
    click.echo(f"Computing Y2R")
    if percent:
        y2r = obj.Y2R(percent_of_target=percent)
    else:
        y2r = obj.Y2R()
    return y2r

@cli.command("YrYr")
@click.pass_obj
@click.option("-t", "--timestep", type=int, required=False)
def YrYr(obj, timestep):
    click.echo(f"Computing YrYr")
    if timestep:
        yryr = obj.YrYr(timestep=timestep)
    else:
        yryr = obj.YrYr()
    return yryr

@cli.command("dNBR")
@click.pass_obj
@click.option("-t", "--timestep", type=int, required=False)
def dNBR(obj, timestep):
    click.echo(f"Computing dNBR")
    if timestep:
        dnbr = obj.dNBR(timestep=timestep)
    else: 
        dnbr = obj.dNBR()
    return dnbr

@cli.command("P80R")
@click.pass_obj
@click.option("-p", "--percent", type=int, required=False)
def P80R(obj, percent):
    click.echo(f"Computing P80R")
    if percent:
        p80r = obj.P80R(percent_of_target=percent)
    else:
        p80r = obj.P80R()
    return p80r

@cli.result_callback()
def write_metrics(result, **kwargs):
    concated_metrics = xr.concat(result, dim="metric")
    click.echo(f"Writing metrics to {kwargs['out']}")
    kwargs["out"].mkdir(parents=True, exist_ok=True)
    metrics_to_tifs(
        metrics_array=concated_metrics,
        out_dir=kwargs["out"],
    )
    click.echo(f"Finished.")