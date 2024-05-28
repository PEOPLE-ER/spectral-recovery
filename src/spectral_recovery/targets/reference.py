"""Methods for computing reference-based recovery targets"""

import geopandas as gpd
import xarray as xr

from pandas import Index as pdIndex

def _window_time_clip(timeseries_data, site, reference_start, reference_end):
    """Clip data to """
    clipped_stacks = {}
    for i, row in site.iterrows():
        polygon_stack = timeseries_data.rio.clip(gpd.GeoSeries(row.geometry).values)
        clipped_stacks[i] = polygon_stack

    reference_image_stack = xr.concat(
        clipped_stacks.values(),
        dim=pdIndex(clipped_stacks.keys(), name="poly_id"),
    )
    return reference_image_stack.sel(time=slice(reference_start, reference_end))

def _check_reference_years(reference_start, reference_end, timeseries_data):
        print(reference_start, timeseries_data.time.dt.year)
        if int(reference_start) not in timeseries_data.time.dt.year:
            raise ValueError(f"Invalid reference start. {reference_start} not in timeseries_data time coordinates.")
        if int(reference_end) not in timeseries_data.time.dt.year:
            raise ValueError(f"Invalid reference end. {reference_end} not in timeseries_data time coordinates.")
def median(
    reference_sites: gpd.GeoDataFrame | str,
    timeseries_data: xr.DataArray,
    reference_start: str | int,
    reference_end: str | int,
):
    """Median target method for reference sites.

    Sequentially computes the median over time and the
    spatial dimensions (x and y). If there is more than 
    one reference site in the GeoDataFrame then the median
    is also then taken from all reference site medians.

    Parameters
    ----------
    polygon : gpd.GeoDataFrame
        The polygon/area to compute a recovery target for.
    timeseries_data : xr.DataArrah
        The timeseries of indices to derive the recovery target from.
        Must contain band, time, y, and x dimensions.
    reference_start : str
        Start year of reference window. Must exist in timeseries_data's
        time coordinates.
    reference_end :
        End year of reference window. Must exist in timeseries_data's
        time coordinates.

    Returns
    -------
    median_t : xr.DataArray
        DataArray of the median recovery targets with 1 coordinate dimension, "band".

    Notes
    ------
    Differs from spectral_recovery.targets.historic.median because 1) no scale 
    parameter because a reference site can only be of scale "polygon",
    and 2) because multiple polygons are reduced to a single value.

    """
    reference_start = str(reference_start)
    reference_end = str(reference_end)
    _check_reference_years(reference_start, reference_end, timeseries_data)
    if isinstance(reference_sites, str):
        reference_sites = gpd.read_file(reference_sites)
    # Clip timeseries data to polygon(s) and time dim
    clipped_data = _window_time_clip(
        timeseries_data=timeseries_data,
        site=reference_sites,
        reference_start=reference_start,
        reference_end=reference_end,
    )
    # Compute median sequentially
    target_data = clipped_data.sel(time=slice(reference_start, reference_end))
    median_time = target_data.median(dim="time", skipna=True)

    median_time = median_time.median(dim=["y", "x"], skipna=True)
    median_time = median_time.median(dim="poly_id", skipna=True)

    # Re-assign lost band coords.
    median_target = median_time.assign_coords(band=target_data.coords["band"])
    return median_target