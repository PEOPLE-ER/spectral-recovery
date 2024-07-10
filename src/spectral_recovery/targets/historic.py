"""Methods for computing historic recovery targets"""

import geopandas as gpd
import xarray as xr


def _clip_to_dict(timeseries_data, sites, reference_years) -> dict:
    """Get spatial and temporal clip for all restoration sites"""
    clipped_sites = {}
    for index, site in sites.iterrows():
            ref_s = str(reference_years[index]["reference_start"])
            ref_e = str(reference_years[index]["reference_end"])
            clipped_data = timeseries_data.rio.clip(gpd.GeoSeries(site.geometry).values)
            clipped_sites[index] = clipped_data.sel(time=slice(ref_s, ref_e))
    return clipped_sites

def _check_reference_years(reference_years, restoration_sites, timeseries_data):
     """Check that reference years are in timeseries coordinates and map to a polygon"""
     for polyid, years in reference_years.items():
        try:
            if years["reference_start"] not in timeseries_data.time.dt.year:
                raise ValueError(f"Invalid reference years for polygon {polyid}. {years['reference_start']} not in timeseries_data time coordinates.")
            if years["reference_end"] not in timeseries_data.time.dt.year:
                raise ValueError(f"Invalid reference years for polygon {polyid}. {years['reference_end']} not in timeseries_data time coordinates.")
        except KeyError:
            raise TypeError("Invalid reference_years format. Must be dict mapping polygon id's to nested dict of reference start and end years, e.g {0: {'reference_start': 2010, 'reference_end': 2011}, 1: {...}, ...}")
        for polyid in restoration_sites.index.tolist():
            if polyid not in list(reference_years.keys()):
                raise ValueError(f"Missing reference_years for polygon {polyid}")

def median(
    restoration_sites: gpd.GeoDataFrame | str,
    timeseries_data: xr.DataArray,
    reference_years: dict,
    scale: str,
) -> dict:
    """Median target method for historic targets.

    Sequentially computes the median over time and, optionally, the
    spatial dimensions (x and y) for each restoration site in the
    GeoDataFrame. Scale parameter used to determine the scale of 
    the target for each polygon. 

    Parameters
    ----------
    restoration_sites : gpd.GeoDataFrame
        The restoration sites to compute a recovery targets for.
    timeseries_data : xr.DataArray
        The timeseries of indices to derive the recovery target from.
        Must contain band, time, y, and x dimensions.
    reference_years : dict
        A dictionary mapping reference_start and reference_end years
        to each polygon in restoration_sites, e.g {0: {"reference_start": 2017, "reference_end": 2018}}
    scale : {"polygon", "pixel"}
        The scale to compute target for. Either 'polygon' which results
        in one value per-band (median of the polygon(s) across time), or
        'pixel' which results in a value for each pixel per-band (median
        of each pixel across time).

    Returns
    -------
    median_targets : dict
        Dictionary of DataArrays containing the median recovery target
        for each restoration site. Dictionary keys are the row indexes
        from restoration_sites. If scale="polygon", then DataArrays
        have coordinate dimensions "band". If scale="pixel", has
        coordinate dimensions "band", "y" and "x".
    
    Notes
    ------
    Differs from spectral_recovery.targets.reference.median because 1)
    can be parameterized on scale, and 2) because multiple polygons not
    reduced into a single value, i.e if 3 sites are given then recovery 
    targets are given for each of those 3 sites.

    """
    if not ((scale == "polygon") or (scale == "pixel")):
            raise ValueError(f"scale must be 'polygon' or 'pixel' ('{scale}' provided)")
    if isinstance(restoration_sites, str):
        restoration_sites = gpd.read_file(restoration_sites)
    _check_reference_years(reference_years, restoration_sites, timeseries_data)
    # Get dictionary of a time/space data clip for each polygon 
    clipped_site_data = _clip_to_dict(
        timeseries_data=timeseries_data,
        sites=restoration_sites,
        reference_years=reference_years
    )
    median_targets = {}
    for poly_id, clipped_data in clipped_site_data.items():
        # Get the median value across the time dimension
        median_time_data = clipped_data.median(dim="time", skipna=True)
        # Additional median calculations based on scale and dimensions
        if scale == "polygon":
            median_time_data = median_time_data.median(dim=["y", "x"], skipna=True)
        # Re-assign lost band coords.
        median_time_data = median_time_data.assign_coords(band=clipped_data.coords["band"])
        # Store in the output dictionary
        median_targets[poly_id] = median_time_data

    return median_targets


def window(
    restoration_sites: gpd.GeoDataFrame | str,
    timeseries_data: xr.DataArray,
    reference_years: dict,
    N: int = 3,
    na_rm: bool = False
):
    """Windowed recovery target method, parameterized on window size.

    The windowed method first computes the median along the time
    dimension and then for each pixel p in the restoration site
    polygon, computes the mean of a window of NxN pixels centred
    on pixel p, setting the mean to the recovery target value.

    Implementation is based on raster focal method in R.


    Parameters
    ----------
    polygon : gpd.GeoDataFrame
        The polygon/area to compute a recovery target for.
    timeseries_data : xr.DataArrah
        The timeseries of indices to derive the recovery target from.
        Must contain band, time, y, and x dimensions.
    reference_years : dict
        A dictionary mapping reference_start and reference_end years
        to each polygon in restoration_sites, e.g {0: {"reference_start": 2017, "reference_end": 2018}}
    N : int
        Size of the window (NxN). Must be odd. Default is 3.
    na_rm : bool
        If True, NaN will be removed from focal computations. The result will
        only be NA if all focal cells are NA., using na.rm=TRUE may not be a good idea in this function because it can unbalance the effect of the weights 

    """
    if not isinstance(N, int):
        raise TypeError("N must be int.")
    if N < 1:
        raise ValueError("N must be greater than or equal to 1.")
    if (N % 2) == 0:
        raise ValueError("N must be an odd int.")
    if not isinstance(na_rm, bool):
        raise TypeError("na_rm must be boolean.")

    if isinstance(restoration_sites, str):
        restoration_sites = gpd.read_file(restoration_sites)
    _check_reference_years(reference_years, restoration_sites, timeseries_data)

    window_targets = {}
    for poly_id, site_data in restoration_sites.iterrows():
        ref_s = str(reference_years[poly_id]["reference_start"])
        ref_e = str(reference_years[poly_id]["reference_end"])
        sliced_data = timeseries_data.sel(time=slice(ref_s, ref_e))
        median_time = sliced_data.median(dim="time", skipna=True)
        if na_rm:
            # Only 1 non-NaN value is required to set a value.
            min_periods = 1
        else:
            # All values in the window must be non-NaN
            min_periods = None
        median_window = median_time.rolling(
            dim={"y": N, "x": N}, center=True, min_periods=min_periods
        ).mean()
        window_targets[poly_id] = median_window.rio.clip(gpd.GeoSeries(site_data.geometry).values)
    
    return window_targets
