"""Methods for computing recovery metrics."""

from typing import Dict, List
import warnings

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from spectral_recovery.utils import maintain_rio_attrs

warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

NEG_TIMESTEP_MSG = "timestep cannot be negative."
VALID_PERC_MSP = "percent must be between 0 and 100."
METRIC_FUNCS = {}

def _register_metrics(f):
    """Add function and name to global name/func dict"""
    METRIC_FUNCS[f.__name__] = f
    return f

@maintain_rio_attrs
def compute_metrics(
    metrics: List[str],
    timeseries_data: xr.DataArray,
    restoration_sites: gpd.GeoDataFrame,
    recovery_targets: xr.DataArray | Dict = None,
    timestep: int = 5,
    percent_of_target: int = 80,
) -> Dict:
    """Compute recovery metrics for each restoration site.

    Parameters
    ----------
    metrics : list of str
        The names of recovery metrics to compute. Accepted values:
            - "Y2R": Years-to-Recovery
            - "R80P": Recovered 80 Percent
            - "deltaIR": delta Index Regrowth
            - "YrYr": Year-on-Year Average
            - "RRI": Relative Recovery Indicator
    timeseries_data : xarray.DataArray
        The timeseries of indices to compute recovery metrics with.
        Must contain band, time, y, and x dimensions.
    restoration_sites : geopandas.GeoDataFrame
        The restoration sites to compute a recovery targets for.
    recovery_targets : xarray.DataArray or dict
        The recovery targets. Either a dict mapping polygon IDs to 
        xarray.DataArrays of recovery targets or a single xarray.DataArray.
    timestep : int, optional
        The timestep post-restoration to consider when computing recovery
        metrics. Only used for "R80P", "deltaIR", and "YrYr" and "RRI" recovery
        metrics. Default = 5.
    percent_of_target : int, optional
        The percent of the recovery target to consider when computing
        recovery metrics. Only used for "Y2R" and "R80P". Default = 80.

    Returns
    -------
    metric_ds : xarray.Dataset
        Dataset of restoration site ID variables, each containing an array
        of recovery metrics specific to each site.

    Notes
    -----
    Recovery target arrays _must_ be broadcastable to the timeseries_data 
    when timeseries_data is clipped to each restoration site.

    """
    if recovery_targets is None:
        for tmetric in ["Y2R", "R80P"]:
            if tmetric in metrics:
                raise ValueError(
                    f"{tmetric} requires a recovery target but recovery_target is None"
                )
    per_polygon_metrics = {}
    for site_id, row in restoration_sites.iterrows():
        # Prepare arguments being passed to the metric functions
        clipped_ts = timeseries_data.rio.clip([row.geometry])
        all_kwargs = {
            "disturbance_start": row["dist_start"],
            "restoration_start": row["rest_start"],
            "timeseries_data": clipped_ts,
            "timestep": timestep,
            "percent_of_target": percent_of_target,
        }
        if isinstance(recovery_targets, dict):
            all_kwargs["recovery_target"] = recovery_targets[site_id]
        else:
            # if a DataArray or None, just pass as-is
            all_kwargs["recovery_target"] = recovery_targets
        m_results = []
        for m in metrics:
            try:
                m_func = METRIC_FUNCS[m.lower()]
            except KeyError:
                raise ValueError(f"{m} is not a valid metric choice!") from None
            func_kwargs = {k: all_kwargs[k] for k in m_func.__code__.co_varnames if k in list(all_kwargs.keys())}
            m_results.append(m_func(**func_kwargs).assign_coords({"metric": m}))
        per_polygon_metrics[site_id] = xr.concat(m_results, "metric")

    return per_polygon_metrics


def _has_continuous_years(images: xr.DataArray):
    """Check for continous set of years in DataArray"""
    years = images.coords["time"].dt.year.values
    for year in list(range(years[0], years[-1] + 1)):
        if year not in years:
            return False
    return True

@_register_metrics
def deltair(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    timestep: int = 5,
) -> xr.DataArray:
    """Per-pixel deltaIR.

    The absolute change in a spectral index’s value at a point in the
    restoration monitoring window from the start of the restoration monitoring
    window. The default is the change that has occurred 5 years into the
    restoration from the start of the restoration.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data: 
        The timeseries of indices to compute dIR with. Must contain
        band, time, y, and x coordinate dimensions.
    timestep : int
        The timestep post-restoration to compute deltaIR with. 

    Returns
    -------
    deltair_v : xr.DataArray
        DataArray containing the deltaIR value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    rest_post_t = str(restoration_start + timestep)
    timesries_end = (
        np.max(timeseries_data.time.values).astype("datetime64[Y]").astype(int) + 1970
    )
    if int(rest_post_t) > int(timesries_end):
        raise ValueError(
            f" {restoration_start}+{timestep}={rest_post_t} is greater"
            f" than end of timeseries: {timesries_end}. "
        ) from None

    deltair_v = (
        timeseries_data.sel(time=rest_post_t).drop_vars("time")
        - timeseries_data.sel(time=str(restoration_start)).drop_vars("time")
    ).squeeze("time")

    return deltair_v


@_register_metrics
def yryr(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    timestep: int = 5,
):
    """Per-pixel YrYr.

    The average annual recovery rate relative to a fixed time interval
    during the restoration monitoring window. The default is the first 5
    years of the restoration window, however this can be changed by specifying
    the parameter `timestep`.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data: 
        The timeseries of indices to compute YrYr with. Must contain
        band, time, y, and x coordinate dimensions.
    timestep : int
        The timestep post-restoration to compute YrYr with. 

    Returns
    -------
    yryr_v : xr.DataArray
        DataArray containing the YrYr value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    rest_post_t = str(restoration_start + timestep)
    obs_post_t = timeseries_data.sel(time=rest_post_t).drop_vars("time")
    obs_start = timeseries_data.sel(time=str(restoration_start)).drop_vars("time")
    yryr_v = ((obs_post_t - obs_start) / timestep).squeeze("time")
    return yryr_v


@_register_metrics
def r80p(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    recovery_target: xr.DataArray,
    timestep: int = 5,
    percent_of_target: int = 80,
) -> xr.DataArray:
    """Per-pixel R80P.

    The extent to which the trajectory has reached 80% of the recovery
    target value. The metric commonly uses the maximum value from the
    4th or 5th year of restoration window to show the extent to which a
    pixel has reached 80% of the target value 5 years into the restoration
    window. However for monitoring purposes, this tool uses the selected `timestep`
    or defaults to the current timestep to provide up to date recovery
    progress. 80% of the recovery target value is the default, however this
    can be changed by modifying the value of `percent`.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data: 
        The timeseries of indices to compute R80P with. Must contain
        band, time, y, and x coordinate dimensions.
    recovery_target : xarray.DataArray
        Recovery target values. Must be broadcastable to timeseries_data.
    timestep : int
        The timestep post-restoration to compute R80P with. 
    percent_of_target : int
        The percent of the recovery target to consider when computing
        R80P.

    Returns
    -------
    r80p_v : xr.DataArray
        DataArray containing the R80P value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    elif percent_of_target <= 0 or percent_of_target > 100:
        raise ValueError(VALID_PERC_MSP)
    else:
        rest_post_t = str(restoration_start + timestep)
    r80p_v = (timeseries_data.sel(time=rest_post_t)).drop_vars("time") / (
        (percent_of_target / 100) * recovery_target
    )
    try:
        # if using the default timestep (the max/most recent),
        # the indexing will not get rid of the "time" dim
        r80p_v = r80p_v.squeeze("time")
    except KeyError:
        pass
    return r80p_v

@_register_metrics
def y2r(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    recovery_target: xr.DataArray,
    percent_of_target: int = 80,
) -> xr.DataArray:
    """Per-pixel Y2R.

    The length of time taken (in time steps/years) for a given pixel to
    first reach 80% of its recovery target value. The percent can be modified
    by changing the value of `percent`.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data: 
        The timeseries of indices to compute Y2R with. Must contain
        band, time, y, and x coordinate dimensions.
    recovery_target : xarray.DataArray
        Recovery target values. Must be broadcastable to timeseries_data.
    percent_of_target : int
        The percent of the recovery target to consider when computing
        Y2R.

    Returns
    -------
    y2r_v : xr.DataArray
        DataArray containing the number of years taken for each pixel
        to reach the recovery target value. NaN represents pixels that
        have not yet reached the recovery target value.

    """
    if percent_of_target <= 0 or percent_of_target > 100:
        raise ValueError(VALID_PERC_MSP)

    recovery_window = timeseries_data.sel(time=slice(str(restoration_start), None))
    if not _has_continuous_years(recovery_window):
        raise ValueError(
            f"Missing years in `timeseries_data`, cannot compute Y2R. Y2R requires a continuous timeseries from the restoration start year onwards."
        )
    y2r_target = recovery_target * (percent_of_target / 100)
    years_to_recovery = (recovery_window >= y2r_target).argmax(dim="time", skipna=True)
    # Pixels with value 0 could be:
    # 1. pixels that were recovered at the first timestep
    # 2. pixels that never recovered (argmax returns 0 if all values are False)
    # 3. pixels that were NaN for the entire recovery window.
    #
    # Only 1. is a valid 0, so set pixels that never recovered to -9999,
    # and pixels that were NaN for the entire recovery window back to NaN.
    not_zero = years_to_recovery != 0
    recovered_at_zero = recovery_window.sel(time=str(restoration_start)) >= y2r_target
    is_nan = recovery_window.isnull().all("time")
    valid_output = not_zero | recovered_at_zero | is_nan

    # Set unrecovered 0's to -9999, aund NaN 0's to NaN
    y2r_v = years_to_recovery.where(valid_output, -9999)
    y2r_v = y2r_v.where(~is_nan, np.nan).drop_vars("time")

    try:
        y2r_v = y2r_v.squeeze("time")
    except KeyError:
        pass
    return y2r_v


@_register_metrics
def rri(
    disturbance_start: int,
    restoration_start: int,
    timeseries_data: xr.DataArray,
    timestep: int = 5,
) -> xr.DataArray:
    """Per-pixel RRI.

    A modified version of the commonly used RI, the RRI accounts for
    noise in trajectory by using the maximum from the 4th or 5th year
    in monitoring window. The metric relates recovery magnitude to
    disturbance magnitude, and is the change in index value in 4 or 5
    years divided by the change due to disturbance.

    Parameters
    ----------
    disturbance_start : int
        The start year of the disturbance event.
    restoration_start : int
        The start year of restoration activities.
    timeseries_data: 
        The timeseries of indices to compute RRI with. Must contain
        band, time, y, and x coordinate dimensions.
    timestep : int
        The timestep post-restoration to compute RRI with. 

    Returns
    -------
    rri_v : xr.DataArray
        DataArray containing the RRI value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    if timestep == 0:
        raise ValueError("timestep for RRI must be greater than 0.")

    rest_post_tm1 = str(restoration_start + (timestep - 1))
    rest_post_t = str(restoration_start + timestep)

    if pd.to_datetime(rest_post_tm1) not in timeseries_data.time.values:
        raise ValueError(
            f"{rest_post_tm1} (year of timestep - 1) not found in time dim."
        )
    if pd.to_datetime(rest_post_t) not in timeseries_data.time.values:
        raise ValueError(f"{rest_post_t} (year of timestep) not found in time dim.")

    max_rest_t_tm1 = timeseries_data.sel(time=slice(rest_post_tm1, rest_post_t)).max(
        dim=["time"]
    )
    rest_start = timeseries_data.sel(time=str(restoration_start)).drop_vars("time")
    dist_start = timeseries_data.sel(time=str(disturbance_start)).drop_vars("time")
    dist_end = rest_start

    rri_v = (max_rest_t_tm1 - rest_start) / (dist_start - dist_end)
    # if dist_pre_1_2 has length greater than one we will need to squeeze the time dim
    try:
        rri_v = rri_v.squeeze("time")
    except KeyError:
        pass
    return rri_v


def _year_dt(dt, dt_type: str = "int"):
    """Get int or str representation of year from datetime-like object."""
    # TODO: refuse to move forward if dt isn't datetime-like
    try:
        dt_dt = pd.to_datetime(dt)
        year = dt_dt.year
    except ValueError:
        raise ValueError(
            f"Unable to get year {type} from {dt} of type {type(dt)}"
        ) from None
    if dt_type == "str":
        return str(year)
    return year
