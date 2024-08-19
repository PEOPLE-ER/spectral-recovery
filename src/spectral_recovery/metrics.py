"""Methods for computing recovery metrics."""

from typing import Dict, List

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from spectral_recovery._utils import maintain_rio_attrs

NEG_TIMESTEP_MSG = "timestep cannot be negative."
VALID_PERC_MSP = "percent must be between 0 and 100."
METRIC_FUNCS = {}


def register_metric(f):
    """Add function and name to global name/func dict"""
    METRIC_FUNCS[f.__name__] = f
    return f


@maintain_rio_attrs
def compute_metrics(
    timeseries_data: xr.DataArray,
    restoration_sites: gpd.GeoDataFrame,
    metrics: List[str],
    recovery_targets: xr.DataArray = None,
    timestep: int = 5,
    percent_of_target: int = 80,
):
    """
    TODO: Add docstring.

    """
    if recovery_targets is None:
        for tmetric in ["Y2R", "R80P"]:
            if tmetric in metrics:
                raise ValueError(
                    f"{tmetric} requires a recovery target but recovery_target is None"
                )

    per_polygon_metrics = {}
    for index, row in restoration_sites.iterrows():
        # Prepare arguments being passed to the metric functions
        clipped_ts = timeseries_data.rio.clip([row.geometry])
        m_kwargs = dict(
            disturbance_start=row["dist_start"],
            restoration_start=row["rest_start"],
            timeseries_data=clipped_ts,
            params={
                "timestep": timestep,
                "percent_of_target": percent_of_target,
            },
        )
        if isinstance(recovery_targets, dict):
            m_kwargs["recovery_target"] = recovery_targets[index]
        else:
            # if a DataArray or None, just pass as-is
            m_kwargs["recovery_target"] = recovery_targets

        m_results = []
        for m in metrics:
            try:
                m_func = METRIC_FUNCS[m.lower()]
            except KeyError:
                raise ValueError(f"{m} is not a valid metric choice!")
            m_results.append(m_func(**m_kwargs).assign_coords({"metric": m}))
        per_polygon_metrics[index] = xr.concat(m_results, "metric")
    metric_da = xr.concat(
        per_polygon_metrics.values(), pd.Index(per_polygon_metrics.keys(), name="site")
    )
    metric_ds = metric_da.to_dataset(dim="site")
    return metric_ds


def has_no_missing_years(images: xr.DataArray):
    """Check for continous set of years in DataArray"""
    years = images.coords["time"].dt.year.values
    if not np.all((years == list(range(years[0], years[-1] + 1)))):
        return False
    return True


@register_metric
def dnbr(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    params: Dict = {"timestep": 5},
    recovery_target: xr.DataArray = None,
    disturbance_start: int = None,
) -> xr.DataArray:
    """Per-pixel dNBR.

    The absolute change in a spectral index’s value at a point in the
    restoration monitoring window from the start of the restoration monitoring
    window. The default is the change that has occurred 5 years into the
    restoration from the start of the restoration.

    Parameters
    ----------
    ra : RestorationArea
        The restoration area to compute dnbr for.
    params : Dict
        Parameters to customize metric computation. dnbr uses
        the 'timestep' parameter with default = {"timestep": 5}

    Returns
    -------
    dnbr_v : xr.DataArray
        DataArray containing the dNBR value for each pixel.

    """
    if params["timestep"] < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    rest_post_t = str(restoration_start + params["timestep"])
    timesries_end = (
        np.max(timeseries_data.time.values).astype("datetime64[Y]").astype(int) + 1970
    )
    if int(rest_post_t) > int(timesries_end):
        raise ValueError(
            f" {restoration_start}+{params['timestep']}={rest_post_t} is greater"
            f" than end of timeseries: {timesries_end}. "
        ) from None

    dnbr_v = (
        timeseries_data.sel(time=rest_post_t).drop_vars("time")
        - timeseries_data.sel(time=str(restoration_start)).drop_vars("time")
    ).squeeze("time")

    return dnbr_v


@register_metric
def yryr(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    params: Dict = {"timestep": 5},
    recovery_target: xr.DataArray = None,
    disturbance_start: int = None,
):
    """Per-pixel YrYr.

    The average annual recovery rate relative to a fixed time interval
    during the restoration monitoring window. The default is the first 5
    years of the restoration window, however this can be changed by specifying
    the parameter `timestep`.

    Parameters
    ----------
    ra : RestorationArea
        The restoration area to compute yryr for.
    params : Dict
        Parameters to customize metric computation. yryr uses
        the 'timestep' parameter with default = {"timestep": 5}

    Returns
    -------
    yryr_v : xr.DataArray
        DataArray containing the YrYr value for each pixel.

    """
    if params["timestep"] < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    rest_post_t = str(restoration_start + params["timestep"])
    obs_post_t = timeseries_data.sel(time=rest_post_t).drop_vars("time")
    obs_start = timeseries_data.sel(time=str(restoration_start)).drop_vars("time")
    yryr_v = ((obs_post_t - obs_start) / params["timestep"]).squeeze("time")

    return yryr_v


@register_metric
def r80p(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    recovery_target: xr.DataArray,
    params: Dict = {"percent_of_target": 80, "timestep": 5},
    disturbance_start: int = None,
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
    ra : RestorationArea
        The restoration area to compute r80p for.
    params : Dict
        Parameters to customize metric computation. r80p uses
        the 'timestep' and 'percent_of_target' parameters with
        default = {"percent_of_target": 80, "timestep": 5}.

    Returns
    -------
    r80p_v : xr.DataArray
        DataArray containing the R80P value for each pixel.

    """
    if params["timestep"] is None:
        rest_post_t = timeseries_data["time"].data[-1]
    elif params["timestep"] < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    elif params["percent_of_target"] <= 0 or params["percent_of_target"] > 100:
        raise ValueError(VALID_PERC_MSP)
    else:
        rest_post_t = str(restoration_start + params["timestep"])
    r80p_v = (timeseries_data.sel(time=rest_post_t)).drop_vars("time") / (
        (params["percent_of_target"] / 100) * recovery_target
    )
    try:
        # if using the default timestep (the max/most recent),
        # the indexing will not get rid of the "time" dim
        r80p_v = r80p_v.squeeze("time")
    except KeyError:
        pass
    return r80p_v


@register_metric
def y2r(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    recovery_target: xr.DataArray,
    params: Dict = {"percent_of_target": 80},
    disturbance_start: int = None,
) -> xr.DataArray:
    """Per-pixel Y2R.

    The length of time taken (in time steps/years) for a given pixel to
    first reach 80% of its recovery target value. The percent can be modified
    by changing the value of `percent`.

    Parameters
    ----------
    ra : RestorationArea
        The restoration area to compute r80p for.
    params : Dict
        Parameters to customize metric computation. r80p uses
        the 'percent_of_target' parameter with default = {"percent_of_target": 80}

    Returns
    -------
    y2r_v : xr.DataArray
        DataArray containing the number of years taken for each pixel
        to reach the recovery target value. NaN represents pixels that
        have not yet reached the recovery target value.

    """
    if params["percent_of_target"] <= 0 or params["percent_of_target"] > 100:
        raise ValueError(VALID_PERC_MSP)

    recovery_window = timeseries_data.sel(time=slice(str(restoration_start), None))
    if not has_no_missing_years(recovery_window):
        raise ValueError(
            f"Missing years. Y2R requires data for all years between {recovery_window.time.min()}-{recovery_window.time.max()}."
        )

    print(recovery_target, params["percent_of_target"])
    y2r_target = recovery_target * (params["percent_of_target"] / 100)

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


@register_metric
def rri(
    disturbance_start: int,
    restoration_start: int,
    timeseries_data: xr.DataArray,
    params: Dict = {"timestep": 5},
    recovery_target: xr.DataArray = None,
) -> xr.DataArray:
    """Per-pixel RRI.

    A modified version of the commonly used RI, the RRI accounts for
    noise in trajectory by using the maximum from the 4th or 5th year
    in monitoring window. The metric relates recovery magnitude to
    disturbance magnitude, and is the change in index value in 4 or 5
    years divided by the change due to disturbance.

    Parameters
    ----------
    ra : RestorationArea
        The restoration area to compute r80p for.
    params : Dict
        Parameters to customize metric computation. r80p uses
        the 'timestep' and 'use_dist_avg' parameters with
        default = {"timestep": 5}.

    Returns
    -------
    rri_v : xr.DataArray
        DataArray containing the RRI value for each pixel.

    """
    if params["timestep"] < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    if params["timestep"] == 0:
        raise ValueError("timestep for RRI must be greater than 0.")

    rest_post_tm1 = str(restoration_start + (params["timestep"] - 1))
    rest_post_t = str(restoration_start + params["timestep"])

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


def year_dt(dt, dt_type: str = "int"):
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
