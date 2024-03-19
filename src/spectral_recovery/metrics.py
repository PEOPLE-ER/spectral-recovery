"""Methods for computing recovery metrics."""

from typing import Dict, List

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from spectral_recovery._utils import maintain_rio_attrs
from spectral_recovery.restoration import RestorationArea
from spectral_recovery.indices import compute_indices
from spectral_recovery.targets import MedianTarget


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
    restoration_polygons: gpd.GeoDataFrame,
    metrics: List[str],
    indices: List[str],
    reference_polygons: gpd.GeoDataFrame = None,
    index_constants: Dict[str, int] = {},
    timestep: int = 5,
    percent_of_target: int = 80,
    recovery_target_method=MedianTarget(scale="polygon"),
):
    indices_stack = compute_indices(
        image_stack=timeseries_data, indices=indices, constants=index_constants
    )
    restoration_area = RestorationArea(
        restoration_polygon=restoration_polygons,
        reference_polygons=reference_polygons,
        composite_stack=indices_stack,
        recovery_target_method=recovery_target_method,
    )
    m_results = []
    for m in metrics:
        try:
            m_func = METRIC_FUNCS[m.lower()]
        except KeyError:
            raise ValueError(f"{m} is not a valid metric choice!")
        m_results.append(
            m_func(
                ra=restoration_area,
                params={
                    "timestep": timestep,
                    "percent_of_target": percent_of_target,
                },
            ).assign_coords({"metric": m})
        )

    metrics = xr.concat(m_results, "metric")

    return metrics


@register_metric
def dnbr(ra: RestorationArea, params: Dict = {"timestep": 5}) -> xr.DataArray:
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

    rest_post_t = str(int(ra.restoration_start) + params["timestep"])
    if rest_post_t > ra.timeseries_end:
        raise ValueError(
            f"timestep={params['timestep']}, but"
            f" {ra.restoration_start}+{params['timestep']}={rest_post_t} not within"
            f" time coordinates: {ra.restoration_image_stack.coords['time'].values}. "
        ) from None

    dnbr_v = (
        ra.restoration_image_stack.sel(time=rest_post_t).drop_vars("time")
        - ra.restoration_image_stack.sel(time=ra.restoration_start).drop_vars("time")
    ).squeeze("time")

    return dnbr_v


@register_metric
def yryr(ra: RestorationArea, params: Dict = {"timestep": 5}):
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

    rest_post_t = str(int(ra.restoration_start) + params["timestep"])
    obs_post_t = ra.restoration_image_stack.sel(time=rest_post_t).drop_vars("time")
    obs_start = ra.restoration_image_stack.sel(time=ra.restoration_start).drop_vars(
        "time"
    )
    yryr_v = ((obs_post_t - obs_start) / params["timestep"]).squeeze("time")

    return yryr_v


@register_metric
def r80p(
    ra: RestorationArea, params: Dict = {"percent_of_target": 80, "timestep": 5}
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
        rest_post_t = ra.restoration_image_stack["time"].data[-1]
    elif params["timestep"] < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    elif params["percent_of_target"] <= 0 or params["percent_of_target"] > 100:
        raise ValueError(VALID_PERC_MSP)
    else:
        rest_post_t = str(int(ra.restoration_start) + params["timestep"])
    r80p_v = (ra.restoration_image_stack.sel(time=rest_post_t)).drop_vars("time") / (
        (params["percent_of_target"] / 100) * ra.recovery_target
    )
    try:
        # if using the default timestep (the max/most recent),
        # the indexing will not get rid of the "time" dim
        r80p_v = r80p_v.squeeze("time")
    except KeyError:
        pass
    return r80p_v


@register_metric
def y2r(ra: RestorationArea, params: Dict = {"percent_of_target": 80}) -> xr.DataArray:
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
    y2r_target = ra.recovery_target * (params["percent_of_target"] / 100)
    recovery_window = ra.restoration_image_stack.sel(
        time=slice(ra.restoration_start, None)
    )

    years_to_recovery = (recovery_window >= y2r_target).argmax(dim="time", skipna=True)
    # Pixels with value 0 could be:
    # 1. pixels that were recovered at the first timestep
    # 2. pixels that never recovered (argmax returns 0 if all values are False)
    # 3. pixels that were NaN for the entire recovery window.
    #
    # Only 1. is a valid 0, so set pixels that never recovered to -9999,
    # and pixels that were NaN for the entire recovery window back to NaN.
    not_zero = years_to_recovery != 0
    recovered_at_zero = recovery_window.sel(time=ra.restoration_start) >= y2r_target
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
    ra: RestorationArea, params: Dict = {"timestep": 5}
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

    rest_post_tm1 = str(int(ra.restoration_start) + (params["timestep"] - 1))
    rest_post_t = str(int(ra.restoration_start) + params["timestep"])

    if pd.to_datetime(rest_post_tm1) not in ra.restoration_image_stack.time.values:
        raise ValueError(f"{rest_post_tm1} (year of timestep - 1) not found in time dim.")
    if  pd.to_datetime(rest_post_t) not in ra.restoration_image_stack.time.values:
        raise ValueError(f"{rest_post_t} (year of timestep) not found in time dim.")

    max_rest_t_tm1 = ra.restoration_image_stack.sel(time=slice(rest_post_tm1, rest_post_t)).max(dim=["time"])
    rest_start = ra.restoration_image_stack.sel(time=ra.restoration_start).drop_vars("time")
    dist_start = ra.restoration_image_stack.sel(time=ra.disturbance_start).drop_vars("time")
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
