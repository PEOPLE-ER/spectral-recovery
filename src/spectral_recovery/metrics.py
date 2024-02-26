"""Methods for computing recovery metrics."""
from typing import Dict, List

import xarray as xr
import numpy as np
import pandas as pd

from spectral_recovery._utils import maintain_rio_attrs
from spectral_recovery.restoration import RestorationArea
from spectral_recovery.indices import compute_indices
from spectral_recovery.targets import MedianTarget


NEG_TIMESTEP_MSG = "timestep cannot be negative."
VALID_PERC_MSP = "percent must be between 0 and 100."
METRIC_FUNCS = {}


def register_metric(f):
    """ Add function and name to global name/func dict """
    METRIC_FUNCS[f.__name__] = f
    return f


@maintain_rio_attrs
def compute_metrics(
        timeseries_data: xr.DataArray,
        restoration_polygons: "geopandas.GeoDataFrame",
        metrics: List[str],
        indices: List[str],
        reference_polygons: "geopandas.GeoDataFrame" = None,
        index_constants: Dict[str, int] = {},
        timestep: int = 5, 
        percent_of_target: int = 80,
        recovery_target_method = MedianTarget(scale="polygon"),
    ):

    indices_stack = compute_indices(image_stack=timeseries_data, indices=indices, constants=index_constants)
    restoration_area = RestorationArea(
        restoration_polygon=restoration_polygons,
        reference_polygons=reference_polygons,
        composite_stack=indices_stack,
        recovery_target_method=recovery_target_method
    )
    m_results = []
    for m in metrics:
        try:
            m_func = METRIC_FUNCS[m.lower()]
        except KeyError:
            raise ValueError("{m} is not a valid metric choice!")
        m_results.append(m_func(ra=restoration_area, timestep=timestep, percent_of_target=percent_of_target))

    metrics = xr.concat(m_results, "metric")

    return metrics


@register_metric
def dnbr(
    ra: RestorationArea,
    timestep: int = 5,
) -> xr.DataArray:
    """Per-pixel dNBR.

    The absolute change in a spectral index’s value at a point in the
    restoration monitoring window from the start of the restoration monitoring
    window. The default is the change that has occurred 5 years into the
    restoration from the start of the restoration.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray of images over which to compute per-pixel dNBR.
    rest_start : str
        The starting year of the restoration monitoring window.
    timestep : int, optional
        The timestep (years) in the restoration monitoring
        window (post rest_start) from which to evaluate absolute
        change. Default = 5.

    Returns
    -------
    dnbr_v : xr.DataArray
        DataArray containing the dNBR value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    
    rest_post_t = str(int(ra.restoration_start) + timestep)
    if rest_post_t > ra.timeseries_end:
        raise ValueError(
                f"timestep={timestep}, but {ra.restoration_start}+{timestep}={rest_post_t} not"
                f" within time coordinates: {ra.restoration_image_stack.coords['time'].values}. "
            ) from None
    
    dnbr_v = (
        ra.restoration_image_stack.sel(time=rest_post_t).drop_vars("time")
        - ra.restoration_image_stack.sel(time=ra.restoration_start).drop_vars("time")
    ).squeeze("time")

            
    return dnbr_v

@register_metric
def yryr(
    ra: RestorationArea,
    timestep: int = 5,
):
    """Per-pixel YrYr.

    The average annual recovery rate relative to a fixed time interval
    during the restoration monitoring window. The default is the first 5
    years of the restoration window, however this can be changed by specifying
    the parameter `timestep`.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray of images over which to compute per-pixel YrYr.
    rest_start : str
        The starting year of the restoration monitoring window.
    timestep : int, optional
        The timestep (years) in the restoration monitoring
        window (post rest_start) from which to evaluate absolute
        change. Default = 5.

    Returns
    -------
    yryr_v : xr.DataArray
        DataArray containing the YrYr value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    rest_post_t = str(int(ra.restoration_start) + timestep)
    obs_post_t = ra.restoration_image_stack.sel(time=rest_post_t).drop_vars("time")
    obs_start = ra.restoration_image_stack.sel(time=ra.restoration_start).drop_vars("time")
    yryr_v = ((obs_post_t - obs_start) / timestep).squeeze("time")

    return yryr_v

@register_metric
def r80p(
    ra: RestorationArea,
    timestep: int = None,
    percent: int = 80,
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
    image_stack : xr.DataArray
        DataArray of images over which to compute per-pixel dNBR.
    rest_start : str
        The starting year of the restoration monitoring window.
    recovery_target : xr.DataArray
        Recovery target values. Must be broadcastable to image_stack.
    timestep : int, optional
        The timestep (years) in the restoration monitoring window
        from which to evaluate absolute change. Default = -1 which
        represents the max/most recent timestep.
    percent: int, optional
        Percent of recovery to compute recovery against. Default = 80.

    Returns
    -------
    r80p_v : xr.DataArray
        DataArray containing the R80P value for each pixel.

    """
    if timestep is None:
        rest_post_t = ra.restoration_image_stack["time"].data[-1]
    elif timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    elif percent <= 0 or percent > 100:
        raise ValueError(VALID_PERC_MSP)
    else:
        rest_post_t = str(int(ra.restoration_start) + timestep)
    r80p_v = (ra.restoration_image_stack.sel(time=rest_post_t)).drop_vars("time") / (
        (percent / 100) * ra.recovery_target
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
    ra: RestorationArea,
    percent: int = 80,
) -> xr.DataArray:
    """Per-pixel Y2R.

    The length of time taken (in time steps/years) for a given pixel to
    first reach 80% of its recovery target value. The percent can be modified
    by changing the value of `percent`.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray of images over which to compute per-pixel Y2R.
    rest_start : str
        The starting year of the restoration monitoring window.
    recovery_target : xr.DataArray
        Recovery target values. Must be broadcastable to image_stack.
    percent: int, optional
        Percent of recovery to compute recovery against. Default = 80.

    Returns
    -------
    y2r_v : xr.DataArray
        DataArray containing the number of years taken for each pixel
        to reach the recovery target value. NaN represents pixels that
        have not yet reached the recovery target value.

    """
    if percent <= 0 or percent > 100:
        raise ValueError(VALID_PERC_MSP)
    reco_target = ra.recovery_target * (percent / 100)
    recovery_window = ra.restoration_image_stack.sel(time=slice(ra.restoration_start, None))

    years_to_recovery = (recovery_window >= reco_target).argmax(dim="time", skipna=True)
    # Pixels with value 0 could be pixels that were recovered at the first timestep, or
    # pixels that never recovered (argmax returns 0 if all values are False).
    # Only the former are valid 0's, so set pixels that never recovered to NaN.
    zero_mask = years_to_recovery == 0
    recovered_at_zero = recovery_window.sel(time=ra.restoration_start) >= reco_target
    valid_zeros = zero_mask & recovered_at_zero
    valid_output = valid_zeros | (~zero_mask)

    y2r_v = years_to_recovery.where(valid_output, np.nan).drop_vars("time")
    try:
        y2r_v = y2r_v.squeeze("time")
    except KeyError:
        pass
    return y2r_v


@register_metric
def rri(
    ra: RestorationArea, 
    timestep: int = 5,
    use_dist_avg: bool = False,
) -> xr.DataArray:
    """Per-pixel RRI.

    A modified version of the commonly used RI, the RRI accounts for
    noise in trajectory by using the maximum from the 4th or 5th year
    in monitoring window. The metric relates recovery magnitude to
    disturbance magnitude, and is the change in index value in 4 or 5
    years divided by the change due to disturbance. Users can optionally
    choose to use the average of the disturbance period and the pre-disturbance
    window to calculate the disturbance magnitude, by setting `use_dist_avg=True`.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray of images over which to compute per-pixel dNBR.
    rest_start : str
        The starting year of the restoration monitoring window.
    timestep : int, optional
        The timestep (years) in the restoration monitoring window
        (post rest_start) from which to evaluate absolute change.
        Default = 5.
    use_dist_avg : bool, optional
        Whether to use the average of the disturbance period to
        calculate the disturbance magnitude. Default = False.

    Returns
    -------
    rri_v : xr.DataArray
        DataArray containing the RRI value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    if timestep == 0:
        raise ValueError("timestep for RRI must be greater than 0.")
    dist_end = ra.restoration_start

    rest_post_tm1 = str(int(ra.restoration_start) + (timestep - 1))
    rest_post_t = str(int(ra.restoration_start) + timestep)
    rest_t_tm1 = [
        (date == pd.to_datetime(rest_post_tm1) or date == pd.to_datetime(rest_post_t))
        for date in ra.restoration_image_stack.coords["time"].values
    ]
    if any(rest_t_tm1) == 0:
        raise ValueError(
            f"timestep={timestep}, but ({ra.restoration_start}-1)+{timestep}={rest_post_tm1} or"
            f" {ra.restoration_start}+{timestep}={rest_post_t} not within time coordinates:"
            f" {ra.restoration_image_stack.coords['time'].values}. "
        )
    max_rest_t_tm1 = ra.restoration_image_stack.sel(time=rest_t_tm1).max(dim=["time"])

    if use_dist_avg:
        dist_pre_1 = str(int(ra.disturbance_start) - 1)
        dist_pre_2 = str(int(ra.disturbance_start) - 2)
        dist_pre_1_2 = [
            (date == pd.to_datetime(dist_pre_1) or date == pd.to_datetime(dist_pre_2))
            for date in ra.restoration_image_stack.coords["time"].values
        ]
        if any(dist_pre_1_2) == 0:
            raise ValueError(
                f"use_dist_avg={use_dist_avg} uses the 2 years prior to disturbance"
                f" start, but {ra.disturbance_start}-2={dist_pre_1} or"
                f" {ra.disturbance_start}-1={dist_pre_2} not within time coordinates:"
                f" {ra.restoration_image_stack.coords['time'].values}."
            )
        dist_pre = ra.restoration_image_stack.sel(time=dist_pre_1_2).max(dim=["time"])

        dist_s_e = [
            (date >= pd.to_datetime(ra.disturbance_start) and date <= pd.to_datetime(dist_end))
            for date in ra.restoration_image_stack.coords["time"].values
        ]
        dist_avg = ra.restoration_image_stack.sel(time=dist_s_e).mean(dim=["time"])

        zero_denom_mask = dist_pre - dist_avg == 0
        dist_pre = xr.where(zero_denom_mask, np.nan, dist_pre)
        rri_v = (max_rest_t_tm1 - dist_avg) / (dist_pre - dist_avg)
        # if dist_pre_1_2 or dist_s_e has length greater than one we will need
        # to squeeze the time dim
        try:
            rri_v = rri_v.squeeze("time")
        except KeyError:
            pass
    else:
        rest_0 = ra.restoration_image_stack.sel(time=ra.restoration_start).drop_vars("time")
        ra.disturbance_start = ra.restoration_image_stack.sel(time=ra.disturbance_start).drop_vars("time")
        dist_e = rest_0

        # Find if/where dist_start - dist_e == 0, set to NaN to avoid divide by zero
        # NaN - X will always be NaN, so no need to worry about the other side of the equation
        # Note: this is likely a safer way to do this that doesn't count on x / NaN. We could
        # mask where 0, set to num, and then use that mask aftwards to set to NaN.
        zero_denom_mask = ra.disturbance_start - dist_e == 0
        ra.disturbance_start = xr.where(zero_denom_mask, np.nan, ra.disturbance_start)
        rri_v = (max_rest_t_tm1 - rest_0) / (ra.disturbance_start - dist_e)
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
