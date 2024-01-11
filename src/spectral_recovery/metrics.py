"""Methods for computing recovery metrics."""

import xarray as xr
import numpy as np
import pandas as pd

from spectral_recovery._utils import maintain_rio_attrs

# TODO: should methods that take 'percent' params not allow negative percent
# or greater than 100 values? Right now we just throw a ValueError. This avoids
# weird divides, so seems like the safest option, but maybe we should be more flexible?
NEG_TIMESTEP_MSG = "timestep cannot be negative."
VALID_PERC_MSP = "percent must be between 0 and 100."


@maintain_rio_attrs
def dnbr(
    image_stack: xr.DataArray,
    rest_start: str,
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
    try:
        rest_post_t = str(int(rest_start) + timestep)
        dnbr_v = (
            image_stack.sel(time=rest_post_t).drop_vars("time")
            - image_stack.sel(time=rest_start).drop_vars("time")
        ).squeeze("time")
    except KeyError as e:
        if int(rest_post_t) > year_dt(image_stack["time"].data.max(), int):
            raise ValueError(
                f"timestep={timestep}, but {rest_start}+{timestep}={rest_post_t} not"
                f" within time coordinates: {image_stack.coords['time'].values}. "
            ) from None
        raise e
    return dnbr_v


@maintain_rio_attrs
def yryr(
    image_stack: xr.DataArray,
    rest_start: str,
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

    rest_post_t = str(int(rest_start) + timestep)
    obs_post_t = image_stack.sel(time=rest_post_t).drop_vars("time")
    obs_start = image_stack.sel(time=rest_start).drop_vars("time")
    yryr_v = ((obs_post_t - obs_start) / timestep).squeeze("time")

    return yryr_v


@maintain_rio_attrs
def r80p(
    image_stack: xr.DataArray,
    rest_start: str,
    recovery_target: xr.DataArray,
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
        rest_post_t = image_stack["time"].data[-1]
    elif timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    elif percent <= 0 or percent > 100:
        raise ValueError(VALID_PERC_MSP)
    else:
        rest_post_t = str(int(rest_start) + timestep)
    r80p_v = (image_stack.sel(time=rest_post_t)).drop_vars("time") / (
        (percent / 100) * recovery_target
    )
    try:
        # if using the default timestep (the max/most recent),
        # the indexing will not get rid of the "time" dim
        r80p_v = r80p_v.squeeze("time")
    except KeyError:
        pass
    return r80p_v


@maintain_rio_attrs
def y2r(
    image_stack: xr.DataArray,
    rest_start: str,
    recovery_target: xr.DataArray,
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
    reco_target = recovery_target * (percent / 100)
    recovery_window = image_stack.sel(time=slice(rest_start, None))

    years_to_recovery = (recovery_window >= reco_target).argmax(dim="time", skipna=True)
    # Pixels with value 0 could be pixels that were recovered at the first timestep, or
    # pixels that never recovered (argmax returns 0 if all values are False).
    # Only the former are valid 0's, so set pixels that never recovered to NaN.
    zero_mask = years_to_recovery == 0
    recovered_at_zero = recovery_window.sel(time=rest_start) >= reco_target
    valid_zeros = zero_mask & recovered_at_zero
    valid_output = valid_zeros | (~zero_mask)

    y2r_v = years_to_recovery.where(valid_output, np.nan).drop_vars("time")
    try:
        y2r_v = y2r_v.squeeze("time")
    except KeyError:
        pass
    return y2r_v


@maintain_rio_attrs
def rri(
    image_stack: xr.DataArray,
    rest_start: str,
    dist_start: int,
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
    dist_end = rest_start

    rest_post_tm1 = str(int(rest_start) + (timestep - 1))
    rest_post_t = str(int(rest_start) + timestep)
    rest_t_tm1 = [
        (date == pd.to_datetime(rest_post_tm1) or date == pd.to_datetime(rest_post_t))
        for date in image_stack.coords["time"].values
    ]
    if any(rest_t_tm1) == 0:
        raise ValueError(
            f"timestep={timestep}, but ({rest_start}-1)+{timestep}={rest_post_tm1} or"
            f" {rest_start}+{timestep}={rest_post_t} not within time coordinates:"
            f" {image_stack.coords['time'].values}. "
        )
    max_rest_t_tm1 = image_stack.sel(time=rest_t_tm1).max(dim=["time"])

    if use_dist_avg:
        dist_pre_1 = str(int(dist_start) - 1)
        dist_pre_2 = str(int(dist_start) - 2)
        dist_pre_1_2 = [
            (date == pd.to_datetime(dist_pre_1) or date == pd.to_datetime(dist_pre_2))
            for date in image_stack.coords["time"].values
        ]
        if any(dist_pre_1_2) == 0:
            raise ValueError(
                f"use_dist_avg={use_dist_avg} uses the 2 years prior to disturbance"
                f" start, but {dist_start}-2={dist_pre_1} or"
                f" {dist_start}-1={dist_pre_2} not within time coordinates:"
                f" {image_stack.coords['time'].values}."
            )
        dist_pre = image_stack.sel(time=dist_pre_1_2).max(dim=["time"])

        dist_s_e = [
            (date >= pd.to_datetime(dist_start) and date <= pd.to_datetime(dist_end))
            for date in image_stack.coords["time"].values
        ]
        dist_avg = image_stack.sel(time=dist_s_e).mean(dim=["time"])

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
        rest_0 = image_stack.sel(time=rest_start).drop_vars("time")
        dist_start = image_stack.sel(time=dist_start).drop_vars("time")
        dist_e = rest_0

        # Find if/where dist_start - dist_e == 0, set to NaN to avoid divide by zero
        # NaN - X will always be NaN, so no need to worry about the other side of the equation
        # Note: this is likely a safer way to do this that doesn't count on x / NaN. We could
        # mask where 0, set to num, and then use that mask aftwards to set to NaN.
        zero_denom_mask = dist_start - dist_e == 0
        dist_start = xr.where(zero_denom_mask, np.nan, dist_start)
        rri_v = (max_rest_t_tm1 - rest_0) / (dist_start - dist_e)
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
