import xarray as xr
import numpy as np
import pandas as pd

from spectral_recovery.utils import maintain_rio_attrs

# TODO: should methods that take 'percent' params not allow negative percent
# or greater than 100 values? Right now we just throw a ValueError. This avoids
# weird divides, so seems like the safest option, but maybe we should be more flexible?
NEG_TIMESTEP_MSG = f"timestep cannot be negative."
VALID_PERC_MSP = f"percent must be between 0 and 100."


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

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    try:
        rest_post_t = str(int(rest_start) + timestep)
        dNBR = (
            image_stack.sel(time=rest_post_t).drop_vars("time")
            - image_stack.sel(time=rest_start).drop_vars("time")
        ).squeeze("time")
    except KeyError as e:
        if int(rest_post_t) > year_dt(image_stack["time"].data.max(), int):
            raise ValueError(
                f"timestep={timestep}, but {rest_start}+{timestep}={rest_post_t} not"
                f" within time coordinates: {image_stack.coords['time'].values}. "
            )
        else:
            raise e
    return dNBR


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
    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    rest_post_t = str(int(rest_start) + timestep)
    obs_post_t = image_stack.sel(time=rest_post_t).drop_vars("time")
    obs_start = image_stack.sel(time=rest_start).drop_vars("time")
    YrYr = ((obs_post_t - obs_start) / timestep).squeeze("time")

    return YrYr


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

    """
    if timestep is None:
        rest_post_t = image_stack["time"].data[-1]
    elif timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    elif percent <= 0 or percent > 100:
        raise ValueError(VALID_PERC_MSP)
    else:
        rest_post_t = str(int(rest_start) + timestep)
    r80p = (image_stack.sel(time=rest_post_t)).drop_vars("time") / (
        (percent / 100) * recovery_target
    )
    try:
        # if using the default timestep (the max/most recent), the indexing will not get rid of the "time" dim
        r80p = r80p.squeeze("time")
    except KeyError:
        pass
    return r80p


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

    """
    if percent <= 0 or percent > 100:
        raise ValueError(VALID_PERC_MSP)
    reco_target = recovery_target * (percent / 100)
    post_rest = image_stack.sel(time=slice(rest_start, None))
    post_rest_years = post_rest["time"].values
    rest_window_count = np.arange(len(post_rest_years))

    recovered_pixels = post_rest.where(image_stack >= reco_target)
    # NOTE: the following code was my best attempt to get "find the first year that recovered"
    # working with xarray. Likely not the best way to do it, but can't figure anything else out right now.
    year_of_recovery = recovered_pixels.idxmin(dim="time")

    Y2R = xr.full_like(recovered_pixels[:, 0, :, :], fill_value=np.nan)
    for i, recovery_time in enumerate(rest_window_count):
        Y2R_t = year_of_recovery
        Y2R_t = Y2R_t.where(Y2R_t == post_rest_years[i]).notnull()
        Y2R_mask = Y2R.where(Y2R_t, False)
        Y2R = xr.where(Y2R_mask, recovery_time, Y2R)

    Y2R = Y2R.drop_vars("time")
    return Y2R


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
        rri = (max_rest_t_tm1 - dist_avg) / (dist_pre - dist_avg)
        # if dist_pre_1_2 or dist_s_e has length greater than one we will need to squeeze the time dim
        try:
            rri = rri.squeeze("time")
        except KeyError as e:
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
        rri = (max_rest_t_tm1 - rest_0) / (dist_start - dist_e)
        # if dist_pre_1_2 has length greater than one we will need to squeeze the time dim
        try:
            rri = rri.squeeze("time")
        except KeyError as e:
            pass
    return rri


def year_dt(dt, type: str = "int"):
    """Get int or str representation of year from datetime-like object."""
    # TODO: refuse to move forward if dt isn't datetime-like
    try:
        dt_dt = pd.to_datetime(dt)
        year = dt_dt.year
    except ValueError:
        raise ValueError(f"Unable to get year {type} from {dt} of type {type(dt)}")
    if type == "str":
        return str(year)
    else:
        return year
