import xarray as xr
import numpy as np
import pandas as pd

from spectral_recovery.utils import maintain_rio_attrs

# NOTE: percent_recovered will likely be removed before publishing
@maintain_rio_attrs
def percent_recovered(
    eval_stack: xr.DataArray, recovery_target: xr.DataArray, event_obs: xr.DataArray
) -> xr.DataArray:
    """Per-pixel percent recovery

    Parameters
    ----------
    eval_stack : xr.DataArray
        The stack of images to evaluate percent recovered over.

    baseline : xr.DataArray
        The baseline for recovery. Dimensions must be able to broadcast
        to `eval_stack` and `event_obs`.

    event_obs : xr.DataArray
        The image/values of the event which we are measuring recovery
        from. x and y dimensions must match `eval_stack`.

    """
    total_change = abs(recovery_target - event_obs)  
    recovered = abs(eval_stack - event_obs)     
    return recovered / total_change

#TODO: P80R should be using a target recovery value like the others
@maintain_rio_attrs
def P80R(
    image_stack: xr.DataArray,
    rest_start: str,
) -> xr.DataArray:
    """ Extent (percent) that trajectory has reached 80% of pre-disturbance value.

    Modified metric from Y2R. Value equal to 1 indicates 80% has been reached and value more or less than 1
    indicates more or less than 80% has been reached.

    
    """
    dist_start = str((int(rest_start) - 1))
    pre_rest = [date < pd.to_datetime(dist_start) for date in image_stack.coords["time"].values]
    post_rest = [date >= pd.to_datetime(dist_start) for date in image_stack.coords["time"].values]
    pre_rest_avg = image_stack.sel(time=pre_rest).mean(dim=["y", "x"])
    post_rest_max = image_stack.sel(time=post_rest).max(dim=["y", "x"])

    return post_rest_max / (0.8 * pre_rest_avg)


@maintain_rio_attrs
def YrYr(
    image_stack: xr.DataArray,
    rest_start: str,
):
    
    dist_start = str((int(rest_start) - 1))
    dist_post_5 = str(int(dist_start) + 5)

    dist_post_5_val = image_stack.sel(time=dist_post_5)
    dist_val = image_stack.sel(time=dist_start)

    return dist_post_5_val - dist_val


@maintain_rio_attrs
def Y2R(
    image_stack: xr.DataArray,
    recovery_target: xr.DataArray,
    rest_start: str,
    rest_end: str,
    percent: int = 80,
) -> xr.DataArray:
    """Per-pixel years-to-recovery

    Parameters
    ----------
    image_stack : xr.DataArray
        Timeseries of images to compute years-to-recovery across.

    """
    reco_target = recovery_target * (percent / 100)
    post_rest = image_stack.sel(time=slice(rest_start, rest_end))
    post_rest_years = post_rest["time"].values
    possible_years_to_recovery = np.arange(len(post_rest_years))

    recovered_pixels = post_rest.where(image_stack >= reco_target)
    # NOTE: the following code was my best attempt to get "find the first year that recovered"
    # working with xarray. Likely not the best way to do it, but can't figure anything else out now.
    year_of_recovery = recovered_pixels.idxmax(dim="time")

    Y2R = xr.full_like(recovered_pixels[:,0,:,:], fill_value=np.nan)
    for i, recovery_time in enumerate(possible_years_to_recovery):
        Y2R_t = year_of_recovery
        Y2R_t = Y2R_t.where(Y2R_t == post_rest_years[i]).notnull()
        Y2R_mask = Y2R.where(Y2R_t, False)
        Y2R = xr.where(Y2R_mask, recovery_time, Y2R)

    Y2R = Y2R.drop_vars("time")
    return Y2R


@maintain_rio_attrs
def dNBR(
    restoration_stack: xr.DataArray,
    rest_start: str,
) -> xr.DataArray:
    """Delta-NBR

    Parameters
    ----------
    restoration_stack :
    trajectory_func : callable, optional

    """
    rest_post_5 = str(int(rest_start) + 5)
    dNBR = (
        restoration_stack.sel(time=rest_post_5).drop_vars("time")
        - restoration_stack.sel(time=rest_start).drop_vars("time")
    ).squeeze("time")
    return dNBR


@maintain_rio_attrs
def RI(
    image_stack: xr.DataArray,
    rest_start: str,
) -> xr.DataArray:
    """
    Notes
    -----
    This implementation currently assumes that the disturbance period is 1 year long.
    TODO: allow for multi-year disturbances?
    """
    rest_post_5 = str(int(rest_start) + 5)
    dist_start = str(int(rest_start) - 1)
    dist_end = rest_start
    RI = (
        (
            image_stack.sel(time=rest_post_5).drop_vars("time")
            - image_stack.sel(time=rest_start)
        ).drop_vars("time")
        / (
            image_stack.sel(time=dist_start).drop_vars("time")
            - image_stack.sel(time=dist_end).drop_vars("time")
        )
    ).squeeze("time")
    return RI


@maintain_rio_attrs
def NBRRegrowth(
    image_stack: xr.DataArray,
    rest_start: str,
    time_interval: int,
):
    raise NotImplementedError
    rest_post_5 = str(int(rest_start) + 5)
    interval_end = str(int(rest_start) + time_interval)
    interval_dates = [((date < pd.to_datetime(interval_end)) and date > pd.to_datetime(rest_start)) for date in image_stack.coords["time"].values]
    interval_avg = image_stack.sel(time=interval_dates).mean(dim=["y", "x"])
    # NOTE: no averaging happening because multi-year disturbances are not implemented yet
    dist_avg = image_stack.sel(time=rest_start).mean(dim=["y", "x"])

    
    return 
