import xarray as xr
import numpy as np


from typing import Callable, Optional

from spectral_recovery.trajectory import per_pixel_theil_sen
from spectral_recovery.utils import maintain_rio_attrs


# TODO: generalize to take image_stack, rest_start, and baseline. Get event_obs within func.
@maintain_rio_attrs
def percent_recovered(
    image_stack: xr.DataArray, recovery_target: xr.DataArray, event_obs: xr.DataArray
) -> xr.DataArray:
    """Per-pixel percent recovery

    Parameters
    ----------
    image_stack : xr.DataArray
        The 4D DataArray (time, band, y, x) of images over which to
        evaluate percent recovered.
    recovery_target : xr.DataArray
        The DataArray containing the recovery targets. Dimensions
        must be able to broadcast to `eval_stack` and `event_obs`.
    event_obs : xr.DataArray
        The DataArray of images for the restoration event (indexed from
        `image_stack`) from which we are measuring recovery.

    Returns
    -------
    P80R : xr.DataArray
        3D (band, y, x) DataArray containing Y2R for each pixel. The band
        coordinate will contain "Y2R" label. Will either be predictive Y2R
        or non-predictive Y2R. For non-predictive Y2R, nan values represents
        pixels that have not hit percent recovered yet.

    """
    total_change = abs(recovery_target - event_obs)
    recovered = abs(image_stack - event_obs)
    return recovered / total_change


@maintain_rio_attrs
def years_to_recovery(
    image_stack: xr.DataArray,
    recovery_target: xr.DataArray,
    percent: int = 80,
    predictive: bool = False,
) -> xr.DataArray:
    """Per-pixel years-to-recovery

    Parameters
    ----------
    image_stack : xr.DataArray
        4D (time, band, y, x) stack of images to compute Y2R over.
    recovery_target : xr.DataArray
        1D (band) DataArray containing target recovery values for each band.
    percent : int
        The percent of `recovery_target` to consider "recovered".
    predictive :
        A boolean indicating whether or not to use predictive Y2R. If True,
        predictive Y2R is used. If False, traditional Y2R is used.

    Returns
    -------
    Y2R : xr.DataArray
        3D (band, y, x) DataArray containing Y2R for each pixel. The band
        coordinate will contain "Y2R" label. Will either be predictive Y2R
        or non-predictive Y2R. For non-predictive Y2R, nan values represents
        pixels that have not hit percent recovered yet.

    """
    # TODO: generalize this for multiple trajectory methods
    reco_80 = recovery_target * (percent / 100)
    # _theil_sen calls apply_ufunc along the time dimension so stack's
    # chunks need to contain the entire timestack before being passed
    y_vals = image_stack.chunk(dict(time=-1))
    x_vals = image_stack.time.dt.year

    ts = per_pixel_theil_sen(data_array=y_vals, time=x_vals)
    y2r = (reco_80 - ts.sel(parameter="intercept")) / ts.sel(parameter="slope")

    predictive_y2r = y2r - x_vals[0]
    total_years_recovering = len(x_vals) - 1
    if predictive:
        return predictive_y2r
    else:
        non_predictive_y2r = xr.where(
            predictive_y2r > total_years_recovering, np.nan, predictive_y2r
        )
        return non_predictive_y2r.drop_vars("time")


@maintain_rio_attrs
def dNBR(
    image_stack: xr.DataArray,
    rest_start: str,
    trajectory_func: Optional[Callable] = None,
) -> xr.DataArray:
    """Delta-NBR

    Parameters
    ----------
    image_stack : xr.DataArray
        4D (time, band, y, x) stack of images to compute dNBR over.
    rest_start : xr.DataArray
        Date of restoration event. Value must be within the time dimension
        coordinates of `image_stack` param.
    trajectory_func : callable, optional
        The function/method used to compute TS trajectory. Otherwise,
        metric method is computed on the raw values available in `image_stack`

    Returns
    -------
    dNBR : xr.DataArray
        3D (band, y, x) DataArray containing dNBR for each pixel. The band
        coordinate will contain "dNBR" label.
    """
    if trajectory_func is not None:
        # Fit timeseries to trajectory and use fitted values for formula.
        # Something like(?):
        # >> image_stack = trajectory_func(...)
        raise NotImplementedError
    # TODO: make this date increment/decrement easier or more reliable.
    rest_post_5 = str(int(rest_start) + 5)
    dNBR = (
        image_stack.sel(time=rest_post_5).drop_vars("time")
        - image_stack.sel(time=rest_start).drop_vars("time")
    ).squeeze("time")
    return dNBR


@maintain_rio_attrs
def recovery_indicator(
    image_stack: xr.DataArray,
    rest_start: str,
    trajectory_func: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Parameters
    ----------
    image_stack : xr.DataArray
        4D (time, band, y, x) stack of images to compute RI over.
    rest_start : xr.DataArray
        Date of restoration event. Value must be within the time dimension
        coordinates of `image_stack` param.
    trejectory_func : callable, optional
        The function/method used to compute TS trajectory. Otherwise,
        metric method is computed on the raw values available in `image_stack`.

    Returns
    -------
    RI : xr.DataArray
        3D (band, y, x) DataArray containing RI for each pixel. The band
        coordinate will contain "RI" label.
    Notes
    -----
    This implementation currently assumes that the disturbance period is 1 year long.
    Meaning, the parameter `rest_start` is the start of the restoration and the
    start of the disturbance is exactly one-year prior.

    """
    if trajectory_func is not None:
        # Fit timeseries to trajectory and use fitted values for formula
        # fit trajectory here!
        raise NotImplementedError

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
