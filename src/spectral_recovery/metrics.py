import xarray as xr
import numpy as np
import pandas as pd

from spectral_recovery.utils import maintain_rio_attrs


@maintain_rio_attrs
def dNBR(
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
    rest_post_t = str(int(rest_start) + timestep)
    dNBR = (
        image_stack.sel(time=rest_post_t).drop_vars("time")
        - image_stack.sel(time=rest_start).drop_vars("time")
    ).squeeze("time")
    return dNBR


@maintain_rio_attrs
def YrYr(
    image_stack: xr.DataArray,
    rest_start: str,
    timestep: int,
):
    """Per-pixel YrYr.

    The average annual recovery rate relative to a fixed time intervald
    during the restoration monitoring window. The default is the first 5
    years of the restoration window, however this can be changed by specifying
    the index value at a specific time step (Ri).

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
    rest_post_t = str(int(rest_start) + timestep)
    obs_post_t = image_stack.sel(time=rest_post_t).drop_vars("time")
    obs_start = image_stack.sel(time=rest_start).drop_vars("time")
    d_t = (
        pd.to_datetime(image_stack.sel(time=rest_post_t)["time"].data).year
        - pd.to_datetime(image_stack.sel(time=rest_start)["time"].data).year
    )

    YrYr = ((obs_post_t - obs_start) / d_t).squeeze("time")

    return YrYr


@maintain_rio_attrs
def R80P(
    image_stack: xr.DataArray,
    rest_start: str,
    recovery_target: xr.DataArray,
    percent: int,
    max_4_5: bool = False,
) -> xr.DataArray:
    """Per-pixel R80P.

    The extent to which the trajectory has reached 80% of the recovery
    target value. The metric commonly uses the maximum value from the
    4th or 5th year of restoration window to show the extent to which a
    pixel has reached 80% of the target value 5 years into the restoration
    window, however for monitoring purposes, this tool uses the current
    time step or last year of the time series to provide up to date recovery
    progress. 80% of the recovery target value is the default, however this
    can be changed by modifying the value of P.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray of images over which to compute per-pixel dNBR.
    rest_start : str
        The starting year of the restoration monitoring window.
    recovery_target : xr.DataArray
        Recovery target values. Must be broadcastable to image_stack.
    percent: int, optional
        Percent of recovery to compute recovery against. Default = 80.
    max_4_5: bool, optional
        Flag indicating whether or not to compute using maximum value
        from the 4th or 5th year of restoration window.

    """
    max_year = image_stack["time"].data.max()
    if max_4_5:
        rest_post_4 = str(int(rest_start) + 4)
        rest_post_5 = str(int(rest_start) + 5)
        if int(rest_post_4) > year_dt(max_year) or int(rest_post_5) > year_dt(max_year):
            raise ValueError(
                f"Max year in provided image_stack is {image_stack['time'].data.max()} but need {rest_post_4} and {rest_post_5}."
            )
        rest_4_5 = [
            (date == pd.to_datetime(rest_post_4) or date == pd.to_datetime(rest_post_5))
            for date in image_stack.coords["time"].values
        ]
        r80p = (
            (image_stack.sel(time=rest_4_5).max(dim=["y", "x"])).drop_vars("time")
            / ((percent / 100) * recovery_target)
        ).squeeze("time")
    else:
        r80p = (
            (image_stack.sel(time=max_year)).drop_vars("time")
            / ((percent / 100) * recovery_target)
        ).squeeze("time")

    return r80p


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

    Y2R = xr.full_like(recovered_pixels[:, 0, :, :], fill_value=np.nan)
    for i, recovery_time in enumerate(possible_years_to_recovery):
        Y2R_t = year_of_recovery
        Y2R_t = Y2R_t.where(Y2R_t == post_rest_years[i]).notnull()
        Y2R_mask = Y2R.where(Y2R_t, False)
        Y2R = xr.where(Y2R_mask, recovery_time, Y2R)

    Y2R = Y2R.drop_vars("time")
    return Y2R


@maintain_rio_attrs
def dNBR(
    image_stack: xr.DataArray,
    rest_start: str,
    timestep: int = 5,
) -> xr.DataArray:
    """Delta-NBR

    Parameters
    ----------
    restoration_stack :
    trajectory_func : callable, optional

    """
    rest_post_t = str(int(rest_start) + timestep)
    dNBR = (
        image_stack.sel(time=rest_post_t).drop_vars("time")
        - image_stack.sel(time=rest_start).drop_vars("time")
    ).squeeze("time")
    return dNBR


@maintain_rio_attrs
def RI(
    image_stack: xr.DataArray,
    rest_start: str,
    timestep: int = 5,
) -> xr.DataArray:
    """
    Notes
    -----
    This implementation currently assumes that the disturbance period is 1 year long.
    TODO: allow for multi-year disturbances?
    """
    rest_post_t = str(int(rest_start) + timestep)
    dist_start = str(int(rest_start) - 1)
    dist_end = rest_start
    RI = (
        (
            image_stack.sel(time=rest_post_t).drop_vars("time")
            - image_stack.sel(time=rest_start)
        ).drop_vars("time")
        / (
            image_stack.sel(time=dist_start).drop_vars("time")
            - image_stack.sel(time=dist_end).drop_vars("time")
        )
    ).squeeze("time")
    return RI