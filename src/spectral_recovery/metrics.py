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
    rest_start: str,
    recovery_target: xr.DataArray,
    percent: int = 80,
) -> xr.DataArray:
    """Per-pixel Y2R.

    The length of time taken (in time steps/years) for a given pixel to
    reach 80% of its recovery target value. The percent can be modified
    by changing the value of P.

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

    Notes
    -----
    If a pixel P at timestep X has value V_x=(percent * recovery_target)
    but then at timestep X+i has value V_{xi}<(percent * recovery_target),
    this implementation of Y2R will return X.

    """
    reco_target = recovery_target * (percent / 100)
    post_rest = image_stack.sel(time=slice(rest_start, None))
    post_rest_years = post_rest["time"].values
    rest_window_count = np.arange(len(post_rest_years))

    recovered_pixels = post_rest.where(image_stack >= reco_target)
    # NOTE: the following code was my best attempt to get "find the first year that recovered"
    # working with xarray. Likely not the best way to do it, but can't figure anything else out right now.
    year_of_recovery = recovered_pixels.idxmax(dim="time")

    Y2R = xr.full_like(recovered_pixels[:, 0, :, :], fill_value=np.nan)
    for i, recovery_time in enumerate(rest_window_count):
        Y2R_t = year_of_recovery
        Y2R_t = Y2R_t.where(Y2R_t == post_rest_years[i]).notnull()
        Y2R_mask = Y2R.where(Y2R_t, False)
        Y2R = xr.where(Y2R_mask, recovery_time, Y2R)

    Y2R = Y2R.drop_vars("time")
    return Y2R


@maintain_rio_attrs
def RRI(
    image_stack: xr.DataArray,
    rest_start: str,
    timestep: int = 5,
    use_dist_avg: bool = False,
) -> xr.DataArray:
    """Per-pixel RRI.

    A modified version of the commonly used RI, the RRI accounts for
    noise in trajectory by using the maximum from the 4th or 5th year
    in monitoring window. The metric relates recovery magnitude to
    disturbance magnitude, and is the change in index value in 4 or 5
    years divided by the change due to disturbance.

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

    """
    dist_start = str(int(rest_start) - 1)
    min_year = image_stack["time"].data.min()
    max_year = image_stack["time"].data.max()

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
    rest_4_5 = image_stack.sel(time=rest_4_5).max(dim=["y", "x"])

    if use_dist_avg:
        dist_pre_1 = str(int(dist_start) - 1)
        dist_pre_2 = str(int(dist_start) - 2)
        if int(dist_pre_1) < year_dt(min_year) or int(dist_pre_2) < year_dt(min_year):
            raise ValueError(
                f"Min year in provided image_stack is {image_stack['time'].data.min()} but need {dist_pre_1} and {dist_pre_2}."
            )
        dist_1_2 = [
            (date == pd.to_datetime(dist_pre_1) or date == pd.to_datetime(dist_pre_2))
            for date in image_stack.coords["time"].values
        ]
        dist_pre = image_stack.sel(time=dist_1_2).max(dim=["y", "x"])
        dist_s_e = [
            (date >= pd.to_datetime(dist_start) or date <= pd.to_datetime(rest_start))
            for date in image_stack.coords["time"].values
        ]
        dist_avg = image_stack.sel(time=dist_s_e).mean(dim=["y", "x"])

        rri = ((rest_4_5.max() - dist_avg) / (dist_pre - dist_avg)).squeeze("time")
    else:
        rest_0 = image_stack.sel(time=rest_start).drop_vars("time")
        dist_e = rest_0
        dist_start = image_stack.sel(time=dist_start).drop_vars("time")

        rri = ((rest_4_5.max() - rest_0) / (dist_start - dist_e)).squeeze("time")
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
