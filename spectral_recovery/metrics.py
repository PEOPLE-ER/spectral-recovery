from ast import Str
import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd

from enum import Enum
from datetime import datetime
from spectral_recovery.utils import maintain_spatial_attrs
from scipy import stats
from typing import Callable, Optional


@maintain_spatial_attrs
def percent_recovered(
    eval_stack: xr.DataArray, baseline: xr.DataArray, event_obs: xr.DataArray
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
    total_change = abs(baseline - event_obs)  
    recovered = abs(eval_stack - event_obs)     
    return recovered / total_change

@maintain_spatial_attrs
def P80R(
    image_stack: xr.DataArray,
    rest_start: str,
    trajectory_func: Optional[Callable] = None,
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


@maintain_spatial_attrs
def YrYr(
    image_stack: xr.DataArray,
    rest_start: str,
    trajectory_func: Optional[Callable] = None,
):
    if trajectory_func is not None:
        # Fit timeseries to trajectory and use fitted values for formula
        # fit trajectory here!
        raise NotImplementedError
    
    dist_start = str((int(rest_start) - 1))
    dist_post_5 = str(int(dist_start) + 5)

    dist_post_5_val = image_stack.sel(time=dist_post_5)
    dist_val = image_stack.sel(time=dist_start)

    return dist_post_5_val - dist_val


@maintain_spatial_attrs
def years_to_recovery(
    image_stack: xr.DataArray,
    baseline: xr.DataArray,
    percent: int = 80,
    predictive: bool = False,
) -> xr.DataArray:
    """Per-pixel years-to-recovery

    Parameters
    ----------
    image_stack : xr.DataArray
        Timeseries of images to compute years-to-recovery across.

    # TODO: faster implementation located in `metrics_playground` module
    # TODO: decide on "undetermined" value (e.g not recovered, negative recovery)
    # NOTE: re #5: If current observations are not recovered but at a previous timestep
    # an observation reached the recovered state... do we report the # of years to the
    # previous step? or report nan?
    """
    reco_80 = baseline * (percent / 100)
    # _theil_sen calls apply_ufunc along the time dimension so stack's
    # chunks need to contain the entire timestack before being passed
    y_vals = image_stack.chunk(dict(time=-1))
    x_vals = image_stack.time.dt.year

    ts = _theil_sen(y=y_vals, x=x_vals)
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


def _new_linregress(y, x):
    """Wrapper around mstats.theilslopes for apply_ufunc usage"""
    slope, intercept, low_slope, high_slope = stats.mstats.theilslopes(y, x)
    return np.array([slope, intercept])


def _theil_sen(y, x):
    """Apply theil_sen slope regression across time on each pixel

    Parameters
    ----------
    y : xr.DataArray

    x : list of int

    Returns
    -------
    ts_reg : xr.DataArray
        DataArray of  theil-sen slope and intercept parameters for each
        pixel. 3D DataArray with "parameter", "y" and "x" labelled
        dimensions where "y" and "x" match input "y" and "x".

    """
    ts_dim_name = "parameter"
    ts_reg = xr.apply_ufunc(
        _new_linregress,
        y,
        x,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[ts_dim_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
        dask_gufunc_kwargs={"output_sizes": {ts_dim_name: 2}},
    )
    ts_reg = ts_reg.assign_coords({"parameter": ["slope", "intercept"]})
    return ts_reg


@maintain_spatial_attrs
def dNBR(
    restoration_stack: xr.DataArray,
    rest_start: str,
    trajectory_func: Optional[Callable] = None,
) -> xr.DataArray:
    """Delta-NBR

    Parameters
    ----------
    restoration_stack :
    trajectory_func : callable, optional

    """
    if trajectory_func is not None:
        # Fit timeseries to trajectory and use fitted values for formula
        # fit trajectory here!
        raise NotImplementedError

    rest_post_5 = str(int(rest_start) + 5)
    dNBR = (
        restoration_stack.sel(time=rest_post_5).drop_vars("time")
        - restoration_stack.sel(time=rest_start).drop_vars("time")
    ).squeeze("time")
    return dNBR


@maintain_spatial_attrs
def recovery_indicator(
    image_stack: xr.DataArray,
    rest_start: str,
    trajectory_func: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Notes
    -----
    This implementation currently assumes that the disturbance period is 1 year long.
    TODO: allow for multi-year disturbances?
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


@maintain_spatial_attrs
def NBRRegrowth(
    image_stack: xr.DataArray,
    rest_start: str,
    time_interval: int,
    trajectory_func: Optional[Callable] = None,
):
    
    if trajectory_func is not None:
        # Fit timeseries to trajectory and use fitted values for formula
        # fit trajectory here!
        raise NotImplementedError

    rest_post_5 = str(int(rest_start) + 5)
    interval_end = str(int(rest_start) + time_interval)
    interval_dates = [((date < pd.to_datetime(interval_end)) and date > pd.to_datetime(rest_start)) for date in image_stack.coords["time"].values]
    interval_avg = image_stack.sel(time=interval_dates).mean(dim=["y", "x"])
    # NOTE: no averaging happening because multi-year disturbances are not implemented yet
    dist_avg = image_stack.sel(time=rest_start).mean(dim=["y", "x"])

    
    return 
