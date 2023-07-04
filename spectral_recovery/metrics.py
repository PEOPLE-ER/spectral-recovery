import xarray as xr
import numpy as np
import dask.array as da

from enum import Enum
from utils import maintain_spatial_attrs
from scipy import stats
from typing import Callable, Optional


class Metrics(Enum):
    percent_recovered = "percent_recovered"
    years_to_recovery = "years_to_recovery"

    def __str__(self) -> str:
        return self.name


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
    recovered = abs(eval_stack - baseline)
    return recovered / total_change


@maintain_spatial_attrs
def years_to_recovery(
    image_stack: xr.DataArray,
    baseline: xr.DataArray,
    percent: int = 80,
) -> xr.DataArray:
    """Per-pixel years-to-recovery

    Parameters
    ----------
    image_stack : xr.DataArray
        Timeseries of images to compute years-to-recovery across.

    # TODO: faster implementation located in `metrics_playground` module
    """
    reco_80 = baseline * (percent / 100)
    # theil_sen calls apply_ufunc along the time dimension so stack's
    # chunks need to contain the entire timestack before being passed
    y_vals = image_stack.chunk(dict(time=-1))
    x_vals = image_stack.time.dt.year

    ts = _theil_sen(y=y_vals, x=x_vals)
    y2r = (reco_80 - ts.sel(parameter="intercept")) / ts.sel(parameter="slope")
    # TODO: maybe return NaN if intercept + slope*curr_year is not recovered
    return y2r - x_vals[0]


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
    trajectory_func: Optional[Callable] = None
) -> xr.DataArray:
    if trajectory_func is not None:
        # Fit timeseries to trajectory and use fitted values for formula
        # fit trajectory here!
        raise NotImplementedError
    dNBR = restoration_stack.isel(time=4) - restoration_stack(time=0)
    return dNBR

@maintain_spatial_attrs
def RI(
    image_stack: xr.DataArray,
    restoration_index: int,
    trajectory_func: Optional[Callable] = None
) -> xr.DataArray:
    """
    Notes
    -----
    This implementation currently assumes that the disturbance period is 1 year long.
    # TODO: allow for multi-year disturbances?
    """
    if trajectory_func is not None:
        # Fit timeseries to trajectory and use fitted values for formula
        # fit trajectory here!
        raise NotImplementedError
    dNBR = ((image_stack.isel(time=restoration_index+5) - image_stack(time=restoration_index))
            / (image_stack.isel(time=restoration_index-1)) - image_stack(time=restoration_index))
    return dNBR
    