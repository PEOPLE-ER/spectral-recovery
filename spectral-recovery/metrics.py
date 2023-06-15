import xarray as xr
import numpy as np

from typing import Callable
from enum import Enum
from utils import maintain_spatial_attrs
from scipy import stats


class Metrics(Enum):
    percent_recovered = "percent_recovered"
    years_to_recovery = "years_to_recovery"

    def __str__(self) -> str:
        return self.name


@maintain_spatial_attrs
def percent_recovered(
        stack: xr.DataArray,
        baseline: xr.DataArray,
        event: xr.DataArray
    ) -> xr.DataArray:
    """ Per-pixel percent recovery

    Recovery computation based on a baseline measurement, a restoration 
    event measure, and the current/relevant observations.

    """
    total_change = abs(baseline-event)
    recovered = abs(stack-baseline)
    return recovered / total_change


@maintain_spatial_attrs
def years_to_recovery(
    stack: xr.DataArray,
    baseline: xr.DataArray
) -> xr.DataArray:
    reco_80 = baseline * 0.80
    # print(baseline.data.compute())
    ts = theil_sen(stack.chunk(dict(time=-1)), stack.time.dt.year)
    # print(ts.data.compute())
    temp = reco_80 - stack / ts[0][0][0][0]
    return temp[0,:,0,0] # TODO: this is insane. Fix the dims before computing.


def new_linregress(x, y):
    # Wrapper around scipy mstats.theilslopes to use in apply_ufunc
    print(f"ts input: {x} and {y}")
    slope, intercept, low_slope, high_slope = stats.mstats.theilslopes(x, y)
    return np.array([slope, intercept, low_slope, high_slope])


def theil_sen(spectral, time):

    return xr.apply_ufunc(
        new_linregress,
        spectral,
        time,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[["parameter"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=['float64'],
        dask_gufunc_kwargs={"output_sizes": {"parameter": 4}},
        )