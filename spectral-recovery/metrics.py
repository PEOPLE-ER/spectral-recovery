import xarray as xr
import numpy as np

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

    """
    total_change = abs(baseline-event)
    recovered = abs(stack-baseline)
    return recovered / total_change


@maintain_spatial_attrs
def years_to_recovery(
    stack: xr.DataArray,
    baseline: xr.DataArray,
    percent: int = 80,
    curr_year = int
) -> xr.DataArray:
    """ Years-to-recovery

    
    """
    reco_80 = baseline * (percent / 100)
    print(baseline.data.compute())
    # theil_sen calls apply_ufunc along the time dimension so stack's 
    # chunks need to contain the entire timestack before being passed
    ts = theil_sen(y=stack.chunk(dict(time=-1)), x=stack.time.dt.year)
    y2r = ((reco_80 - ts.sel(parameter="intercept")) 
           / ts.sel(parameter="slope"))
    return y2r - curr_year


def new_linregress(y, x):
    """ Wrapper around  mstats.theilslopes for apply_ufunc usage """
    slope, intercept, low_slope, high_slope = stats.mstats.theilslopes(y, x)
    print(slope, intercept)
    return np.array([slope, intercept])


def theil_sen(y, x):
    """ Apply theil_sen slope regression across time on each pixel
    
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
        new_linregress,
        y,
        x,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[ts_dim_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=['float64'],
        dask_gufunc_kwargs={
            "output_sizes": {ts_dim_name: 2}
            },
        )
    ts_reg = ts_reg.assign_coords({"parameter": ["slope", "intercept"]})
    return ts_reg