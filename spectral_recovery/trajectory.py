import xarray as xr
import numpy as np

from scipy import stats
from typing import List


def per_pixel_theil_sen(data_array: xr.DataArray, time: List[int]):
    """Apply theil_sen slope regression across time over each pixel

    Parameters
    ----------
    data_array : xr.DataArray
        Data to compute per-pixel theil-sen values for. DataArray must
        contain "time", "y", and "x" coordinate dimensions.

    time : list of int
        Time values. Length must be equal to the length of "time" 
        dimension on `data_array`.

    Returns
    -------
    ts_reg : xr.DataArray
        3D (parameter, y, x) DataArray of  theil-sen slope and intercept 
        parameters for each pixel. Parameter dimension will contain both
        slope and intercept t-s values for each pixel.

    """
    ts_dim_name = "parameter"
    ts_reg = xr.apply_ufunc(
        _new_linregress,
        data_array,
        time,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[ts_dim_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
        dask_gufunc_kwargs={"output_sizes": {ts_dim_name: 2}},
    )
    ts_reg = ts_reg.assign_coords({"parameter": ["slope", "intercept"]})
    return ts_reg

def _new_linregress(y, x):
    """Wrapper around mstats.theilslopes for apply_ufunc usage"""
    slope, intercept, low_slope, high_slope = stats.mstats.theilslopes(y, x)
    return np.array([slope, intercept])