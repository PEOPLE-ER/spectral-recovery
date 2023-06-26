import xarray as xr

from typing import Union, Tuple
from datetime import datetime

""" Methods for computing baselines for reference """


def historic_average(
    stack: xr.DataArray, reference_date: Union[datetime, Tuple[datetime]]
) -> xr.DataArray:
    # TODO: should this just return a simple list?
    # TODO: should this take _just_ pd datetimeIndex?
    """
    Compute the average within a stack over a time range.

    Parameters
    ----------
    stack : xr.DataArray
        DataArray of images to derive historic average from. Must be 3D
        and contain "time", "y" and "x" labelled dimensions.

    Returns
    -------
    baseline_avg : xr.DataArray
        A 1D DataArray of historic average.
    """
    if isinstance(reference_date, tuple):
        ranged_stack = stack.sel(time=slice(*reference_date))
    else:
        ranged_stack = stack.sel(time=slice(reference_date))
    print(ranged_stack.data.compute())
    baseline_avg = ranged_stack.mean(dim=["time", "y", "x"], skipna=True)
    return baseline_avg
