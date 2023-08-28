import xarray as xr

from typing import Union, Tuple
from datetime import datetime

""" Methods for computing recovery targets """


def historic_average(
    stack: xr.DataArray, reference_date: Union[datetime, Tuple[datetime]]
) -> xr.DataArray:
    # TODO: should this just return a simple list?
    # TODO: should this take _just_ pd datetimeIndex?
    """
    Compute the average within a stack over all dimensions except time.

    Parameters
    ----------
    stack : xr.DataArray
        DataArray of images to derive historic average from. Must have at least
        4 labelled dimensions: "time", "band", "y" and "x". Optional 5th dimension
        "poly_id".

    Returns
    -------
    historic_average : xr.DataArray
        A 1D DataArray of historic average.
    """
    # TODO: force there to always be a band dimension.
    dims_to_average_over = tuple(
        item for item in stack.dims if (item != "band" and item != "poly_id")
    )
    if isinstance(reference_date, tuple):
        ranged_stack = stack.sel(time=slice(*reference_date))
    else:
        ranged_stack = stack.sel(time=slice(reference_date))
    historic_average = ranged_stack.mean(dim=dims_to_average_over, skipna=True)
    if "poly_id" in stack.dims:
        historic_average = historic_average.mean(dim="poly_id", skipna=True)
    
    historic_average = historic_average.assign_coords(band=stack.coords["band"]) # re-assign lost coords.
    return historic_average
