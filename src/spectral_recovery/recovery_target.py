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

    Will average over time, then y/x, then poly_id (if present). Resulting
    DataArray will have one target value per band.

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
    if isinstance(reference_date, list):
        ranged_stack = stack.sel(time=slice(*reference_date))
    else:
        ranged_stack = stack.sel(time=slice(reference_date))
    # NOTE: unexplained bug here if we take median over all dims at once. Instead, get time then y/x then poly_id.
    historic_average = ranged_stack.median(dim="time", skipna=True).median(dim=["y", "x"], skipna=True)

    if "poly_id" in stack.dims:
        historic_average = historic_average.median(dim="poly_id", skipna=True)

    historic_average = historic_average.assign_coords(
        band=stack.coords["band"]
    )  # re-assign lost coords.
    return historic_average


def windowed_polygon_average(
    stack: xr.DataArray, poly_id: str, window: int = 3
) -> xr.DataArray:
    return NotImplementedError