import xarray as xr

from typing import Union, Tuple, List
from datetime import datetime

""" Methods for computing recovery targets """


def median_target(
    stack: xr.DataArray, reference_date: Union[datetime, Tuple[datetime]], space: bool = True
) -> xr.DataArray:
    """
    Compute the median recovery target.

    Sequentially computes the median over time and, optionally, the spatial
    dimensions (x and y). If there is a "poly_id" dimension, then the median is
    automatically computed along that dimension after the time and space dimensions.
    
    Parameters
    ----------
    stack : xr.DataArray
        DataArray of images to derive historic average from. Must have at least
        4 labelled dimensions: "time", "band", "y" and "x" and optionally,
        "poly_id".
    reference_date : Union[datetime, Tuple[datetime]]
        The date or date range to compute the median over.
    space : bool
        If True, compute median over the y and x dimensions.

    Returns
    -------
    historic_average : xr.DataArray
        A 1D DataArray of historic average.

    """
    if isinstance(reference_date, list):
        ranged_stack = stack.sel(time=slice(*reference_date))
    else:
        ranged_stack = stack.sel(time=slice(reference_date))

    median_target = ranged_stack.median(dim="time", skipna=True)
    if space:
        median_target = median_target.median(dim=["y", "x"], skipna=True)
    if "poly_id" in stack.dims:
        median_target = median_target.median(dim="poly_id", skipna=True)

    median_target = median_target.assign_coords(
        band=stack.coords["band"]
    )  # re-assign lost coords.
    return median_target


def windowed_median_target(
    stack: xr.DataArray, poly_id: str, window: int = 3
) -> xr.DataArray:
    return NotImplementedError