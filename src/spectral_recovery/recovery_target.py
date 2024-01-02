""" Methods for computing recovery targets """
from typing import Union, Tuple
from datetime import datetime

import xarray as xr


# TODO: split this function into per-pixel and per-polygon functions
def median_target(
    stack: xr.DataArray,
    reference_date: Union[datetime, Tuple[datetime]],
    space: bool = True,
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
    median_t : xr.DataArray
        DataArray of the median recovery target. If space=True, then median_t
        has dimensions "band" and optionally, "poly_id". If space=False, has
        dimensions "band", "y" and "x" and optionally, "poly_id".

    """
    if isinstance(reference_date, list):
        ranged_stack = stack.sel(time=slice(*reference_date))
    else:
        ranged_stack = stack.sel(time=slice(reference_date))

    median_t = ranged_stack.median(dim="time", skipna=True)
    if space:
        median_t = median_t.median(dim=["y", "x"], skipna=True)
    if "poly_id" in stack.dims:
        median_t = median_t.median(dim="poly_id", skipna=True)

    median_t = median_t.assign_coords(
        band=stack.coords["band"]
    )  # re-assign lost coords.
    return median_t


def windowed_median_target() -> xr.DataArray:
    """Compute the windowed median recovery target."""
    return NotImplementedError
