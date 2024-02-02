""" Methods for computing recovery targets """
from typing import Union, Tuple
from datetime import datetime

import xarray as xr


def make_median_target(scale: str):
    """ Parameterize a median target method based on desired scale.

    The median target algorithm calculates a recovery target by
    sequentially computing the median over a specified time window from
    a stack of image data. The stack is expected to have dimensions for
    time, bands, and spatial coordinates. Additional median calculations
    are performed based on the scale ("polygon" or "pixel") and the
    presence of a "poly_id" dimension. 

    Parameters
    ----------
    scale : {"polygon", "pixel"}
        The scale to compute target for. Either 'polygon' which results
        in one value per-band (median of the polygon(s) across time), or
        'pixel' which results in a value for each pixel per-band (median
        of each pixel across time).
    
    Returns
    -------
    median_target : callable
        Median target method parameterized by scale.
        
    """

    def median_target(
        image_stack: xr.DataArray,
        reference_date: Tuple[datetime] | datetime,
    ) -> xr.DataArray:
        """
        Closure for median recovery target.

        Sequentially computes the median over time and, optionally, the
        spatial dimensions (x and y). If there is a "poly_id" dimension,
        then the median is automatically computed along that dimension
        after the time and space dimensions.

        Parameters
        ----------
        image_stack : xr.DataArray
            DataArray of images to derive historic average from. Must have at least
            4 labelled dimensions: "time", "band", "y" and "x" and optionally,
            "poly_id".
        reference_date : Union[datetime, Tuple[datetime]]
            The date or date range to compute the median over.
            

        Returns
        -------
        median_t : xr.DataArray
            DataArray of the median recovery target. If scale="polygon", then median_t
            has dimensions "band" and optionally, "poly_id". If scale="pixel", has
            dimensions "band", "y" and "x" and optionally, "poly_id".

        """
        try:
            reference_window = image_stack.sel(time=slice(*reference_date))
        except TypeError:
            reference_window = image_stack.sel(time=slice(reference_date))

        # Compute median sequentially
        median_t = reference_window.median(dim="time", skipna=True)
        
        # Additional median calculations based on scale and dimensions
        # NOTE: scale is referenced from the containing scope make_median_target
        if scale == "polygon":
            median_t = median_t.median(dim=["y", "x"], skipna=True)
        if "poly_id" in image_stack.dims:
            median_t = median_t.median(dim="poly_id", skipna=True)

        # Re-assign lost band coords.
        median_t = median_t.assign_coords(band=image_stack.coords["band"])  
        return median_t
    
    return median_target


def make_windowed_target() -> xr.DataArray:
    """Compute the windowed median recovery target."""
    return NotImplementedError
