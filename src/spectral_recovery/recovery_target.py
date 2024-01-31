""" Methods for computing recovery targets """
from typing import Union, Tuple
from datetime import datetime

import xarray as xr


class MedianTarget:
    """ Callable median target method, parameterized by scale.
    
    Attributes
    ----------
    scale : {"polygon", "pixel"}
        The scale to compute target for. Either 'polygon' which results
        in one value per-band (median of the polygon(s) across time), or
        'pixel' which results in a value for each pixel per-band (median
        of each pixel across time).
        
    """
    def __init__(self, scale: str):
        self.scale = scale

    def __call__(
        self,
        stack: xr.DataArray,
        reference_date: Tuple[datetime],
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
            

        Returns
        -------
        median_t : xr.DataArray
            DataArray of the median recovery target. If scale="polygon", then median_t
            has dimensions "band" and optionally, "poly_id". If scale="pixel", has
            dimensions "band", "y" and "x" and optionally, "poly_id".

        """
        reference_window = stack.sel(time=slice(*reference_date))

        # Compute median sequentially
        median_t = reference_window.median(dim="time", skipna=True)
        
        # Additional median calculations based on scale and dimensions
        if self.scale == "polygon":
            median_t = median_t.median(dim=["y", "x"], skipna=True)
        if "poly_id" in stack.dims:
            median_t = median_t.median(dim="poly_id", skipna=True)

        # Re-assign lost band coords.
        median_t = median_t.assign_coords(band=stack.coords["band"])  
        return median_t


def windowed_median_target() -> xr.DataArray:
    """Compute the windowed median recovery target."""
    return NotImplementedError
