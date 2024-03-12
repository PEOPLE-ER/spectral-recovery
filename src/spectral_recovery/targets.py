""" Methods for computing recovery targets """

from inspect import signature
from typing import Union, Tuple
from datetime import datetime

import geopandas as gpd
import numpy as np

import xarray as xr


def _tight_clip_reference_stack(timeseries, restoration_polygon, reference_start, reference_end, reference_polygons):
    if reference_polygons is not None:
        # reference stack is the same as the restoration image stack
        reference_image_stack = timeseries.rio.clip(restoration_polygon.geometry.values)
    else:
        # reference image stack is clipped using the reference polygons
        # Need to make sure that each polygon is clipped seperately then stacked
        clipped_stacks = {}
        for i, row in reference_polygons.iterrows():
            polygon_stack = timeseries.rio.clip(
                gpd.GeoSeries(row.geometry).values
            )
            clipped_stacks[i] = polygon_stack

        reference_image_stack = xr.concat(
            clipped_stacks.values(),
            dim=pdIndex(clipped_stacks.keys(), name="poly_id"),
        )
    return reference_image_stack.sel(time=slice(reference_start, reference_end))

def _buffered_clip_reference_stack(timeseries, restoration_polygon, reference_start, reference_end, buffer):
    """"""
    tight_clip = timeseries.rio.clip(restoration_polygon.geometry.values)
    tight_x = tight_clip['x'].values
    tight_y = tight_clip['y'].values

    x_indices = np.searchsorted(timeseries['x'].values, tight_x)
    y_indices = np.searchsorted(timeseries['y'].values, tight_y)

    buffered_x_indices = np.clip(x_indices, buffer, timeseries.sizes['x'] - (buffer + 1))[0]
    buffered_y_indices = np.clip(y_indices, buffer, timeseries.sizes['y'] - (buffer + 1))[0]

    
    buffered_clip = timeseries[:, :, buffered_y_indices - buffer:buffered_y_indices + buffer + 1, buffered_x_indices - buffer:buffered_x_indices + buffer + 1]

    return buffered_clip




def compute_recovery_targets(timeseries, restoration_polygon, reference_start, reference_end, reference_polygons, method):
    
    if isinstance(method, MedianTarget):
        reference_image_stack = _tight_clip_reference_stack(timeseries, restoration_polygon, reference_start, reference_end)
        recovery_target = method(reference_image_stack)
    if isinstance(method, WindowedTarget):
        reference_image_stack = _buffered_clip_reference_stack(timeseries, restoration_polygon, reference_start, reference_end)

    return recovery_target

def _template_method(
    image_stack: xr.DataArray, reference_date: Tuple[datetime] | datetime
) -> xr.DataArray:
    """
    Template recovery target method.

    All valid target methods must follow this signature.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray of images. Must have [time, band, y, x] dimensions
        and optionally an additional poly_id dimension.
    reference_date : Union[datetime, Tuple[datetime]]
        The date or date range to compute the method over.

    Returns
    -------
    xr.DataArray
        The resulting recovery targets.

    """
    pass


expected_signature = signature(_template_method)


class MedianTarget:
    """Median target method parameterized on scale.

    The median target algorithm calculates a recovery target by
    sequentially computing the median over a specified time window from
    a stack of image data. The stack is expected to have dimensions for
    time, bands, and spatial coordinates. Additional median calculations
    are performed based on the scale ("polygon" or "pixel") and the
    presence of a "poly_id" dimension.

    Attributes
    ----------
    scale : {"polygon", "pixel"}
        The scale to compute target for. Either 'polygon' which results
        in one value per-band (median of the polygon(s) across time), or
        'pixel' which results in a value for each pixel per-band (median
        of each pixel across time).

    """

    def __init__(self, scale: str):
        if not ((scale == "polygon") or (scale == "pixel")):
            raise ValueError(f"scale must be 'polygon' or 'pixel' ('{scale}' provided)")
        self.scale = scale

    def __call__(
        self,
        reference_window: xr.DataArray,
        reference_date: Tuple[datetime] | datetime,
    ) -> xr.DataArray:
        """
        Median recovery target.

        Sequentially computes the median over time and, optionally, the
        spatial dimensions (x and y). If there is a "poly_id" dimension,
        representing each individual polygon in a multi-polygon reference
        input, then the median is automatically computed along that dimension
        after the time and space dimensions. This results in a single 
        target value for each band, based on the time, y, x and poly_id dims.

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
        # Compute median sequentially
        median_t = reference_window.median(dim="time", skipna=True)

        # Additional median calculations based on scale and dimensions
        # NOTE: scale is referenced from the containing scope make_median_target
        if self.scale == "polygon":
            median_t = median_t.median(dim=["y", "x"], skipna=True)
        if "poly_id" in image_stack.dims:
            median_t = median_t.median(dim="poly_id", skipna=True)

        # Re-assign lost band coords.
        median_t = median_t.assign_coords(band=image_stack.coords["band"])
        return median_t


class WindowedTarget():
    """Windowed recovery target method, parameterized on window size.

    The windowed method first computes the median along the time
    dimension and then for each pixel p in the restoration site
    polygon, computes the mean of a window of NxN pixels centred
    on pixel p, setting the mean to the recovery target value.


    Attributes
    ----------
    N : int
        Size of the window (NxN). Must be odd. Default is 3. 

    """
    def __init__(self, N: int = 3):
        if not isinstance(N, int):
            raise TypeError(f"N must be int not type {type(N)}")
        if N < 1:
            raise ValueError("N must be greater than or equal to 1.")
        if (N % 2) == 0:
            raise ValueError("N must be an odd int.")
        self.N = N
        
    def __call__(
        self,
        reference_window: xr.DataArray,
        ) -> xr.DataArray:
        """Compute the windowed mean recovery target."""
        median_t = reference_window.median(dim="time", skipna=True)

        windowed_mean = median_t.rolling({"y": self.N, "x": self.N}, center=True).mean()
        return windowed_mean
