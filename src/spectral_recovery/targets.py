""" Methods for computing recovery targets """


import geopandas as gpd
import numpy as np
import xarray as xr

from inspect import signature
from typing import Union, Tuple
from datetime import datetime
from pandas import Index as pdIndex

class BufferError(ValueError):
    def __init__(self, message, y_back, y_front, x_back, x_front):
        self.y_back = y_back
        self.y_front = y_front
        self.x_back = x_back
        self.x_front = x_front
        super().__init__(message)


def _tight_window_time_clip(timeseries_data, site, reference_start, reference_end):
    """Get spatial and temporal clip for all polygons in site."""
    if len(site) == 1:
        reference_image_stack = timeseries_data.rio.clip(site.geometry.values)
    else:
        clipped_stacks = {}
        for i, row in site.iterrows():
            polygon_stack = timeseries_data.rio.clip(
                gpd.GeoSeries(row.geometry).values
            )
            clipped_stacks[i] = polygon_stack

        reference_image_stack = xr.concat(
            clipped_stacks.values(),
            dim=pdIndex(clipped_stacks.keys(), name="poly_id"),
        )
    return reference_image_stack.sel(time=slice(reference_start, reference_end))

def _buffered_window_time_clip(timeseries, site, reference_start, reference_end, buffer):
    """ Get buffered spatial clip and temporal clip of timeseries data."""
    tight_clip = timeseries.rio.clip(site.geometry.values)
    tight_x = tight_clip['x'].values
    tight_y = tight_clip['y'].values

    x_indices = np.searchsorted(timeseries['x'].values, tight_x)
    y_indices = np.searchsorted(-timeseries['y'].values, -tight_y)

    y_back_step = int(y_indices[0] - buffer)
    y_front_step = int(y_indices[-1] + buffer)

    x_back_step = int(x_indices[0] - buffer)
    x_front_step = int(x_indices[-1] + buffer)

    req_padding = [0,0,0,0]
    if y_back_step < 0:
        req_padding[1] = abs(y_back_step)
    if y_front_step > (len(timeseries.y.values) - 1):
        req_padding[0] = (y_front_step - (len(timeseries.y.values) - 1))
    if x_back_step < 0:
        req_padding[2] = abs(x_back_step)
    if x_front_step > (len(timeseries.x.values) - 1):
        req_padding[3] = (x_front_step - (len(timeseries.x.values) - 1))
    
    if not all([p == 0 for p in req_padding]):
        raise BufferError(f"Buffer exceeds boundaries, padding required to use rolling window of size {(buffer*2)+1}", *req_padding)

    buffered_clip = timeseries[:, :, y_back_step:y_front_step + 1, x_back_step:x_front_step + 1]

    return buffered_clip.sel(time=slice(reference_start, reference_end))

def median_target(
        polygon: gpd.GeoDataFrame,
        timeseries_data: xr.DataArray,
        reference_start: str,
        reference_end: str,
        scale: str
    ):
    """Median target method parameterized on scale.

    Sequentially computes the median over time and, optionally, the
    spatial dimensions (x and y). If there is a "poly_id" dimension,
    representing each individual polygon in a multi-polygon reference
    input, then the median is automatically computed along that dimension
    after the time and space dimensions. This results in a single 
    target value for each band, based on the time, y, x and poly_id dims.

    Parameters
    ----------
    polygon : gpd.GeoDataFrame
        The polygon/area to compute a recovery target for.
    timeseries_data : xr.DataArrah
        The timeseries of indices to derive the recovery target from.
        Must contain band, time, y, and x dimensions.
    reference_start : str
        Start year of reference window. Must exist in timeseries_data's
        time coordinates.
    reference_end : 
        End year of reference window. Must exist in timeseries_data's
        time coordinates.
    scale : {"polygon", "pixel"}
        The scale to compute target for. Either 'polygon' which results
        in one value per-band (median of the polygon(s) across time), or
        'pixel' which results in a value for each pixel per-band (median
        of each pixel across time).

        if not ((scale == "polygon") or (scale == "pixel")):
            raise ValueError(f"scale must be 'polygon' or 'pixel' ('{scale}' provided)")
        self.scale = scale

    Returns
    -------
    median_t : xr.DataArray
        DataArray of the median recovery target. If scale="polygon", then median_t
        has dimensions "band" and optionally, "poly_id". If scale="pixel", has
        dimensions "band", "y" and "x" and optionally, "poly_id".

    """
    # Clip timeseries data to polygon(s) and time dim
    target_data = _tight_window_time_clip(
        timeseries_data=timeseries_data,
        site=polygon, 
        reference_start=reference_start,
        reference_end=reference_end
    )

    # Compute median sequentially
    target_data = timeseries_data.sel(time=slice(reference_start, reference_end))
    median_time = target_data.median(dim="time", skipna=True)

    # Additional median calculations based on scale and dimensions
    # NOTE: scale is referenced from the containing scope make_median_target
    if scale == "polygon":
        median_time = median_time.median(dim=["y", "x"], skipna=True)
    if "poly_id" in timeseries_data.dims:
        median_time = median_time.median(dim="poly_id", skipna=True)

    # Re-assign lost band coords.
    median_target = median_time.assign_coords(band=target_data.coords["band"])
    return median_target

    
def window_target(polygon, timeseries_data, reference_start, reference_end, N: int = 3, na_rm=False):
        """Windowed recovery target method, parameterized on window size.

        The windowed method first computes the median along the time
        dimension and then for each pixel p in the restoration site
        polygon, computes the mean of a window of NxN pixels centred
        on pixel p, setting the mean to the recovery target value.

        Implementation is based on raster focal method in R.


        Parameters
        ----------
        polygon : gpd.GeoDataFrame
            The polygon/area to compute a recovery target for.
        timeseries_data : xr.DataArrah
            The timeseries of indices to derive the recovery target from.
            Must contain band, time, y, and x dimensions.
        reference_start : str
            Start year of reference window. Must exist in timeseries_data's
            time coordinates.
        reference_end : 
            End year of reference window. Must exist in timeseries_data's
            time coordinates.
        N : int
            Size of the window (NxN). Must be odd. Default is 3. 

        """
        if not isinstance(N, int):
            raise TypeError("N must be int.")
        if N < 1:
            raise ValueError("N must be greater than or equal to 1.")
        if (N % 2) == 0:
            raise ValueError("N must be an odd int.")
        if not isinstance(na_rm, bool):
            raise TypeError("na_rm must be boolean.")

        target_data = _buffered_window_time_clip(
            site=polygon, 
            timeseries=timeseries_data, 
            reference_start=reference_start, 
            reference_end=reference_end,
            buffer=(N-1)/2,
        )

        median_time = target_data.median(dim="time", skipna=True)

        if na_rm:
            # Only 1 non-NaN value is required to set a value.
            min_periods = 1
        else:
            # All values in the window must be non-NaN 
            min_periods = None
        windowed_mean = median_time.rolling({"y": N, "x": N}, center=True, min_periods=min_periods).mean()
        return windowed_mean
    