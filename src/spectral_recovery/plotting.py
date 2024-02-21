
from typing import List, Dict

import geopandas as gpd
import xarray as xr

from spectral_recovery.restoration import RestorationArea
from spectral_recovery.targets import MedianTarget
from spectral_recovery.indices import compute_indices

# TODO: Refactor. Bring plot_spectral_trajectory into this module.
def plot_spectral_trajectory(
        timeseries_data: xr.DataArray,
        restoration_polygons: gpd.GeoDataFrame,
        indices: List[str],
        reference_polygons: gpd.GeoDataFrame = None,
        index_constants: Dict[str, int] = {},
        recovery_target_method = MedianTarget(scale="polygon"),
        path: str = None
):
    """ Plot the spectral trajectory of the restoration polygon
    
    Parameters
    ----------
    timeseries_data : xr.DataArray
        Timeseries data (annual composites)
    restoration_polygons : gpd.GeoDataFrame
        Restoration polygon and dates
    indices : list of str
        Indices to visualize trajectory for
    reference_polygons : gpd.GeoDataFrame, optional
        The refernce polygons to compute recovery target with
    indices_constants : dict
        Constant values for indices
    recovery_target_method : callable
        Recovery target method to derive recovery target values
    
    """
    
    indices_stack = compute_indices(image_stack=timeseries_data, indices=indices, constants=index_constants)
    restoration_area = RestorationArea(
        restoration_polygon=restoration_polygons,
        reference_polygons=reference_polygons,
        composite_stack=indices_stack,
        recovery_target_method=recovery_target_method
    )
    if path:
        restoration_area.plot_spectral_trajectory(path=path)
    else:
        restoration_area.plot_spectral_trajectory()

    
