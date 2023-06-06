import rioxarray
import dask

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Polygon
from typing import Optional, Union, List, Dict

from restoration_area import RestorationArea



def spectral_recovery(
        restoration_poly: Union[str, Polygon],
        restoration_year: int, 
) -> None:
        """
        The main calling function. Better doc-string incoming. 

        Parameters
        -----------
        restoration_poly_path : str
            Path to vector file containing polygon representing restoration area.

        restoration_year :
            Year of restoration event.

        """
        if isinstance(restoration_area, str):
            restoration_vectors = gpd.read_file(restoration_poly)
            if restoration_vectors.shape[0] != 1:
                return ValueError("Only one restoration polygon can be provided.")
        
            restoration_poly = restoration_vectors.iloc[0].geom
            restoration_area = RestorationArea(restoration_poly, restoration_year)
        else:
            restoration_area = restoration_poly


        np.random.seed(0)
        temperature = 15 + 8 * np.random.randn(3, 2, 2)
        lon = [[-99.83, -99.32], [-99.79, -99.23]]
        lat = [[42.25, 42.21], [42.63, 42.59]]
        time = pd.date_range("2014-09-06", periods=3)
        reference_time = pd.Timestamp("2014-09-05")
        test_stack = xr.DataArray(
        data=temperature,
        dims=["time", "y", "x"],
        coords=dict(
                lon=(["y", "x"], lat),
                lat=(["y", "x"], lon),
                time=time,
                reference_time=reference_time,
        ),
        attrs=dict(
                description="Ambient temperature.",
                units="degC",
        ),
        )
        restoration_area.clip_from_stack(test_stack)

        return