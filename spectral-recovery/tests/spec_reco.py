from typing import Optional, Union

import rioxarray
import Dask

import xarray as xr

from shapely.geometry import Polygon



def spectral_recovery(
        composite_timeseries: xr.DataArray,
        restoration_poly: Union[Polygon, str],
        reference_poly: Union[Polygon, str]
) -> int:
        """
        The main function. Better doc-string incoming. 


        Parameters
        -----------
        composite_timeseries:
            Stack of composite images ordered in ascending order of date. 

        restoration_poly:
            Polygon(s) representing the areas of interest for recovery
            assesment. Each polygon must contain a date attribute representing
            the date-of-interest (e.g year of disturbance)

        reference_poly:
            Polygon(s) representing the reference areas for recovery assesment. 
            Each polygon must contain a date attribute representing the date to 
            consider as reference.
            NOTE: Discussions about expanding the role of reference polygons, 
            how we handle the polygons, (Melissa is passionate and will bring
            justification). 

        start_year:
            Relevant start year of restoration efforts. (docs)
            - if break date provided then 

        end_year:  
            End year of recovery metric computation.
            - default to last year represented in composite_timeseries 

        spectral_indices:
            - NDVI
            - NBR
            NOTE: Dicussion surrounding short/long-term. Whether we 
            enfore it, how we define it, how to decide time ranges, etc.

        metrics: 
            The metrics to compute
            - Percentage index comparison (user provided percentage)
            - Y2R (user provded percentage)
            - Melissa will keep looking!

        
        
        
        config_file: 

        """
    return NotImplementedError