import rioxarray
import Dask

import xarray as xr

from typing import Optional, Union, List, Dict
from shapely.geometry import Polygon



def spectral_recovery(
        composite_timeseries: xr.DataArray,
        restoration_poly: Polygon,
        reference_poly: Polygon,
        start_year: int, 
        end_year: int, 
        spectral_indices: Union[str, List[str]],
        metrics: Dict[str], 
        config_file: Optional[Dict[str]] = None,
        cluster_config: Optional[Dict[str]] = None

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
            TODO: Further discussions about expanding the role of reference polygons, 
            how we handle and/or interpret the polygons, etc.

        start_year:
            Relevant start year of restoration efforts.

        end_year:  
            End year of recovery metric computation.

        spectral_indices:
            Name or list of names of indices and/or bands to compute. Each named 
            index/band will have have `metrics` computed.
            - NDVI
            - NBR
            TODO: Further discussion surrounding short/long-term indices and
            good default choices.

        metrics: 
            The metrics to compute
            - % Recovered (user provided percentage)
            - Y2R (user provded percentage)
            TODO: Good defaults. More metrics to include. 

        config_file: 
            A description of inputs and (optional) outputs. 

        cluster_config:
            Configuration file for Dask Gateway.

        """
        if config_file:
                # read in inputs, if outputs present begin "continued" computation
                return NotImplementedError
        
        
        # if end_year not provided: end_year = last year in composite_timeseries
        
        # % recovered = 
        # compute slope/rate from start_year to end_year
        # 

        return NotImplementedError