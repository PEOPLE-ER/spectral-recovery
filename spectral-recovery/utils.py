import pandas as pd 
import functools

from datetime import datetime, timezone
from typing import List, Union

def to_datetime(value: Union[str, List[str], datetime, pd.Timestamp]):
    """ Format year and year ranges to UTC datetime."""
    # TODO
    # convert to UTC if not in UTC, if no timezone then assume UTC
    # Convert a year range from X to Y as start-of-year X to end-of-year Y
    return value

def maintain_spatial_attrs(func):
    """ A wrapper for maintaining rioxarray spatial information (like 
    CRS and encoding) when performing xarray operations.
    """
    @functools.wraps(func)
    def wrapper_maintain_spatial_attrs(stack, *args, **kwargs):
        indice = func(stack, *args, **kwargs)
        indice.rio.write_crs(stack.rio.crs, inplace=True)
        indice.rio.update_encoding(stack.encoding, inplace=True)
        return indice
    return wrapper_maintain_spatial_attrs