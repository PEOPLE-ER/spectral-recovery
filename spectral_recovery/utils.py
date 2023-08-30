import functools

import pandas as pd

from datetime import datetime, timezone
from typing import List, Union


def to_datetime(value: Union[str, List[str], datetime, pd.Timestamp]):
    """Format year and year ranges to UTC datetime."""
    # TODO
    # convert to UTC if not in UTC, if no timezone then assume UTC
    # Convert a year range from X to Y as start-of-year X to end-of-year Y
    return value


def maintain_rio_attrs(func):
    """A wrapper for maintaining rioxarray crs/encoding info.

    Rioxarray information (nodata, CRS, etc.) is lost through
    operations that create new instances of an xarray object
    because new accessors (e.g rio) are created for every instance.

    This method ensures that a returned xarray object has the
    same rio CRS and encoding as the input xarray object.

    """

    @functools.wraps(func)
    def wrapper_maintain_rio_attrs(*args, **kwargs):
        """
        Returns
        --------
        indice : xr.DataArray
            DataArray object returned by func with spatial attrs

        Notes
        -----
        The first argument passed to a wrapped function (keyword or
        non-keyword) must be the xarray DataArray whose rio info
        is to be maintained.

        """
        # Take the first argument
        if args:
            xarray_obj = args[0]
        else:
            # NOTE: this only works for Python 3.6+ where dicts keep insertion order by default
            xarray_obj = next(iter(kwargs.values()))
        indice = func(*args, **kwargs)
        indice.rio.write_crs(xarray_obj.rio.crs, inplace=True)
        indice.rio.update_encoding(xarray_obj.encoding, inplace=True)
        return indice

    return wrapper_maintain_rio_attrs
