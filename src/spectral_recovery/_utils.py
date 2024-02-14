"""Utility functions for spectral-recovery.

Currently, only contains decorators for maintaining rio attributes
when performing operations on xarray objects.

"""

import functools
import json
import spyndex as spx

from rioxarray.exceptions import MissingCRS
from prettytable import PrettyTable, ALL


def maintain_rio_attrs(func: callable) -> callable:
    """A wrapper for maintaining rioxarray crs/encoding info.

    Rioxarray information (nodata, CRS, etc.) is lost through
    operations that create new instances of an xarray object
    because new accessors (e.g rio) are created for every instance.

    This method ensures that a returned xarray object has the
    same rio CRS and encoding as the input xarray object.

    Notes
    -----
    Similar to https://github.com/pydata/xarray/pull/2482, but specifically
    for a subset of rio attributes to be maintained.

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
        try:
            indice.rio.write_crs(xarray_obj.rio.crs, inplace=True)
        except MissingCRS:
            # TODO: add warning log here?
            pass
        indice.rio.update_encoding(xarray_obj.encoding, inplace=True)
        return indice

    return wrapper_maintain_rio_attrs

def common_and_long_to_short(standard):
    """Dict of short and common names to standard names"""
    common_and_short = {}
    for band in standard:
        common_and_short[spx.bands[band].short_name] = band
        common_and_short[spx.bands[band].common_name] = band
    return common_and_short

def bands_pretty_table():
    """ Create a PrettyTable of all bands (names and id info).
    
    Returns
    -------
    band_table : PrettyTable
        table for displaying short names, common names, long
        names, wavelength and platform info for bands in the
        spyndex package.
        
    """
    band_table = PrettyTable()
    band_table.hrules=ALL
    band_table.field_names = [
        "Standard/Short Name",
        "Common Name",
        "Long Name",
        "Wavelength (min, max)",
        "Platforms",
    ]
    for st in list(spx.bands):
        platforms = _format_platforms(_platforms_from_band(spx.bands[st]), 3)
        band_table.add_row([
            st,
            spx.bands[st].common_name,
            spx.bands[st].long_name,
            f"{spx.bands[st].min_wavelength, spx.bands[st].max_wavelength}",
            platforms,
        ])
    return band_table


def _platforms_from_band(band_object):
    """Get list of platform names supported by each band"""
    platforms = []
    for p in ["sentinel2a", "sentinel2b", "landsat4", "landsat5", "landsat7", "landsat8", "landsat9", "modis", "planetscope"]:
        try:
            platforms.append(getattr(band_object, p).platform)
        except AttributeError:
            continue
    return platforms


def _format_platforms(comment_list, max_items_on_line):
    """Format list of platform strs into prettier multi-line str"""
    ACC_length = 0
    formatted_comment = ""
    for word in comment_list:
        if ACC_length + 1 < max_items_on_line:
            formatted_comment = formatted_comment + word + ", "
            ACC_length = ACC_length + 1
        else:
            formatted_comment = formatted_comment + "\n" + word + ", "
            ACC_length =+ 1
    return formatted_comment
