"""Utility functions for spectral-recovery."""

import functools
import spyndex as spx
import xarray as xr

from rioxarray.exceptions import MissingCRS


def maintain_rio_attrs(func: callable) -> callable:
    """A wrapper for maintaining rioxarray crs/encoding info.

    Rioxarray information (nodata, CRS, etc.) is lost through
    operations that create new instances of an xarray object
    because new accessors (e.g rio) are created for every instance.

    This method ensures tnhat a returned xarray object has the
    same rio CRS and encodig as the input xarray object.

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
        result : xr.DataArray
            DataArray object returned by func with spatial attrs

        Notes
        -----
        The first argument passed to a wrapped function (keyword or
        non-keyword) must be the xarray DataArray whose rio info
        is to be maintained.

        """
        kwarg_vals = list(kwargs.values())
        # Take the first xarray arg
        arg_da = [isinstance(arg, xr.DataArray) for arg in args]
        kwarg_da = [isinstance(kwv, xr.DataArray) for kwv in kwarg_vals]
        if sum(arg_da + kwarg_da) > 1:
            epsgs = []
            for i, val in enumerate(arg_da):
                if val:
                    crs = args[i].rio.crs
                    epsgs.append(crs)
            for i, val in enumerate(kwarg_da):
                if val:
                    crs = kwarg_vals[i].rio.crs
                    epsgs.append(crs)
            if not epsgs.count(epsgs[0]) == len(epsgs):
                raise ValueError(
                    f"Ambiguous input for wrapper. CRS on xarray.DataArray inputs do not match."
                )
        for i, val in enumerate(arg_da):
            if val:
                xarray_obj = args[i]
        for i, val in enumerate(kwarg_da):
            if val:
                xarray_obj = kwarg_vals[i]

        result = func(*args, **kwargs)
        try:
            result.rio.write_crs(xarray_obj.rio.crs, inplace=True)
            result.rio.update_encoding(xarray_obj.encoding, inplace=True)
        except AttributeError as ae:
            if isinstance(result, dict):
                for i, elem in result.items():
                    result[i] = elem.rio.write_crs(xarray_obj.rio.crs, inplace=True)
                    result[i] = elem.rio.update_encoding(
                        xarray_obj.encoding, inplace=True
                    )
        except MissingCRS as mcrs:
            # TODO: add warning log here?
            pass
        return result

    return wrapper_maintain_rio_attrs


def common_and_long_to_short(standard):
    """Dict of short and common names to standard names

    Notes
    -----
    This manually changes the G1, RE1, RE2, and RE3 common
    names to green1, rededge1, rededge2, and rededge3 respectively
    to be less ambiguous. This means that the common names returned
    will be slightly different than those used in spyndex.

    """
    # make 'green' and 'rededge' common names unambiguous
    spx.bands["G1"].common_name = "green1"
    spx.bands["RE1"].common_name = "rededge1"
    spx.bands["RE2"].common_name = "rededge2"
    spx.bands["RE3"].common_name = "rededge3"

    common_and_short = {}
    for band in standard:
        common_and_short[spx.bands[band].short_name] = band
        common_and_short[spx.bands[band].common_name] = band
    return common_and_short


def _platforms_from_band(band_object):
    """Get list of platform names supported by each band"""
    platforms = []
    for p in [
        "sentinel2a",
        "sentinel2b",
        "landsat4",
        "landsat5",
        "landsat7",
        "landsat8",
        "landsat9",
        "modis",
        "planetscope",
    ]:
        try:
            platforms.append(getattr(band_object, p).platform)
        except AttributeError:
            continue
    return platforms
