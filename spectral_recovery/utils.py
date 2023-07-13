import functools

import pandas as pd
import numpy as np

from osgeo import gdal, ogr, os, osr
from datetime import datetime, timezone
from typing import List, Union, Tuple


def to_datetime(value: Union[str, List[str], datetime, pd.Timestamp]):
    """Format year and year ranges to UTC datetime."""
    # TODO
    # convert to UTC if not in UTC, if no timezone then assume UTC
    # Convert a year range from X to Y as start-of-year X to end-of-year Y
    return value


def maintain_spatial_attrs(func):
    """A wrapper for maintaining rioxarray crs/encoding info.

    Rioxarray information (nodata, CRS, etc.) is lost through
    operations that create new instances of an xarray object
    because new accessors are created for every instance.

    This method ensures that a returned xarray object has the
    same rio CRS and encoding that the input xarray object had.

    """

    @functools.wraps(func)
    def wrapper_maintain_spatial_attrs(*args, **kwargs):
        """
        Returns
        --------
        indice : xr.DataArray
            DataArray object returned by func with spatial attrs

        Notes
        -----
        The first argument passed to a wrapped function (unname or
        keyword arguments) must be the xarray object whose rio
        information is being maintained.

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

    return wrapper_maintain_spatial_attrs


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, epsg):
    cols = array.shape[-2]
    rows = array.shape[-1]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(newRasterfn, cols, rows, array.shape[0], gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    for i in range(array.shape[0]):  # write each band
        print(i)
        outband = outRaster.GetRasterBand(i + 1)
        outband.WriteArray(array[i, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(epsg)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    return


def set_band_descriptions(filepath, bands):
    """
    filepath: path/virtual path/uri to raster
    bands:    ((band, description), (band, description),...)
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc in bands.items():
        rb = ds.GetRasterBand(band + 1)
        rb.SetDescription(desc)
    del ds


if __name__ == "__main__":
    pass
    # tif = "../test_bigger.tif"
    # list_bands = dict(
    #     zip(
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #         [
    #             "2008",
    #             "2009",
    #             "2011",
    #             "2010",
    #             "2012",
    #             "2013",
    #             "2014",
    #             "2015",
    #             "2016",
    #             "2017",
    #             "2018",
    #             "2019",
    #         ],
    #     )
    # )
    # print(list_bands)
    # set_band_descriptions(tif, list_bands)

    # rasterOrigin = (-123.25745, 45.43013)
    # pixelWidth = 30
    # pixelHeight = 30
    # for i, year in enumerate([2008, 2009, 2010, 2011]):
    #     newRasterfn = f"../test_{year}_time.tif"
    #     epsg = 4326
    #     array = np.array(
    #         [
    #             np.ones((200, 200)) * 1 * i + 1,
    #             np.ones((200, 200)) * 2 * i + 1,
    #             np.ones((200, 200)) * 3 * i + 1,
    #         ]
    #     )
    #     print(array.shape)
    #     print(array)
    #     array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, epsg)
