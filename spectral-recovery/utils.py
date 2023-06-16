import functools

import pandas as pd 
import numpy as np

from osgeo import gdal, ogr, os, osr
from datetime import datetime, timezone
from typing import List, Union

def to_datetime(value: Union[str, List[str], datetime, pd.Timestamp]):
    """ Format year and year ranges to UTC datetime."""
    # TODO
    # convert to UTC if not in UTC, if no timezone then assume UTC
    # Convert a year range from X to Y as start-of-year X to end-of-year Y
    return value

def maintain_spatial_attrs(func):
    """ A wrapper for maintaining rioxarray crs/encoding info.
    
    Takes rioxarray spatial information (crs and encoding) from an
    input DataArray and reassigns/updates the information onto an 
    output DataArray returned by `func`, a function which performs 
    operations that lose spatial information on DataArrays.
    
    """
    @functools.wraps(func)
    def wrapper_maintain_spatial_attrs(stack, *args, **kwargs):
        """
        Parameters
        ----------
        stack : xr.DataArray
            The DataArray object whose spatial attributes will be
            maintained/passed onto the output object

        Returns
        --------
        indice : xr.DataArray
            DataArray object returned by func with spatial attrs
        """
        indice = func(stack, *args, **kwargs)
        indice.rio.write_crs(stack.rio.crs, inplace=True)
        indice.rio.update_encoding(stack.encoding, inplace=True)
        return indice
    return wrapper_maintain_spatial_attrs


def array2raster(newRasterfn,
                 rasterOrigin,
                 pixelWidth,
                 pixelHeight,
                 array,
                 epsg
                 ):
    
    # array = array[::-1] # reverse array so the tif looks like the array
    cols = array.shape[-2]
    rows = array.shape[-1]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn,
                              cols,
                              rows,
                              array.shape[0],
                              gdal.GDT_Byte)
    outRaster.SetGeoTransform(
        (originX, pixelWidth, 0, originY, 0, pixelHeight)
        )
    for i in range(array.shape[0]): # write each band
        print(i)
        outband = outRaster.GetRasterBand(i+1)
        outband.WriteArray(array[i,:,:])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(epsg)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    return 


# if __name__ == "__main__":
#     rasterOrigin = (-123.25745,45.43013)
#     pixelWidth = 30
#     pixelHeight = 30
#     newRasterfn = 'test.tif'
#     epsg = 4326
#     array = np.array([[[1000]],
#                       [[1000]],
#                       [[1000]],
#                       [[1000]],
#                       [[10]],
#                       [[20]],
#                       [[30]],
#                       [[40]],
#                       [[45]],
#                       [[50]],
#                       [[55]],
#                       [[60]]
#                       ])
    # array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,epsg)