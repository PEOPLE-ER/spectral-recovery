import stackstac
import planetary_computer
import fiona 

import geopandas as gpd
import xarray as xr
import numpy as np

from shapely.geometry import Polygon, Point
from pystac_client import Client
from typing import Dict, List, Union
from itertools import chain

# get a stack, mask clouds, and median composite them
def get_STAC_items(
    collection: str,
    area_of_interest: Union[Polygon, Point],
    perc_covered_by_cloud: float,
    year_window: List[str],
    month_window: List[str],
    bands: List[str],
    resolution: int,
    epsg: int,
    property_filters: Dict,
    datestr: str = None
) -> xr.DataArray:
    """ Get STAC items and load into a lazy xarray.DataArray.

    Parameters
    ----------
    collection : {'sentinel-2-l2a', 'sentinel-1-rtc'}
    area_of_interest : Polygon or Point
    perc_covered_by_cloud : float
        The maximum amount of cloud cover allowed in
        in images.
    year_window : list of str
        The window of years to request images within.
    month_window : list of str
        The window of months to request images within.
    bands : list of str
        The bands to request.
    resolution : int
        Resolution (m) to set image stack to.
    epsg : int
        EPSG code to project image stack to.

    Returns
    -------
    stack : xarray.DataArray
        Requested image stack.
        
    Raises 
    ------
    UnconfiguredEnvironment
        If the Planetary Computer subscription key is not 
        set as an environemnt variable when the Sentinel-1
        RTC collection is requested.
    ValueError
        If the requested bands are not available as assets in 
        any or all of Items in the recieved ItemCollection.

    Notes
    -----
    Only STAC Items whose metadata contain ALL the passed key/value pairs
    defined in `property_filters` will be kept in the returned stack.

    """
    if collection == 'sentinel-1-rtc':
        if planetary_computer.settings.Settings.get().subscription_key is None:
            raise UnconfiguredEnvironment(
                f"Planetary Computer subscription key is not set but is required "
                f"to retrieve collection '{collection}'. To continue, restart the "
                f"current process then set the environment file using the "
                f"`planetarycomputer configure`CLI tool or directly set the environment "
                f"variable `PC_SDK_SUBSCRIPTION_KEY`."
            )
            
    api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    # TODO: update query dictionary logic based on PR !22's comments.
    if 'sentinel-2' in collection:
        query = {
                    "eo:cloud_cover": {
                        "lt": perc_covered_by_cloud
                    }
        }
    else:
        query = {}
        
    items_iterable = []
    if datestr is not None:
        search = api.search(
            collections=[collection],
            intersects=area_of_interest,
            datetime=datestr,
            query=query,
        )
        items = search.get_all_items()
        if property_filters is not None:
            # https://stackoverflow.com/questions/18170459
            items = [items[i] for i, item in enumerate(items)
                                  if property_filters.items() <= item.properties.items()]
        signed_items = [planetary_computer.sign(item).to_dict() for item in items]
    else:
        for year in year_window:
            if (len(month_window) == 1) or (month_window[0] == month_window[1]):
                daterange = f"{year}-{month_window[0]}"
            else:
                daterange = f"{year}-{month_window[0]}/{year}-{month_window[1]}"

            search = api.search(
                collections=[collection],
                intersects=area_of_interest,
                datetime=daterange,
                query=query,
            )
            items = search.get_all_items()
            if property_filters is not None:
                # https://stackoverflow.com/questions/18170459
                items = [items[i] for i, item in enumerate(items)
                                  if property_filters.items() <= item.properties.items()]
            items_iterable.append(items)

        chained_items = chain.from_iterable(items_iterable)
        signed_items = [planetary_computer.sign(item).to_dict()
                        for item in chained_items]

    if not signed_items:
        return None
 
    if 'sentinel-2' in collection:
        if "SCL" not in bands:
            bands = bands + ["SCL"]
    
    if 'landsat-c2-l2' in collection:
        if "QA_PIXEL" not in bands:
            bands = bands + ["QA_PIXEL"]
            
    bands_in_assets = [ 
        all(band in item['assets'].keys() for band in bands)
        for item in signed_items 
    ]
    if not all(bands_in_assets):
        # TODO: give more detail in this error if possible
        # e.g which bands aren't in the assets? 
        assets_missing_bands = (
            len(bands_in_assets) - np.count_nonzero(bands_in_assets)) - len(bands_in_assets)
        assets_bands = [signed_items]
        raise ValueError(
            f"{assets_missing_bands} out of {len(signed_items)} Items do not contain all "
            f"or some of the requested bands {bands}."
        )
        
    stackstac_kwargs = {
        "items" : signed_items,
        "assets" : bands,
        "epsg" : epsg,
        "chunksize" : "auto",
        "resolution" : resolution
    }
    if isinstance(area_of_interest, Polygon):
        stackstac_kwargs["bounds_latlon"] = area_of_interest.bounds
    stack = (
        stackstac.stack(**stackstac_kwargs).where(lambda x: x > 0, other=np.nan)
    )
        
    return stack

def landsat_qa_mask(stack: xr.DataArray):
    """ Mask clouds out of landsat stack (using qa band) """
    mask_bitfields = [1, 2, 3, 4]  # dilated cloud, cirrus, cloud, cloud shadow
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    qa = stack.sel(band="QA_PIXEL").astype("uint16")
    bad = qa & bitmask  # just look at those 4 bits
    masked = stack.where(bad == 0)

    return masked

def median_composite(stack: xr.DataArray) -> xr.DataArray:
    """ Composite time dimension via median.

    Parameters
    ----------
    stack : xr.DataArray
        4D (time, band, y, x) set of images to
        get median composite of.

    Returns
    -------
    median : xr.DataArray
        3D (band, y, x) median composite

    """
    median = stack.median(dim='time', skipna=True, keep_attrs=True)
    median.attrs['composite_method'] = 'median'

    return median


if __name__ == "__main__":
    fiona.drvsupport.supported_drivers['libkml'] = 'rw' # enable KML support which is disabled by default
    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

    saikuz = gpd.read_file("C:/Users/sarahvz/Downloads/SERNBC_Spectral_Recovery/SERNBC_Spectral_Recovery/Saikuz Restoration Plan area.kml").to_crs("EPSG:3005")
    saikuz_centroid = saikuz["geometry"].centroid
    saikuz.plot()
    # silvi = gpd.read_file("C:/Users/sarahvz/Downloads/BCGW_7113060B_1684427454413_1000/RSLT_FOREST_COVER_SILV_SVW.geojson")



    
    # stack = get_STAC_items(
    #     collection="landsat-c2-l2",
    #     area_of_interest=,
    #     perc_covered_by_cloud=30,
    #     year_window=["1995", "2021"],
    #     month_window=["07","08"]
    #     bands=[""],

    # )

    # Compute indices


# 1995 - 2021
# July 1st - September 1st
# NDVI and NBR