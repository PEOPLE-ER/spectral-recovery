import geopandas as gpd
import pandas as pd

from typing import Dict, List


# TODO: allow users to pass attribute col names for date cols
def read_restoration_polygons(
        path: str,
        dist_rest_years: Dict[int, List[int]] = None,
    ):
    """Read restoration polygons

    A loose wrapper of the geopandas.read_file function. If
    check_dates=True then this function will check that the
    vector file at path contains 2 or 4 columns containing
    str or int values (the restoration site dates).

    Parameters
    ----------
    path : str
        path to restoration polygon vector file

    """
    # Read the vector file and check there is only one polygon
    restoration_polygons = gpd.read_file(path)
    if dist_rest_years:
        # Add the passed dates into gpd cols
        dates = {
                "dist_start": [],
                "rest_start": [],
        }
        # Check that dates make sense
        for polyid, years in dist_rest_years.items():
            disturbance_start, restoration_start = years
            if disturbance_start >= restoration_start:
                raise ValueError(
                "Disturbance start year cannot be greater than or equal to the restoration start year"
                f" ({disturbance_start} >= {restoration_start})"
            )
            dates["dist_start"].append(disturbance_start)
            dates["rest_start"].append(restoration_start)
        
        restoration_polygons = restoration_polygons.assign(**dates)
    
    # Dates must be in order: dist, rest then geom 
    restoration_polygons = restoration_polygons[["dist_start", "rest_start", "geometry"]]
    # Look at the dates within the geodataframe (data from attribute table)
    dates_frame = pd.DataFrame(restoration_polygons.drop(columns="geometry"))
    if len(dates_frame.columns) < 2:
        raise ValueError(
            "Attribute table must have at least 2 columns containing disturbance start"
            f" year and restoration start year ({len(dates_frame.columns)} column"
            " given)"
        )
    # Check that years are either str or int types
    types = dates_frame.dtypes
    for column_name, data_type in types.items():
        if data_type != "int64" and data_type != "object" :
            raise ValueError(
                f"Date fields must be type int or str (given {data_type}) in field"
                f" {column_name})"
            )
        

    return restoration_polygons