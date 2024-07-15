import geopandas as gpd
import pandas as pd
import numpy as np

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
    if not dist_rest_years:
        if "rest_start" not in list(restoration_polygons) or "dist_start" not in list(
            restoration_polygons
        ):
            raise ValueError(
                "Missing disturbance and restoration years. Must pass year values to `dist_rest_years` param or as polygon attributes in the vector file."
            )
    else:
        # Check if the given keys actually reference polygons
        for polyid in dist_rest_years.keys():
            if polyid not in restoration_polygons.index.values.tolist():
                raise ValueError("polygon id {polyid} is not found in {path}.")

        restoration_polygons["dist_start"] = 0
        restoration_polygons["rest_start"] = 0
        # Check that dates make sense
        for polyid, years in dist_rest_years.items():
            disturbance_start, restoration_start = years
            if disturbance_start >= restoration_start:
                raise ValueError(
                    "Disturbance start year cannot be greater than or equal to the restoration start year"
                    f" ({disturbance_start} >= {restoration_start})"
                )
            restoration_polygons.loc[polyid, "dist_start"] = disturbance_start
            restoration_polygons.loc[polyid, "rest_start"] = restoration_start
        # Check that all polygons were given dates
        if (restoration_polygons["dist_start"] == 0).any() or (
            restoration_polygons["rest_start"] == 0
        ).any():
            raise ValueError(
                "Missing dist/rest start years for some polygons. Please provide a mapping for each polygon."
            )

    # Dates must be in order: dist, rest then geom
    restoration_polygons = restoration_polygons[
        ["dist_start", "rest_start", "geometry"]
    ]
    # Look at the dates within the geodataframe (data from attribute table)
    dates_frame = pd.DataFrame(restoration_polygons.drop(columns="geometry"))
    # Check that years are either str or int types
    types = dates_frame.dtypes
    for column_name, data_type in types.items():
        if data_type != "int64":
            raise ValueError(
                f"Date fields must be type int in"
                f" {column_name} field. Given {data_type}."
            )

    return restoration_polygons
